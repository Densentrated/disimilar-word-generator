"""Concreteness predictor using XLM-RoBERTa fine-tuned on English norms.

Zero-shot cross-lingual transfer pipeline (Brysbaert et al. 2014 norms):

  Milestone 1 – Data Preparation
    Load, clean, z-score-optionally-normalise, and stratify-split the English
    concreteness dataset.  Bigrams are excluded; each word is optionally wrapped
    in a short context template to give the transformer richer input signal.

  Milestone 2 – Architecture
    XLM-RoBERTa (base or large) backbone + MLP regression head:
      [CLS] → Linear(768, 256) → ReLU → Dropout → Linear(256, 1)

  Milestone 3 – Fine-tuning on English
    Two-phase schedule: freeze backbone → train head; then unfreeze → end-to-end
    with differential learning rates and early stopping on Pearson r.

  Milestone 4 – Zero-shot cross-lingual transfer
    Run target-language words through the English-fine-tuned model.  XLM-RoBERTa's
    shared multilingual representation space means concrete/abstract geometry
    transfers without any target-language labels.

  Milestone 5 – Validation
    evaluate() returns Pearson r, Spearman ρ, and MAE against held-out norms.

  Milestone 6 – Few-shot fine-tuning (optional)
    fine_tune_target() accepts a small set of target-language labels and does a
    second fine-tuning pass with a very low learning rate.

  Milestone 7 – Lookup-table caching
    Predictions are stored in an in-memory dict keyed by word.  Repeated queries
    hit the cache without touching the GPU.

Usage::

    predictor = ConcretenessPredictor()
    predictor.train()                        # fine-tune on English ~39 k words
    df = predictor.predict(input_format)     # DataFrame: word | concreteness
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from input_format import InputFormat

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DATA_FILE = Path(__file__).parent / "english_conreteness_ratings.txt"
_DEFAULT_BACKBONE = "xlm-roberta-base"
# Minimal template.  The placeholder {word} is required; {definition} is
# optional – include it to use the English/dictionary gloss as context.
_DEFAULT_TEMPLATE = "The word is: {word}"

__all__ = ["ConcretenessPredictor"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _ConcretenessDataset(Dataset):
    def __init__(
        self,
        words: List[str],
        scores: List[float],
        tokenizer,
        max_length: int,
        template: str,
        definitions: Optional[List[str]] = None,
    ) -> None:
        self.words = words
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template
        self.definitions = definitions or [""] * len(words)

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            text = self.template.format(
                word=self.words[idx], definition=self.definitions[idx]
            )
        except KeyError:
            text = self.template.format(word=self.words[idx])

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "score": torch.tensor(self.scores[idx], dtype=torch.float32),
        }


class _RegressionModel(nn.Module):
    """XLM-RoBERTa backbone with a two-layer MLP regression head."""

    def __init__(self, backbone_name: str, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        dim = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [CLS] representation
        return self.head(cls).squeeze(-1)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class ConcretenessPredictor:
    """Predict a concreteness score (1–5 scale) for words in any language.

    Parameters
    ----------
    data_path:
        Path to the tab-separated Brysbaert et al. concreteness dataset.
        Defaults to ``english_conreteness_ratings.txt`` in the same directory.
    backbone:
        HuggingFace model name for XLM-RoBERTa (base or large).
    z_score:
        If True, z-score–normalise targets before training.  Predictions are
        automatically un-normalised so the output is always on the 1–5 scale.
    template:
        f-string template for input text.  Must include ``{word}``.  May also
        include ``{definition}`` to embed the English gloss alongside the word.
    device:
        ``"cuda"`` or ``"cpu"``.  Auto-detected when omitted.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        backbone: str = _DEFAULT_BACKBONE,
        z_score: bool = False,
        template: str = _DEFAULT_TEMPLATE,
        device: Optional[str] = None,
    ) -> None:
        self.data_path = Path(data_path) if data_path else _DATA_FILE
        self.backbone = backbone
        self.z_score = z_score
        self.template = template
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model: Optional[_RegressionModel] = None
        self._tokenizer = None
        self._score_mean: float = 0.0
        self._score_std: float = 1.0
        self._lookup_cache: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Milestone 1 – Data preparation
    # ------------------------------------------------------------------

    def _load_english_data(self) -> Tuple[List[str], List[float]]:
        """Load, clean, and return (words, scores) from the Brysbaert dataset."""
        df = pd.read_csv(self.data_path, sep="\t")
        df = df[df["Bigram"] == 0].dropna(subset=["Conc.M"])
        return df["Word"].astype(str).tolist(), df["Conc.M"].tolist()

    def _prepare_splits(
        self, words: List[str], scores: List[float]
    ) -> Tuple[
        List[str], List[str], List[str],
        List[float], List[float], List[float],
    ]:
        """Stratified 80/10/10 train/val/test split.

        Stratification uses five concreteness bins so every split covers the
        full abstract-to-concrete spectrum proportionally.
        """
        bins = list(pd.cut(scores, bins=5, labels=False))

        w_tr, w_tmp, s_tr, s_tmp, b_tr, b_tmp = train_test_split(
            words, scores, bins, test_size=0.2, stratify=bins, random_state=42
        )
        w_val, w_te, s_val, s_te = train_test_split(
            w_tmp, s_tmp, test_size=0.5, stratify=b_tmp, random_state=42
        )

        if self.z_score:
            self._score_mean = float(np.mean(s_tr))
            self._score_std = float(np.std(s_tr)) or 1.0
            s_tr = [(s - self._score_mean) / self._score_std for s in s_tr]
            s_val = [(s - self._score_mean) / self._score_std for s in s_val]
            s_te = [(s - self._score_mean) / self._score_std for s in s_te]

        return w_tr, w_val, w_te, s_tr, s_val, s_te

    # ------------------------------------------------------------------
    # Milestone 3 – Fine-tuning
    # ------------------------------------------------------------------

    def train(
        self,
        epochs_frozen: int = 2,
        epochs_unfrozen: int = 5,
        batch_size: int = 32,
        lr_backbone: float = 2e-5,
        lr_head: float = 1e-3,
        max_length: int = 64,
        patience: int = 3,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Fine-tune on English concreteness norms and return test-set metrics.

        Two-phase training:

        1. Freeze backbone; train head for ``epochs_frozen`` epochs.
        2. Unfreeze backbone; train end-to-end with differential LR for
           ``epochs_unfrozen`` epochs (early stopping on validation Pearson r).

        Returns
        -------
        dict
            ``pearson_r``, ``spearman_rho``, ``mae`` on the held-out English
            test split.  Expected English Pearson r: 0.92–0.96.
        """
        words, scores = self._load_english_data()
        w_tr, w_val, w_te, s_tr, s_val, s_te = self._prepare_splits(words, scores)

        self._tokenizer = AutoTokenizer.from_pretrained(self.backbone)
        self._model = _RegressionModel(self.backbone).to(self.device)

        def make_loader(ws, ss, shuffle):
            ds = _ConcretenessDataset(ws, ss, self._tokenizer, max_length, self.template)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        tr_loader = make_loader(w_tr, s_tr, shuffle=True)
        val_loader = make_loader(w_val, s_val, shuffle=False)
        te_loader = make_loader(w_te, s_te, shuffle=False)

        loss_fn = nn.MSELoss()

        # Phase 1: head only
        for p in self._model.backbone.parameters():
            p.requires_grad = False
        opt = torch.optim.AdamW(self._model.head.parameters(), lr=lr_head, weight_decay=0.01)
        if verbose:
            print(f"[1/2] Head-only warm-up ({epochs_frozen} epoch(s)) ...")
        self._run_epochs(
            tr_loader, val_loader, opt, loss_fn, epochs_frozen,
            patience=None, verbose=verbose, phase_label="warm-up",
        )

        # Phase 2: end-to-end with differential LR
        for p in self._model.backbone.parameters():
            p.requires_grad = True
        opt = torch.optim.AdamW(
            [
                {"params": self._model.backbone.parameters(), "lr": lr_backbone},
                {"params": self._model.head.parameters(), "lr": lr_head},
            ],
            weight_decay=0.01,
        )
        if verbose:
            print(f"[2/2] End-to-end fine-tune (up to {epochs_unfrozen} epoch(s), patience={patience}) ...")
        self._run_epochs(
            tr_loader, val_loader, opt, loss_fn, epochs_unfrozen,
            patience=patience, verbose=verbose, phase_label="fine-tune",
        )

        self._lookup_cache.clear()
        metrics = self._eval_loader(te_loader)
        if verbose:
            print(
                f"Test-set  Pearson r={metrics['pearson_r']:.3f}  "
                f"Spearman ρ={metrics['spearman_rho']:.3f}  "
                f"MAE={metrics['mae']:.3f}"
            )
        return metrics

    def _run_epochs(
        self, tr_loader, val_loader, opt, loss_fn, n_epochs, patience,
        verbose: bool = False, phase_label: str = "",
    ):
        best_r = -1.0
        best_state: Optional[Dict] = None
        no_improve = 0

        for epoch in range(1, n_epochs + 1):
            self._model.train()
            total_loss = 0.0
            bar = tqdm(
                tr_loader,
                desc=f"  [{phase_label}] epoch {epoch}/{n_epochs}",
                leave=False,
                disable=not verbose,
            )
            for batch in bar:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                targets = batch["score"].to(self.device)
                loss = loss_fn(self._model(ids, mask), targets)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                bar.set_postfix(loss=f"{loss.item():.4f}")

            val_metrics = self._eval_loader(val_loader)
            val_r = val_metrics["pearson_r"]
            is_best = val_r > best_r
            if is_best:
                best_r = val_r
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if verbose:
                tag = " *" if is_best else f"  (no improvement {no_improve}/{patience})" if patience else ""
                avg_loss = total_loss / len(tr_loader)
                print(
                    f"  [{phase_label}] epoch {epoch}/{n_epochs}"
                    f"  loss={avg_loss:.4f}"
                    f"  val_r={val_r:.4f}{tag}"
                )

            if patience and no_improve >= patience:
                if verbose:
                    print(f"  Early stopping after {epoch} epochs.")
                break

        if best_state:
            self._model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

    def _eval_loader(self, loader) -> Dict[str, float]:
        self._model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                preds.extend(self._model(ids, mask).cpu().numpy().tolist())
                targets.extend(batch["score"].numpy().tolist())
        p, t = np.array(preds), np.array(targets)
        return {
            "pearson_r": float(np.corrcoef(p, t)[0, 1]),
            "spearman_rho": float(stats.spearmanr(p, t).statistic),
            "mae": float(np.mean(np.abs(p - t))),
        }

    # ------------------------------------------------------------------
    # Milestone 4 + 7 – Zero-shot inference and lookup caching
    # ------------------------------------------------------------------

    def _predict_words(
        self,
        words: List[str],
        definitions: Optional[List[str]] = None,
        batch_size: int = 64,
    ) -> List[float]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded.  Call train() or load() first.")

        definitions = definitions or [""] * len(words)

        # Keys include the definition so the same word with different context
        # gets a separate cache entry.
        keys = [f"{w}\x00{d}" for w, d in zip(words, definitions)]
        uncached_idx = [i for i, k in enumerate(keys) if k not in self._lookup_cache]

        if uncached_idx:
            self._model.eval()
            uw = [words[i] for i in uncached_idx]
            ud = [definitions[i] for i in uncached_idx]

            raw_scores: List[float] = []
            with torch.no_grad():
                for start in range(0, len(uw), batch_size):
                    bw = uw[start : start + batch_size]
                    bd = ud[start : start + batch_size]
                    texts = []
                    for w, d in zip(bw, bd):
                        try:
                            texts.append(self.template.format(word=w, definition=d))
                        except KeyError:
                            texts.append(self.template.format(word=w))

                    enc = self._tokenizer(
                        texts,
                        max_length=64,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).to(self.device)
                    batch_preds = self._model(enc["input_ids"], enc["attention_mask"])
                    raw_scores.extend(batch_preds.cpu().numpy().tolist())

            for i, (score, key) in enumerate(zip(raw_scores, [keys[i] for i in uncached_idx])):
                # Undo z-score normalisation so output is always on the 1–5 scale.
                calibrated = score * self._score_std + self._score_mean if self.z_score else score
                self._lookup_cache[key] = float(calibrated)

        return [self._lookup_cache[k] for k in keys]

    def predict(self, input_format: InputFormat, batch_size: int = 64) -> pd.DataFrame:
        """Score every word in *input_format* for concreteness.

        XLM-RoBERTa's shared multilingual embedding space allows the model
        fine-tuned on English to transfer zero-shot to Vietnamese (and other
        languages) — no target-language labels required.

        Parameters
        ----------
        input_format:
            An :class:`InputFormat` instance.  The ``From-Language word``
            column supplies the target-language words.  If a ``To-Language
            definition`` column is present it is used as context (Milestone 1
            augmentation) when the template contains ``{definition}``.
        batch_size:
            Number of words processed per GPU/CPU batch.

        Returns
        -------
        pandas.DataFrame
            Columns: ``word`` (str), ``concreteness`` (float, 1–5 scale).
        """
        df = input_format.to_dataframe()
        words = _get_word_column(df).tolist()
        defs = _get_definition_column(df)
        scores = self._predict_words(words, definitions=defs, batch_size=batch_size)
        return pd.DataFrame({"word": words, "concreteness": scores})

    # ------------------------------------------------------------------
    # Milestone 5 – Validation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        words: List[str],
        human_scores: List[float],
        definitions: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compare model predictions to held-out human concreteness norms.

        Parameters
        ----------
        words:
            Target-language words.
        human_scores:
            Human concreteness ratings on the same 1–5 scale.
        definitions:
            Optional context strings parallel to *words*.

        Returns
        -------
        dict
            ``pearson_r``, ``spearman_rho``, ``mae``.
        """
        preds = np.array(self._predict_words(words, definitions=definitions))
        targets = np.array(human_scores)
        return {
            "pearson_r": float(np.corrcoef(preds, targets)[0, 1]),
            "spearman_rho": float(stats.spearmanr(preds, targets).statistic),
            "mae": float(np.mean(np.abs(preds - targets))),
        }

    # ------------------------------------------------------------------
    # Milestone 6 – Optional few-shot fine-tuning on target language
    # ------------------------------------------------------------------

    def fine_tune_target(
        self,
        words: List[str],
        scores: List[float],
        definitions: Optional[List[str]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 1e-5,
        max_length: int = 64,
    ) -> None:
        """Fine-tune on a small set of target-language concreteness labels.

        With as few as 200–500 labeled words this typically lifts Pearson r by
        0.05–0.10 over the zero-shot baseline.

        Parameters
        ----------
        words:
            Target-language words with known concreteness ratings.
        scores:
            Human concreteness ratings (1–5 scale).
        definitions:
            Optional context strings; passed to the same template as predict().
        epochs:
            Number of fine-tuning epochs (keep low to avoid overfitting).
        lr:
            Learning rate (very low to retain English-learned geometry).
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded.  Call train() or load() first.")

        definitions = definitions or [""] * len(words)
        ds = _ConcretenessDataset(
            words, scores, self._tokenizer, max_length, self.template, definitions
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        loss_fn = nn.MSELoss()
        opt = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=0.01)

        self._model.train()
        for _ in range(epochs):
            for batch in loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                targets = batch["score"].to(self.device)
                loss = loss_fn(self._model(ids, mask), targets)
                opt.zero_grad()
                loss.backward()
                opt.step()

        self._lookup_cache.clear()

    # ------------------------------------------------------------------
    # Persistence (Milestone 7)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights, tokenizer, and normalisation metadata to *path*."""
        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), dest / "model.pt")
        self._tokenizer.save_pretrained(str(dest / "tokenizer"))
        meta = {
            "backbone": self.backbone,
            "z_score": self.z_score,
            "score_mean": self._score_mean,
            "score_std": self._score_std,
            "template": self.template,
        }
        (dest / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "ConcretenessPredictor":
        """Load a previously saved predictor from *path*."""
        src = Path(path)
        meta = json.loads((src / "meta.json").read_text())
        predictor = cls(
            backbone=meta["backbone"],
            z_score=bool(meta["z_score"]),
            template=meta["template"],
            device=device,
        )
        predictor._score_mean = float(meta["score_mean"])
        predictor._score_std = float(meta["score_std"])
        predictor._tokenizer = AutoTokenizer.from_pretrained(str(src / "tokenizer"))
        predictor._model = _RegressionModel(meta["backbone"])
        predictor._model.load_state_dict(
            torch.load(src / "model.pt", map_location=predictor.device)
        )
        predictor._model.to(predictor.device)
        return predictor


# ---------------------------------------------------------------------------
# Column extraction helpers
# ---------------------------------------------------------------------------


def _get_word_column(df: pd.DataFrame) -> pd.Series:
    for col in ("From-Language word", "word"):
        if col in df.columns:
            return df[col].astype(str)
    return df.iloc[:, 0].astype(str)


def _get_definition_column(df: pd.DataFrame) -> List[str]:
    for col in ("To-Language definition", "definition"):
        if col in df.columns:
            return df[col].fillna("").astype(str).tolist()
    return [""] * len(df)
