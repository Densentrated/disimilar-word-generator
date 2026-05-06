#!/usr/bin/env python3
"""Pipeline CLI: convert a Kaikki JSONL file and score each word.

Usage:
    python pipeline.py <input_file> [options]

Steps:
  1. Parse the input JSONL (Kaikki format) into word/definition pairs.
  2. Embed word and definition; compute cosine distance.
  3. Count words in each definition.
  4. Compute orthographic similarity between word and definition.
  5. Compute cognate similarity between word and definition.
  6. Predict concreteness for each word.
  7. Min-max normalize all metrics and compute a weighted normalized score.
  8. Output a CSV sorted by normalized_score descending.
"""

import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataframe_processor as dfp
from concreteness_predictor import ConcretenessPredictor
from input_format import InputFormat
from kaikki_jsonl_input_converter import KaikkiJsonInputConverter

_DEFAULT_MODEL_PATH = ".data/concreteness_predictor"


def _get_predictor(model_path: str) -> ConcretenessPredictor:
    if os.path.isdir(model_path):
        print(f"Loading concreteness predictor from {model_path} ...")
        predictor = ConcretenessPredictor.load(model_path)
        print("Predictor loaded.")
    else:
        print("No saved predictor found — training on English concreteness norms.")
        print("(This may take 10–30 min on CPU; much faster on GPU.)")
        predictor = ConcretenessPredictor()
        predictor.train(verbose=True)
        os.makedirs(model_path, exist_ok=True)
        predictor.save(model_path)
        print(f"Predictor saved to {model_path}")
    return predictor


_ALT_SPELLING_PATTERN = re.compile(
    r"\b(alternative|variant|obsolete|archaic|eye dialect|misspelling|alternate)\s+"
    r"(spelling|form|capitalization|romanization|transcription)\s+of\b",
    re.IGNORECASE,
)


def _drop_unusable(df: pd.DataFrame) -> pd.DataFrame:
    defs = df["To-Language definition"].fillna("").astype(str)
    alt_mask = defs.apply(lambda d: bool(_ALT_SPELLING_PATTERN.search(d)))
    failed_mask = defs.str.startswith("failed:")
    mask = alt_mask | failed_mask
    dropped = mask.sum()
    if dropped:
        print(f"Removed {dropped} unusable entries ({alt_mask.sum()} alternative spellings, {failed_mask.sum()} parse failures).")
    return df[~mask].reset_index(drop=True)


def _minmax_normalize(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score words from a Kaikki JSONL file and output a ranked CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Weights control each metric's contribution to normalized_score.
All metrics are min-max normalized to [0, 1] before weighting.
Final score = sum(weight_i * norm_metric_i) / sum(weights).

Metric directions (higher raw value means ...):
  cosine_distance         more semantically dissimilar
  definition_word_count   longer definition
  orthographic_similarity more orthographically similar  (higher = more similar)
  cognate_similarity      more phonologically similar    (higher = more similar)
  concreteness            more concrete (1=abstract, 5=concrete)

Use --invert-<metric> to flip a metric before scoring (uses 1 - normalized_value).
Example: --invert-orthographic --invert-cognate rewards words that look/sound foreign.
""",
    )
    parser.add_argument("input_file", help="Path to the Kaikki JSONL input file.")
    parser.add_argument(
        "--output",
        default="output.csv",
        help="Output CSV path (default: output.csv).",
    )
    parser.add_argument(
        "--model-path",
        default=_DEFAULT_MODEL_PATH,
        help=f"Directory for the concreteness predictor (default: {_DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--weight-cosine",
        type=float,
        default=1.0,
        help="Weight for cosine_distance (default: 1.0).",
    )
    parser.add_argument(
        "--weight-word-count",
        type=float,
        default=1.0,
        help="Weight for definition_word_count (default: 1.0).",
    )
    parser.add_argument(
        "--weight-orthographic",
        type=float,
        default=1.0,
        help="Weight for orthographic_similarity (default: 1.0).",
    )
    parser.add_argument(
        "--weight-cognate",
        type=float,
        default=1.0,
        help="Weight for cognate_similarity (default: 1.0).",
    )
    parser.add_argument(
        "--weight-concreteness",
        type=float,
        default=1.0,
        help="Weight for concreteness (default: 1.0).",
    )
    parser.add_argument(
        "--invert-cosine",
        action="store_true",
        help="Invert cosine_distance before scoring.",
    )
    parser.add_argument(
        "--invert-word-count",
        action="store_true",
        help="Invert definition_word_count before scoring.",
    )
    parser.add_argument(
        "--invert-orthographic",
        action="store_true",
        help="Invert orthographic_similarity before scoring (rewards foreign-looking words).",
    )
    parser.add_argument(
        "--invert-cognate",
        action="store_true",
        help="Invert cognate_similarity before scoring (rewards foreign-sounding words).",
    )
    parser.add_argument(
        "--invert-concreteness",
        action="store_true",
        help="Invert concreteness before scoring (rewards abstract words).",
    )

    args = parser.parse_args()

    # 1. Convert input file to intermediary format
    print(f"Parsing {args.input_file} ...")
    converter = KaikkiJsonInputConverter()
    input_obj = converter.Kaikki_To_Input(args.input_file)
    df = input_obj.to_dataframe()
    print(f"Loaded {len(df)} rows.")

    # 2. Remove alternative-spelling entries
    df = _drop_unusable(df)
    print(f"{len(df)} rows after filtering.")

    # 3. Embed word and definition columns
    print("Embedding words ...")
    df = dfp.embed_column(df, "From-Language word", "From-Language word Embedding")
    print("Embedding definitions ...")
    df = dfp.embed_column(
        df, "To-Language definition", "To-Language definition Embedding"
    )

    # 4. Cosine distance between word and definition embeddings
    print("Computing cosine distance ...")
    df = dfp.add_cosine_distance(
        df,
        "From-Language word Embedding",
        "To-Language definition Embedding",
        out_col="cosine_distance",
    )

    # 5. Word count of the definition
    print("Computing definition word count ...")
    df = dfp.add_word_count_column(
        df, "To-Language definition", output_column="definition_word_count"
    )

    # 6. Orthographic similarity between word and definition
    print("Computing orthographic similarity ...")
    df = dfp.add_orthographic_similarity(
        df,
        "From-Language word",
        "To-Language definition",
        output_column="orthographic_similarity",
    )

    # 7. Cognate similarity between word and definition
    print("Computing cognate similarity ...")
    df = dfp.add_cognate_similarity(
        df,
        "From-Language word",
        "To-Language definition",
        output_column="cognate_similarity",
        sound_class_model="sca",
    )

    # 9. Concreteness of each word
    print("Predicting concreteness ...")
    predictor = _get_predictor(args.model_path)
    fmt = InputFormat(df)
    scores_df = predictor.predict(fmt)
    df["concreteness"] = scores_df["concreteness"].values

    # 10. Normalized weighted score
    weights = {
        "cosine_distance": args.weight_cosine,
        "definition_word_count": args.weight_word_count,
        "orthographic_similarity": args.weight_orthographic,
        "cognate_similarity": args.weight_cognate,
        "concreteness": args.weight_concreteness,
    }
    inverted = {
        "cosine_distance": args.invert_cosine,
        "definition_word_count": args.invert_word_count,
        "orthographic_similarity": args.invert_orthographic,
        "cognate_similarity": args.invert_cognate,
        "concreteness": args.invert_concreteness,
    }

    total_weight = sum(weights.values())
    if total_weight == 0:
        df["normalized_score"] = 0.0
    else:
        score = pd.Series(0.0, index=df.index)
        for col, w in weights.items():
            normalized = _minmax_normalize(df[col].astype(float))
            if inverted[col]:
                normalized = 1.0 - normalized
            score += w * normalized
        df["normalized_score"] = score / total_weight

    # Drop embedding columns, sort by score descending
    df = df.drop(
        columns=["From-Language word Embedding", "To-Language definition Embedding"]
    )
    df = df.sort_values("normalized_score", ascending=False).reset_index(drop=True)

    # Write CSV
    output_cols = [
        "From-Language word",
        "To-Language definition",
        "cosine_distance",
        "definition_word_count",
        "orthographic_similarity",
        "cognate_similarity",
        "concreteness",
        "normalized_score",
    ]
    df[output_cols].to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows to {args.output}.")


if __name__ == "__main__":
    main()
