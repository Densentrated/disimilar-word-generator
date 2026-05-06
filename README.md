# Dissimilar Word Generator

Scores words from a dictionary file across several dissimilarity metrics and outputs a ranked CSV.

## Usage

```bash
python src/pipeline.py <input_file> [options]
```

### Arguments

| Argument | Description |
|---|---|
| `input_file` | Path to a Kaikki JSONL dictionary file |
| `--output` | Output CSV path (default: `output.csv`) |
| `--model-path` | Directory to load/save the concreteness predictor (default: `.data/concreteness_predictor`) |
| `--weight-cosine` | Weight for `cosine_distance` (default: `1.0`) |
| `--weight-word-count` | Weight for `definition_word_count` (default: `1.0`) |
| `--weight-orthographic` | Weight for `orthographic_similarity` (default: `1.0`) |
| `--weight-cognate` | Weight for `cognate_similarity` (default: `1.0`) |
| `--weight-concreteness` | Weight for `concreteness` (default: `1.0`) |
| `--invert-cosine` | Flip `cosine_distance` before scoring (`1 - normalized`) |
| `--invert-word-count` | Flip `definition_word_count` before scoring |
| `--invert-orthographic` | Flip `orthographic_similarity` before scoring (rewards foreign-looking words) |
| `--invert-cognate` | Flip `cognate_similarity` before scoring (rewards foreign-sounding words) |
| `--invert-concreteness` | Flip `concreteness` before scoring (rewards abstract words) |

### Examples

Run with default weights:
```bash
python src/pipeline.py kaikki.org-dictionary-Vietnamese.jsonl --output results.csv
```

Run with custom weights to emphasise semantic dissimilarity and concreteness:
```bash
python src/pipeline.py kaikki.org-dictionary-Vietnamese.jsonl \
  --output results.csv \
  --weight-cosine 2.0 \
  --weight-word-count 0.5 \
  --weight-orthographic 1.0 \
  --weight-cognate 1.0 \
  --weight-concreteness 1.5
```

Find words that are semantically dissimilar from English and look/sound foreign (not native to English):
```bash
python src/pipeline.py kaikki.org-dictionary-Vietnamese.jsonl \
  --output results.csv \
  --weight-cosine 3.0 \
  --weight-word-count 0.0 \
  --weight-orthographic 1.0 \
  --weight-cognate 1.0 \
  --weight-concreteness 1.0 \
  --invert-orthographic \
  --invert-cognate
```

## Output

The output CSV contains one row per word, sorted by `normalized_score` descending:

| Column | Description |
|---|---|
| `From-Language word` | The source word |
| `To-Language definition` | The definition/translation |
| `cosine_distance` | Semantic distance between word and definition embeddings (0–1, higher = more dissimilar) |
| `definition_word_count` | Number of words in the definition |
| `orthographic_similarity` | Character-level similarity between word and definition (0–1, higher = more similar) |
| `cognate_similarity` | Phonological similarity using sound classes (0–1, higher = more similar) |
| `concreteness` | Predicted concreteness of the word (1=abstract, 5=concrete) |
| `normalized_score` | Weighted sum of all metrics normalized to [0, 1] |

All metrics are min-max normalized to [0, 1] before weighting. The final score is:

```
normalized_score = sum(weight_i × normalized_metric_i) / sum(weights)
```

## Concreteness Predictor

The first run will train an XLM-RoBERTa model on English concreteness norms (Brysbaert et al. 2014), which can take 10–30 minutes on CPU. The trained model is saved to `--model-path` and reused on subsequent runs.

## Input Format

Input files should be in [Kaikki](https://kaikki.org) JSONL format. Each line is a JSON object with at minimum a `word` field and a `senses` array containing `glosses`.
