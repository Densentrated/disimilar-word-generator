# Dissimilar Word Generator

Scores words from a dictionary file across several dissimilarity metrics and outputs a ranked CSV. Then uses machine learning to find words similar to user-provided examples.

## Pipeline Part 1: Score All Words

Scores words from a dictionary across multiple metrics.

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
  --invert-cognate \
  --invert-concreteness
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

## Pipeline Part 2: Find Similar Words

After running `pipeline.py`, use machine learning to find words similar to user-provided examples.

```bash
python src/pipeline_part2.py <input_csv> --examples <row_indices> [options]
```

This script trains a binary classifier on your example words and finds other words in the dataset with similar characteristics.

### Arguments

| Argument | Description |
|---|---|
| `input_csv` | Path to the CSV output from `pipeline.py` |
| `--examples` | Space-separated list of row indices (0-based) of example words (required) |
| `--output` | Output CSV path (default: `similar_words.csv`) |
| `--model` | ML classifier to use: `logistic_regression` (default), `svm`, or `random_forest` |
| `--top-n` | Only output top N most similar words (optional) |

### How It Works

1. You provide 2-5 example words (by row index from the pipeline.py output)
2. The script extracts features from those examples (similarity metrics, concreteness, etc.)
3. Trains a machine learning classifier to learn what makes those words special
4. Scores all other words based on how similar they are to your examples
5. Outputs a ranked CSV sorted by similarity score

### Examples

Find words similar to rows 0, 5, and 12:
```bash
python src/pipeline_part2.py output.csv --examples 0 5 12
```

Use SVM classifier (often more accurate) and get top 50 results:
```bash
python src/pipeline_part2.py output.csv --examples 0 5 12 --model svm --top-n 50
```

Use random forest classifier and save to custom file:
```bash
python src/pipeline_part2.py output.csv --examples 10 20 30 --model random_forest --output my_similar_words.csv
```

Find words similar to just one example:
```bash
python src/pipeline_part2.py output.csv --examples 42
```

### Classifier Comparison

| Classifier | Speed | Accuracy | Best For |
|---|---|---|---|
| `logistic_regression` | ⚡⚡⚡ Fast | Good | Linear patterns, large datasets |
| `svm` | ⚡⚡ Medium | Very Good | Complex patterns, smaller datasets |
| `random_forest` | ⚡ Slow | Excellent | Finding non-obvious similarities |

### Output

The output CSV contains results sorted by `similarity_score` descending:

| Column | Description |
|---|---|
| `From-Language word` | The source word |
| `To-Language definition` | The definition/translation |
| `similarity_score` | Probability of similarity to your examples (0–1, higher = more similar) |
| `cosine_distance` | Semantic distance between word and definition embeddings |
| `definition_word_count` | Number of words in the definition |
| `orthographic_similarity` | Character-level similarity between word and definition |
| `cognate_similarity` | Phonological similarity using sound classes |

## Workflow Example

Here's a typical workflow to find dissimilar words that belong to a specific category:

### Step 1: Score all words
```bash
python src/pipeline.py kaikki.org-dictionary-Vietnamese.jsonl \
  --output vietnamese_scores.csv \
  --invert-orthographic \
  --invert-cognate
```

### Step 2: Review results and identify examples
Open `vietnamese_scores.csv` and look through the top-ranked words. Find 3-5 words that match the category you're interested in. Note their row indices.

### Step 3: Find similar words using ML
```bash
python src/pipeline_part2.py vietnamese_scores.csv \
  --examples 0 5 12 \
  --model random_forest \
  --top-n 100 \
  --output category_words.csv
```

### Step 4: Refine and repeat
Review `category_words.csv`. If the results aren't quite right, try:
- Different example indices
- A different classifier (`--model svm`)
- Adjusting pipeline.py weights and re-running step 1

## Input Format

Input files should be in [Kaikki](https://kaikki.org) JSONL format. Each line is a JSON object with at minimum a `word` field and a `senses` array containing `glosses`.
