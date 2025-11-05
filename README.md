# Vietnamese Word Scraper

A Python toolkit for extracting and analyzing Vietnamese words from Wikipedia dumps using FastText embeddings and cross-lingual similarity analysis.

## Overview

This project processes Vietnamese Wikipedia dumps to:
1. Extract clean text from Wikipedia XML dumps
2. Tokenize and lemmatize Vietnamese text
3. Build English word embedding pools
4. Find Vietnamese words that are "distant closest" - words whose nearest English embedding neighbor is still relatively far away

## Files

### Core Processing Scripts

1. **`extract_text.py`** - Extracts clean text from Wikipedia XML dumps
   - Uses WikiExtractor to convert XML to JSON format
   - Filters out disambiguation pages and templates
   - Outputs: `viwiki_clean.txt`

2. **`lemmatise.py`** - Tokenizes and lemmatizes Vietnamese text
   - Uses PyVi tokenizer for Vietnamese text segmentation
   - Filters lemmas by frequency (≥5 occurrences) and FastText coverage
   - Requires: `viwiki_clean.txt`, `cc.vi.300.bin`
   - Outputs: `vi_vocab.json`

3. **`buildenpool.py`** - Builds English word embedding pool
   - Loads top 200,000 most frequent English words from FastText
   - Creates embedding matrix for efficient similarity computation
   - Requires: `cc.en.300.bin`
   - Outputs: `en_matrix.npy`, `en_words.json`

4. **`finddistclosesst.py`** - Finds distant closest matches
   - Computes cosine similarity between Vietnamese and English embeddings
   - Identifies words with low similarity to their closest English match
   - Outputs: `top1000_distant_closest.json`

### Utility Scripts

5. **`cosinesim.py`** - Efficient batch cosine similarity computation
   - Vectorized operations for computing similarity matrices
   - Normalizes vectors before dot product

6. **`view_list.py`** - Display results in a readable format
   - Loads and displays the top distant closest matches
   - Uses pandas for formatting

7. **`wikiscrapyer.py`** - Alternative Wikipedia processing pipeline
   - Downloads Vietnamese Wikipedia dump directly
   - Extracts and processes in one pipeline
   - Outputs: `vi_lemmas.json` (top 50,000 lemmas)
   - ⚠️ Requires Python 3.12 or earlier

8. **`extract_text_simple.py`** - Python 3.13+ compatible extractor
   - Direct XML parsing without wikiextractor dependency
   - Works with Python 3.13 and later
   - Alternative to `extract_text.py`

## Requirements

### Python Version Compatibility

⚠️ **Important**: The `wikiextractor` library has compatibility issues with Python 3.13+ due to deprecated regex syntax.

**Recommended Options:**
1. **Use Python 3.12 or earlier** (recommended for original scripts)
2. **Use the alternative script** `extract_text_simple.py` (works with Python 3.13+)

### Installing Dependencies

**For Python 3.12 and earlier:**
```bash
pip install fasttext numpy pandas tqdm underthesea pyvi wikiextractor
```

**For Python 3.13+:**
```bash
pip install fasttext numpy pandas tqdm underthesea pyvi
# Note: Do not install wikiextractor - use extract_text_simple.py instead
```

### External Dependencies

- **FastText Models**: Download pre-trained models:
  - Vietnamese: `cc.vi.300.bin`
  - English: `cc.en.300.bin`
  - Available from: https://fasttext.cc/docs/en/crawl-vectors.html

## Usage

### Standard Pipeline (Python 3.12 and earlier)

```bash
# 1. Extract text from Wikipedia dump
python extract_text.py

# 2. Lemmatize Vietnamese text
python lemmatise.py

# 3. Build English word pool
python buildenpool.py

# 4. Find distant closest matches
python finddistclosesst.py

# 5. View results
python view_list.py
```

### Alternative Pipeline (Python 3.13+)

```bash
# 1. Extract text from Wikipedia dump (Python 3.13 compatible)
python extract_text_simple.py

# 2. Lemmatize Vietnamese text
python lemmatise.py

# 3. Build English word pool
python buildenpool.py

# 4. Find distant closest matches
python finddistclosesst.py

# 5. View results
python view_list.py
```

### All-in-One Pipeline (Python 3.12 and earlier)

```bash
python wikiscrapyer.py
```

## Code Quality Fixes Applied

### Critical Fixes

1. **`cosinesim.py`**: Fixed constant naming convention
   - Changed uppercase parameter names `X`, `Y` to lowercase `x`, `y`
   - Prevents constant redefinition errors in type checkers

2. **`wikiscrapyer.py`**: Fixed subprocess stdout handling
   - Added None check for `p.stdout` before iteration
   - Added type ignore comment for type checker satisfaction

### Code Style Improvements

3. **All files**: Split multiple imports to separate lines
   - Follows PEP 8 style guidelines
   - Improves readability

4. **`buildenpool.py`**: Removed unused `pandas` import

5. **`extract_text.py`**: Removed unused `os` import

6. **All subprocess calls**: Assigned return values to `_`
   - Indicates intentional unused return values
   - Suppresses linter warnings

## Output Files

- `viwiki_clean.txt` - Cleaned Wikipedia text
- `vi_vocab.json` - Vietnamese vocabulary (filtered lemmas)
- `vi_lemmas.json` - Top 50,000 Vietnamese lemmas with frequencies
- `en_matrix.npy` - English word embedding matrix (200k × 300)
- `en_words.json` - English word list (200k words)
- `top1000_distant_closest.json` - Top 1000 Vietnamese words with distant English matches

## How It Works

### Distant Closest Algorithm

1. **Embedding Space**: Both Vietnamese and English words are represented as 300-dimensional FastText vectors
2. **Similarity Metric**: Cosine similarity measures angle between vectors
3. **Finding Matches**: For each Vietnamese word, find the English word with highest similarity
4. **Ranking**: Sort by similarity (ascending) to find words where even the "closest" match is far away

These "distant closest" words are interesting because they may represent:
- Cultural concepts unique to Vietnamese
- Compound words that don't translate directly
- Technical or specialized vocabulary
- Words that require contextual translation

## Performance Tuning

### extract_text_simple.py Memory and Speed Optimization

The `BATCH_SIZE` parameter controls the trade-off between memory usage and I/O performance:

```python
# In extract_text_simple.py, line ~37
BATCH_SIZE = 250  # Default: balanced performance
```

**Tuning Guidelines:**
- **Low Memory (100-250 pages)**: Uses ~100-150MB RAM, more frequent disk writes
- **Balanced (250-500 pages)**: Uses ~150-250MB RAM, good I/O performance ✅ **Recommended**
- **High Performance (500-1000 pages)**: Uses ~250-500MB RAM, fewer disk writes

**When to adjust:**
- Low RAM system? → Use 100-150
- SSD with fast I/O? → Use 500-1000
- HDD or slow storage? → Keep default 250

### Other Performance Notes

- Processing the full Wikipedia dump can take several hours
- Requires ~10GB disk space for dumps and intermediate files
- FastText models are ~7GB each (Vietnamese and English)
- Batch size in `finddistclosesst.py` can be adjusted based on available RAM

## License

This is a research/educational tool. Please respect Wikipedia's licensing terms when using extracted data.