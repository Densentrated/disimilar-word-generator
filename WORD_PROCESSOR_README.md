# Vietnamese Word to English Neighbor Mapper

This tool processes Vietnamese words extracted from Wikipedia and finds the nearest English word using multilingual embeddings and cosine similarity.

## Overview

The `word_set_processor.py` script:
1. Loads Vietnamese words from a file (e.g., `extracted_words.txt`)
2. Loads an English word corpus from NLTK
3. Uses a multilingual sentence-transformer model to embed both Vietnamese and English words
4. Finds the nearest English word for each Vietnamese word using cosine similarity
5. Outputs: `vietnamese_word,nearest_english_word,cosine_distance`

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `sentence-transformers` - For multilingual embeddings
- `scikit-learn` - For cosine similarity calculations
- `numpy` - For numerical operations
- `nltk` - For English word corpus
- `torch` - Required by sentence-transformers
- `transformers` - Required by sentence-transformers

### 2. Download NLTK Data

The script will automatically download the NLTK words corpus on first run, but you can also pre-download it:

```python
import nltk
nltk.download('words')
```

## Usage

### Basic Usage

```bash
python word_set_processor.py extracted_words.txt
```

This will create `extracted_words_with_neighbors.txt` with the results.

### Custom Output File

```bash
python word_set_processor.py extracted_words.txt output_with_neighbors.txt
```

## Output Format

The output file contains one line per Vietnamese word:

```
vietnamese_word,nearest_english_word,cosine_distance
```

**Example:**
```
việt,viet,0.123456
hóa,化,0.234567
thành,成,0.345678
phố,street,0.456789
```

### Understanding Cosine Distance

- **Cosine Distance** = 1 - Cosine Similarity
- **Range**: 0.0 to 2.0
  - `0.0` = Identical embeddings (perfect match)
  - `< 0.3` = Very similar
  - `0.3 - 0.7` = Somewhat similar
  - `> 0.7` = Not very similar
  - `2.0` = Completely opposite

## Model Information

### Multilingual Embedding Model

The script uses: `paraphrase-multilingual-MiniLM-L12-v2`

**Features:**
- Supports 50+ languages including Vietnamese and English
- 384-dimensional embeddings
- Relatively small and fast (~420MB)
- Good balance between speed and accuracy

**First Run:** The model will be automatically downloaded (~420MB). Subsequent runs will use the cached version.

## Performance Considerations

### Memory Usage

- **English words**: ~235,000 words × 384 dimensions × 4 bytes ≈ **360 MB**
- **Model**: ~420 MB
- **Processing**: ~100-500 MB
- **Total**: ~1-2 GB RAM

### Processing Speed

Processing times (approximate):
- **English embedding**: ~2-5 minutes (one-time per run)
- **Vietnamese processing**: ~1,000 words/minute
- **10,000 Vietnamese words**: ~10-15 minutes total

### Optimization Tips

1. **Batch Size**: Default is 100. Increase for faster processing (but more memory):
   ```python
   # In the script, modify:
   batch_size=500  # instead of 100
   ```

2. **Subset English Words**: If you want faster processing, filter the English corpus:
   ```python
   # Only keep common words
   english_words = [w for w in english_words if len(w) >= 3]
   ```

3. **GPU Acceleration**: If you have a CUDA-capable GPU:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```
   The model will automatically use GPU if available.

## Example Workflow

```bash
# 1. Extract Vietnamese words from Wikipedia dump
python wikepedia_parser.py viwiki-latest-pages-articles.xml.bz2 extracted_words.txt

# 2. Find nearest English neighbors
python word_set_processor.py extracted_words.txt vietnamese_to_english.txt

# 3. Check results
head -20 vietnamese_to_english.txt
```

## Alternative Models

If you need better accuracy or different language support, you can modify the model in the script:

### For Better Accuracy (but slower):
```python
model = SentenceTransformer('sentence-transformers/LaBSE')  # 470MB, 768-dim
```

### For Faster Processing (but less accurate):
```python
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')  # 278MB, 384-dim
```

### For Best Quality (but much larger):
```python
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
```

## Troubleshooting

### Out of Memory Error

**Solution 1**: Reduce batch size in the script:
```python
batch_size=50  # or even 10
```

**Solution 2**: Process in chunks:
```bash
# Split your input file
split -l 10000 extracted_words.txt chunk_

# Process each chunk
python word_set_processor.py chunk_aa output_aa.txt
python word_set_processor.py chunk_ab output_ab.txt
# etc...

# Combine results
cat output_*.txt > final_output.txt
```

### Model Download Fails

If the automatic download fails, manually download the model:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

### NLTK Words Not Found

```python
import nltk
nltk.download('words', quiet=False)
```

## Technical Details

### Cosine Similarity Formula

For embeddings **v1** (Vietnamese) and **v2** (English):

```
cosine_similarity = (v1 · v2) / (||v1|| × ||v2||)
cosine_distance = 1 - cosine_similarity
```

### Why Multilingual Embeddings Work

Multilingual models are trained on parallel corpora (same text in multiple languages), learning to map semantically similar words/phrases to nearby points in the embedding space, regardless of language.

**Example**: 
- Vietnamese "xe" (car/vehicle) and English "car" will have similar embeddings
- Even though they're different languages, their semantic meaning aligns them in the embedding space

## License

This tool uses:
- `sentence-transformers`: Apache 2.0 License
- `nltk`: Apache 2.0 License
- `scikit-learn`: BSD License

See individual package licenses for details.