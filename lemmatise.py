import re
import json
import collections
import fasttext
import pyvi
from tqdm import tqdm

ft_vi = fasttext.load_model("cc.vi.300.bin")  # Vietnamese vectors
WORD_RE = re.compile(r"\w+", re.UNICODE)


def vi_tokenize(text: str):
    # pyvi.ViTokenizer is a lightweight wrapper around RDRSegmenter
    return pyvi.ViTokenizer.tokenize(text).split()


lemmas = collections.Counter()
with open("viwiki_clean.txt", encoding="utf-8") as f:
    for line in tqdm(f, desc="lemmatising"):
        for tok in vi_tokenize(line.lower()):
            if tok.isalpha() and len(tok) > 1:
                lemmas[tok] += 1

# keep lemmas that occur ≥ 5 times and have a fastText vector
vocab = {w for w, c in lemmas.items() if c >= 5 and w in ft_vi}
json.dump(sorted(vocab), open("vi_vocab.json", "w", encoding="utf-8"))
print(f"✅ {len(vocab):,} Vietnamese lemmas")
