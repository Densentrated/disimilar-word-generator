# 4_find_distant_closest.py
import json
import numpy as np
import fasttext
import cosinesim
from tqdm import tqdm

ft_vi = fasttext.load_model("cc.vi.300.bin")
vi_vocab = json.load(open("vi_vocab.json"))
en_words = json.load(open("en_words.json"))
en_matrix = np.load("en_matrix.npy")

results = []
batch_size = 5_000
for i in tqdm(range(0, len(vi_vocab), batch_size), desc="searching"):
    batch = vi_vocab[i : i + batch_size]
    vi_vecs = np.vstack([ft_vi.get_word_vector(w) for w in batch])
    # cosine similarity against entire English matrix
    sims = cosinesim.similarity_batch(vi_vecs, en_matrix)  # shape (B, |EN|)
    best_idx = sims.argmax(1)
    best_sim = sims.max(1)
    for w, idx, s in zip(batch, best_idx, best_sim):
        results.append((w, en_words[idx], float(s)))

# sort by similarity ascending → “farthest closest”
results.sort(key=lambda t: t[2])
json.dump(
    results[:1000],
    open("top1000_distant_closest.json", "w", ensure_ascii=False, indent=2),
)
print("✅ top 1000 saved → top1000_distant_closest.json")
