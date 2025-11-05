# 3_build_en_pool.py
import fasttext
import json
import numpy as np
from tqdm import tqdm

ft_en = fasttext.load_model("cc.en.300.bin")

# We only need the 200 000 most frequent English words
en_words = sorted(ft_en.words, key=lambda w: ft_en.get_word_frequency(w), reverse=True)[
    :200_000
]
en_vecs = np.vstack(
    [ft_en.get_word_vector(w) for w in tqdm(en_words, desc="loading EN")]
)

np.save("en_matrix.npy", en_vecs)
json.dump(en_words, open("en_words.json", "w"))
print("✅ English candidate pool ready")
