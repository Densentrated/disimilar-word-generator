"""
This file processes Vietnamese words and finds their nearest English word
using multilingual embeddings and cosine similarity.
"""

import sys
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import words as nltk_words


def load_english_words() -> List[str]:
    """Load English word list from NLTK"""
    try:
        english_words = set(w.lower() for w in nltk_words.words())
        print(f"Loaded {len(english_words)} English words from NLTK")
        return sorted(english_words)
    except LookupError:
        print("Downloading NLTK words corpus...")
        nltk.download("words", quiet=True)
        english_words = set(w.lower() for w in nltk_words.words())
        print(f"Loaded {len(english_words)} English words from NLTK")
        return sorted(english_words)


def load_vietnamese_words(input_file: str) -> List[str]:
    """Load Vietnamese words from file"""
    with open(input_file, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(words)} Vietnamese words")
    return words


def process_words_in_batches(
    viet_words: List[str],
    english_words: List[str],
    model: SentenceTransformer,
    batch_size: int = 100,
    output_file: str = "extracted_words_with_neighbors.txt",
):
    """Process words in batches to find nearest English neighbors"""

    print(f"\nEmbedding English words in batches...")
    english_embeddings = []
    for i in range(0, len(english_words), batch_size):
        batch = english_words[i : i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        english_embeddings.append(batch_embeddings)
        if (i // batch_size + 1) % 100 == 0:
            print(f"  Embedded {i + len(batch)}/{len(english_words)} English words")

    english_embeddings = np.vstack(english_embeddings)
    print(f"Completed embedding {len(english_words)} English words")

    print(f"\nProcessing Vietnamese words and finding nearest neighbors...")
    with open(output_file, "w", encoding="utf-8") as out_f:
        for i in range(0, len(viet_words), batch_size):
            batch = viet_words[i : i + batch_size]

            # Embed Vietnamese words
            viet_embeddings = model.encode(batch, show_progress_bar=False)

            # Find nearest English word for each Vietnamese word
            for j, viet_word in enumerate(batch):
                viet_emb = viet_embeddings[j : j + 1]

                # Calculate cosine similarity with all English words
                similarities = cosine_similarity(viet_emb, english_embeddings)[0]

                # Find the nearest neighbor
                nearest_idx = np.argmax(similarities)
                nearest_word = english_words[nearest_idx]
                cosine_dist = (
                    1 - similarities[nearest_idx]
                )  # Convert similarity to distance

                # Write to file: vietnamese_word,nearest_english,cosine_distance
                out_f.write(f"{viet_word},{nearest_word},{cosine_dist:.6f}\n")

            if (i // batch_size + 1) % 10 == 0:
                print(
                    f"  Processed {i + len(batch)}/{len(viet_words)} Vietnamese words"
                )

    print(f"\nCompleted! Results saved to {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python word_set_processor.py <input_words.txt> [output_file.txt]")
        print("\nThis script will:")
        print("  1. Load Vietnamese words from input file")
        print("  2. Load English word corpus from NLTK")
        print("  3. Use multilingual sentence-transformers model to embed words")
        print("  4. Find nearest English word for each Vietnamese word")
        print("  5. Output: vietnamese_word,nearest_english,cosine_distance")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = (
        sys.argv[2] if len(sys.argv) > 2 else "extracted_words_with_neighbors.txt"
    )

    print("=" * 60)
    print("Vietnamese Word to English Neighbor Mapper")
    print("=" * 60)

    # Load words
    print("\nStep 1: Loading words...")
    viet_words = load_vietnamese_words(input_file)
    english_words = load_english_words()

    # Load multilingual model
    print("\nStep 2: Loading multilingual embedding model...")
    print("Using: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print("(This may take a moment on first run as it downloads the model)")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("Model loaded successfully!")

    # Process words
    print("\nStep 3: Processing words...")
    process_words_in_batches(
        viet_words=viet_words,
        english_words=english_words,
        model=model,
        batch_size=100,
        output_file=output_file,
    )

    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Check {output_file} for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
