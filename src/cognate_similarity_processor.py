from difflib import SequenceMatcher

from lingpy.sequence.sound_classes import ipa2tokens, token2class


def compute_cognate_similarity(
    word1: str, word2: str, sound_class_model: str = "sca"
) -> float:
    """
    Compute phonological similarity between two words using sound classes and sequence alignment.

    This function:
    1. Tokenizes words into phonological segments using IPA notation
    2. Maps tokens to sound classes (articulatory features)
    3. Aligns the sound class sequences using difflib SequenceMatcher
    4. Returns a normalized similarity score (0-1)

    Args:
        word1: First word in IPA notation (e.g., "naxt" for German "Nacht")
        word2: Second word in IPA notation
        sound_class_model: Sound class model to use ('sca' for Sound Class Alphabet,
                          'dolgo' for Dolgopolsky, 'asjp' for ASJP). Default: 'sca'

    Returns:
        float: Similarity score between 0.0 (completely different) and 1.0 (identical)

    Example:
        >>> similarity = compute_cognate_similarity("naxt", "naxt")
        >>> print(f"Similarity: {similarity:.2f}")  # Output: 1.0
    """
    # Handle empty inputs
    if not word1 or not word2:
        return 1.0 if word1 == word2 else 0.0

    # Remove spaces from the words before tokenizing
    word1_clean = word1.replace(" ", "")
    word2_clean = word2.replace(" ", "")

    try:
        # Tokenize the words into phonological segments
        tokens1 = ipa2tokens(word1_clean)
        tokens2 = ipa2tokens(word2_clean)

        # Map tokens to sound classes
        # Sound classes represent articulatory features of phonemes
        classes1 = [token2class(token, sound_class_model) for token in tokens1]
        classes2 = [token2class(token, sound_class_model) for token in tokens2]

        # Use SequenceMatcher for sequence alignment
        # This aligns the sound class sequences to find matching segments
        matcher = SequenceMatcher(None, classes1, classes2)

        # Get the similarity ratio
        # The ratio is: 2 * M / T where M is number of matches and T is total length
        similarity_score = matcher.ratio()

        return similarity_score

    except Exception as e:
        # Handle cases where tokenization fails (invalid IPA characters, etc.)
        print(f"Error computing cognate similarity for '{word1}' and '{word2}': {e}")
        return 0.0


if __name__ == "__main__":
    # Example usage
    print("Testing cognate similarity computation...")
    print()

    # Example 1: Identical words (should be 1.0)
    sim1 = compute_cognate_similarity("naxt", "naxt")
    print(f"Similarity('naxt', 'naxt'): {sim1:.3f}")

    # Example 2: Similar words (cognates)
    sim2 = compute_cognate_similarity("naxt", "nacht")
    print(f"Similarity('naxt', 'nacht'): {sim2:.3f}")

    # Example 3: Different words
    sim3 = compute_cognate_similarity("naxt", "moːr")
    print(f"Similarity('naxt', 'moːr'): {sim3:.3f}")

    # Example 4: Empty string handling
    sim4 = compute_cognate_similarity("naxt", "")
    print(f"Similarity('naxt', ''): {sim4:.3f}")
