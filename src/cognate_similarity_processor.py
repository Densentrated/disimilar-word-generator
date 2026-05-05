from lingpy.align.pairwise import Pairwise
from lingpy.sequence.sound_classes import ipa2tokens


def compute_cognate_similarity(
    word1: str, word2: str, sound_class_model: str = "sca"
) -> float:
    """
    Compute phonological similarity between two words using sound classes and alignment.

    This function:
    1. Tokenizes words into phonological segments using IPA notation
    2. Performs pairwise alignment using linguistically informed sound classes
    3. Returns a normalized similarity score (0-1)

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

    # Tokenize the words into phonological segments
    tokens1 = ipa2tokens(word1_clean)
    tokens2 = ipa2tokens(word2_clean)

    # Perform pairwise alignment using lingpy with the specified sound class model
    # Pairwise will handle the conversion to sound classes internally using the model parameter
    try:
        alignment = Pairwise(tokens1, tokens2, model=sound_class_model)
    except ValueError as e:
        # Handle case where tokens contain unknown characters (not IPA notation)
        if "unknown characters" in str(e):
            return 0.0
        raise

    # Get the sound class representations (prostrings)
    # prostrings contains the phonological classes for each sequence
    pstr1, pstr2 = alignment.prostrings[0]

    # Calculate similarity score by comparing sound classes
    matches = 0
    for c1, c2 in zip(pstr1, pstr2):
        if c1 == c2:
            matches += 1

    # Normalize: similarity = matches / max length of the two prostrings
    # Using max length accounts for insertions/deletions in phonological alignment
    total_positions = max(len(pstr1), len(pstr2))

    if total_positions == 0:
        return 0.0

    similarity_score = matches / total_positions

    return similarity_score
