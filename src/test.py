import os
import pickle
import sys
from typing import Any

import pandas as pd

# Add src directory to path to allow imports
sys.path.insert(0, os.path.dirname(__file__))

import dataframe_processor as dfp
import kaikki_jsonl_input_converter as kjic


def create_sample_data() -> pd.DataFrame:
    """
    Create a sample DataFrame with IPA words for testing.

    Returns a DataFrame with From-Language words and To-Language definitions.
    """
    data = {
        "From-Language word": [
            "naxt",  # German "Nacht" (night)
            "bunt",  # German "Bund" (league/union)
            "haus",  # German "Haus" (house)
            "tsaɪt",  # German "Zeit" (time)
            "vɪnt",  # German "Wind" (wind)
            "baɪn",  # German "Bein" (leg)
            "hand",  # German "Hand" (hand)
            "kopt",  # German "Kopf" (head)
            "fuːs",  # German "Fuß" (foot)
            "oks",  # German "Ochs" (ox)
            "vaser",  # German "Wasser" (water)
            "vaɪn",  # German "Wein" (wine)
        ],
        "To-Language definition": [
            "nacht",  # Dutch "nacht" (night)
            "bond",  # Dutch "bond" (league)
            "huis",  # Dutch "huis" (house)
            "tɪɪt",  # Similar to German Zeit
            "vɪnd",  # Similar to German Wind
            "been",  # Dutch "been" (leg)
            "hand",  # Dutch "hand" (hand)
            "kop",  # Dutch "kop" (head)
            "voet",  # Dutch "voet" (foot)
            "os",  # Dutch "os" (ox)
            "water",  # Dutch "water" (water)
            "wijn",  # Dutch "wijn" (wine)
        ],
    }
    return pd.DataFrame(data)


def process_first_n(input_obj: Any, n: int = 10) -> pd.DataFrame:
    """
    Take an Input-like object that exposes `to_dataframe()` and perform
    feature computations on only the first `n` rows.

    Returns the resulting DataFrame with:
    - Embeddings for word and definition
    - Cosine distance between embeddings
    - Word count of definition
    - Orthographic similarity between word and definition
    - Cognate similarity between word and definition (phonological)
    """
    df: pd.DataFrame = input_obj.to_dataframe()
    df_first: pd.DataFrame = df.iloc[2:].head(n).copy()

    # Embed the "From-Language word" column
    df_first = dfp.embed_column(
        df_first, "From-Language word", "From-Language word Embedding"
    )

    # Embed the "To-Language definition" column
    df_first = dfp.embed_column(
        df_first, "To-Language definition", "To-Language definition Embedding"
    )

    # Compute cosine distance between the two embeddings
    df_first = dfp.add_cosine_distance(
        df_first,
        "From-Language word Embedding",
        "To-Language definition Embedding",
        out_col="cosine_distance",
    )

    # Add word count of definition
    df_first = dfp.add_word_count_column(
        df_first,
        "To-Language definition",
        output_column="definition_word_count",
    )

    # Add orthographic similarity between word and definition
    df_first = dfp.add_orthographic_similarity(
        df_first,
        "From-Language word",
        "To-Language definition",
        output_column="orthographic_similarity",
    )

    # Add cognate similarity between word and definition (phonological similarity)
    df_first = dfp.add_cognate_similarity(
        df_first,
        "From-Language word",
        "To-Language definition",
        output_column="cognate_similarity",
        sound_class_model="sca",
    )

    return df_first


def process_sample_data(n: int = 10) -> pd.DataFrame:
    """
    Create sample data and process it with feature computations.

    Returns the resulting DataFrame with all computed features.
    """
    df: pd.DataFrame = create_sample_data()
    df_first: pd.DataFrame = df.head(n).copy()

    # Embed the "From-Language word" column
    df_first = dfp.embed_column(
        df_first, "From-Language word", "From-Language word Embedding"
    )

    # Embed the "To-Language definition" column
    df_first = dfp.embed_column(
        df_first, "To-Language definition", "To-Language definition Embedding"
    )

    # Compute cosine distance between the two embeddings
    df_first = dfp.add_cosine_distance(
        df_first,
        "From-Language word Embedding",
        "To-Language definition Embedding",
        out_col="cosine_distance",
    )

    # Add word count of definition
    df_first = dfp.add_word_count_column(
        df_first,
        "To-Language definition",
        output_column="definition_word_count",
    )

    # Add orthographic similarity between word and definition
    df_first = dfp.add_orthographic_similarity(
        df_first,
        "From-Language word",
        "To-Language definition",
        output_column="orthographic_similarity",
    )

    # Add cognate similarity between word and definition (phonological similarity)
    df_first = dfp.add_cognate_similarity(
        df_first,
        "From-Language word",
        "To-Language definition",
        output_column="cognate_similarity",
        sound_class_model="sca",
    )

    return df_first


def main() -> None:
    print("Testing cognate similarity computation with sample data...")
    print()

    # Process sample data with all features (first 10 words, ignoring head)
    data_with_features: pd.DataFrame = process_sample_data(n=10)

    pkl_path: str = ".data/data_with_features_sample.pkl"
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)

    # Serialize (pickle) the dataframe to disk
    with open(pkl_path, "wb") as f:
        pickle.dump(data_with_features, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Load the dataframe back into memory
    with open(pkl_path, "rb") as f:
        loaded_df: pd.DataFrame = pickle.load(f)

    print(f"Pickle saved to: {pkl_path}")
    print("\n" + "=" * 150)
    print("DataFrame with computed features (10 sample words)")
    print("=" * 150)

    # Select relevant columns to display
    display_cols = [
        "From-Language word",
        "To-Language definition",
        "cosine_distance",
        "definition_word_count",
        "orthographic_similarity",
        "cognate_similarity",
    ]

    # Only show columns that exist
    available_cols = [col for col in display_cols if col in loaded_df.columns]

    # Display with better formatting
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 25)
    pd.set_option("display.width", None)

    print(loaded_df[available_cols].to_string(index=True))
    print("=" * 150)
    print("\nFeature Summary:")
    print(
        f"  - cosine_distance: Semantic similarity from embeddings (0-1, lower is more similar)"
    )
    print(f"  - definition_word_count: Number of words in the definition")
    print(
        f"  - orthographic_similarity: Character-level similarity (0-1, higher is more similar)"
    )
    print(
        f"  - cognate_similarity: Phonological similarity using sound classes (0-1, higher is more similar)"
    )
    print()


def test_vietnamese_data() -> None:
    """Load Vietnamese data and process first 10 words (ignoring head)."""
    print("Loading Vietnamese data...")
    print()

    converter = kjic.KaikkiJsonInputConverter()
    vietnamese_input = converter.Kaikki_To_Input(
        ".data/kaikki.org-dictionary-Vietnamese.jsonl"
    )

    # Get the dataframe and skip header row, then take first 10
    df = vietnamese_input.to_dataframe()
    df_first = df.iloc[1:11].copy()  # Skip header (index 0), get rows 1-10

    # Embed the "From-Language word" column
    df_first = dfp.embed_column(
        df_first, "From-Language word", "From-Language word Embedding"
    )

    # Embed the "To-Language definition" column
    df_first = dfp.embed_column(
        df_first, "To-Language definition", "To-Language definition Embedding"
    )

    # Compute cosine distance between the two embeddings
    df_first = dfp.add_cosine_distance(
        df_first,
        "From-Language word Embedding",
        "To-Language definition Embedding",
        out_col="cosine_distance",
    )

    # Add word count of definition
    df_first = dfp.add_word_count_column(
        df_first,
        "To-Language definition",
        output_column="definition_word_count",
    )

    # Add orthographic similarity between word and definition
    df_first = dfp.add_orthographic_similarity(
        df_first,
        "From-Language word",
        "To-Language definition",
        output_column="orthographic_similarity",
    )

    # Add cognate similarity between word and definition (phonological similarity)
    df_first = dfp.add_cognate_similarity(
        df_first,
        "From-Language word",
        "To-Language definition",
        output_column="cognate_similarity",
        sound_class_model="sca",
    )

    data_with_features = df_first

    pkl_path: str = ".data/vietnamese_data_with_features.pkl"
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)

    # Serialize the dataframe
    with open(pkl_path, "wb") as f:
        pickle.dump(data_with_features, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Load back
    with open(pkl_path, "rb") as f:
        loaded_df: pd.DataFrame = pickle.load(f)

    print(f"Pickle saved to: {pkl_path}")
    print("\n" + "=" * 150)
    print("Vietnamese DataFrame with computed features (10 words, skipping head)")
    print("=" * 150)

    # Select relevant columns
    display_cols = [
        "From-Language word",
        "To-Language definition",
        "cosine_distance",
        "definition_word_count",
        "orthographic_similarity",
        "cognate_similarity",
    ]

    # Only show columns that exist
    available_cols = [col for col in display_cols if col in loaded_df.columns]

    # Display with formatting
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.width", None)

    print(loaded_df[available_cols].to_string(index=True))
    print("=" * 150)
    print("\nFeature Summary:")
    print(
        f"  - cosine_distance: Semantic similarity from embeddings (0-1, lower is more similar)"
    )
    print(f"  - definition_word_count: Number of words in the definition")
    print(
        f"  - orthographic_similarity: Character-level similarity (0-1, higher is more similar)"
    )
    print(
        f"  - cognate_similarity: Phonological similarity using sound classes (0-1, higher is more similar)"
    )
    print()


if __name__ == "__main__":
    test_vietnamese_data()
