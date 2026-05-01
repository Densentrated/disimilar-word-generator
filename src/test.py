import os
import pickle
from typing import Any

import pandas as pd

import dataframe_processor as dfp
import kaikki_jsonl_input_converter as kjic
import xedict_input_converter as xic


def process_first_n(input_obj: Any, n: int = 10) -> pd.DataFrame:
    """
    Take an Input-like object that exposes `to_dataframe()` and perform
    feature computations on only the first `n` rows.

    Returns the resulting DataFrame with:
    - Embeddings for word and definition
    - Cosine distance between embeddings
    - Word count of definition
    - Orthographic similarity between word and definition
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

    return df_first


def main() -> None:
    kconv: kjic.KaikkiJsonInputConverter = kjic.KaikkiJsonInputConverter()
    econv: xic.XedictInputConverter = xic.XedictInputConverter()

    data: Any = econv.Xedict_To_Input(".data/denisowski_viet_words.txt")

    # Process 10 words starting from row 2 (skipping first 2 rows)
    data_with_features: pd.DataFrame = process_first_n(data, n=10)

    pkl_path: str = ".data/data_with_features_first10.pkl"
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)

    # Serialize (pickle) the dataframe to disk
    with open(pkl_path, "wb") as f:
        pickle.dump(data_with_features, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Load the dataframe back into memory
    with open(pkl_path, "rb") as f:
        loaded_df: pd.DataFrame = pickle.load(f)

    print(f"Pickle saved to: {pkl_path}")
    print("\n" + "=" * 120)
    print("DataFrame with computed features (10 words, skipping first 2 rows)")
    print("=" * 120)

    # Select relevant columns to display
    display_cols = [
        "From-Language word",
        "To-Language definition",
        "cosine_distance",
        "definition_word_count",
        "orthographic_similarity",
    ]

    # Only show columns that exist
    available_cols = [col for col in display_cols if col in loaded_df.columns]

    # Display with better formatting
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 20)
    pd.set_option("display.width", None)

    print(loaded_df[available_cols].to_string(index=True))
    print("=" * 120)


if __name__ == "__main__":
    main()
