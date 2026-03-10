import os
import pickle

import disimilar_word_generator as dwg
import kaikki_jsonl_input_converter as kjic
import xedict_input_converter as xic


def main():
    kconv = kjic.KaikkiJsonInputConverter()
    econv = xic.XedictInputConverter()
    data = econv.Xedict_To_Input(".data/denisowski_viet_words.txt")
    data_with_embeddings = dwg.embed_column(
        data.to_dataframe(), "From-Language word", "From-Language word Embedding"
    )
    data_with_embeddings = dwg.embed_column(
        data_with_embeddings,
        "To-Language definition",
        "To-Langauge definition Embedding",
    )

    pkl_path = ".data/data_with_embeddings.pkl"
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)

    # Serialize (pickle) the dataframe to disk
    with open(pkl_path, "wb") as f:
        pickle.dump(data_with_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Load the dataframe back into memory
    with open(pkl_path, "rb") as f:
        loaded_df = pickle.load(f)

    print(f"Pickle saved to: {pkl_path}")
    print("Loaded dataframe preview:")
    print(loaded_df.head(4))


if __name__ == "__main__":
    main()
