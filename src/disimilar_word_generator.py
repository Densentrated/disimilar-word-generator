import pandas as pd
from sentence_transformers import SentenceTransformer

"""

Analyzes the InputFormat object, and then outputs a dataframe
that contains the markers for the critera of a disimilar word?

"""


# Pre process functions, adds columns that we would need like the embedding of the word, and the embedding of the definition
# Gets the to and from embeddings of the word and the vector
def embed_column(
    data: pd.DataFrame, col_name: str, embedding_col_name: str
) -> pd.DataFrame:
    """Embeds the specified column of the dataframe, returns a dataframe with the embedding column"""
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    prefixed = ["query: " + w for w in data[col_name].tolist()]
    embeddings = model.encode(prefixed, batch_size=256, show_progress_bar=True)
    output = data.copy()
    output[embedding_col_name] = list(embeddings)
    return output


# Translation Similarity
# to definition embedding and distance to nearest from word
# translation count
# back translation loss
# cross lingual vector distance
# semantic field density
# borrowing resistance, does from word exist in to language
