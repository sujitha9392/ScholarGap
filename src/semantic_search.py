import pandas as pd
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=1)
def load_sentence_model():
    """
    Loads a Sentence Transformer model.

    This model converts text into numerical vectors.
    These vectors help compare meaning between user query and research papers.
    """

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def semantic_search_papers(df, user_query, top_k=5):
    """
    Finds the most relevant research papers based on semantic meaning.

    Input:
    - df: cleaned research papers dataframe
    - user_query: search query typed by user
    - top_k: number of top papers to return

    Output:
    - top relevant papers with similarity score
    """

    if df.empty:
        return pd.DataFrame()

    if user_query.strip() == "":
        return pd.DataFrame()

    search_df = df.copy()

    if "combined_text" not in search_df.columns:
        return pd.DataFrame()

    search_df["combined_text"] = search_df["combined_text"].fillna("")

    try:
        model = load_sentence_model()

        paper_embeddings = model.encode(
            search_df["combined_text"].tolist(),
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_embedding = model.encode(
            [user_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        similarity_scores = np.dot(paper_embeddings, query_embedding)

        search_df["similarity_score"] = similarity_scores

    except Exception as error:
        print(f"Semantic model error: {error}")

        return pd.DataFrame()

    result_df = (
        search_df
        .sort_values("similarity_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    result_df.insert(0, "rank", range(1, len(result_df) + 1))

    return result_df[
        [
            "rank",
            "similarity_score",
            "title",
            "abstract",
            "year",
            "category",
            "paper_link"
        ]
    ]