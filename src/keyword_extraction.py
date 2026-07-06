import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_top_keywords(df, top_n=25):
    """
    Extract top keywords using TF-IDF.
    """

    if df.empty or "combined_text" not in df.columns:
        return pd.DataFrame(columns=["keyword", "score"])

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

    scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    keywords = vectorizer.get_feature_names_out()

    keyword_df = pd.DataFrame({
        "keyword": keywords,
        "score": scores
    })

    keyword_df = keyword_df.sort_values(by="score", ascending=False)
    keyword_df = keyword_df.head(top_n).reset_index(drop=True)

    return keyword_df