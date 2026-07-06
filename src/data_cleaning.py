import re
import pandas as pd


def clean_text(text):
    """
    Clean text for NLP analysis.
    """

    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_papers_data(df):
    """
    Clean title and abstract columns.
    """

    if df.empty:
        return df

    df = df.copy()

    df = df.drop_duplicates(subset=["title", "abstract"])
    df = df.dropna(subset=["title", "abstract"])

    df["clean_title"] = df["title"].apply(clean_text)
    df["clean_abstract"] = df["abstract"].apply(clean_text)
    df["combined_text"] = df["clean_title"] + " " + df["clean_abstract"]

    return df.reset_index(drop=True)