from pathlib import Path
import pandas as pd
import re
from collections import Counter


def load_clean_data(file_path):
    """
    Load cleaned research paper data.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)

    print("Clean data loaded successfully")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


def analyze_papers_by_year(df):
    """
    Count number of papers published each year.
    """

    print("\nPapers by year:")
    papers_by_year = df["year"].value_counts().sort_index()
    print(papers_by_year)

    return papers_by_year


def get_top_words_from_titles(df, top_n=20):
    """
    Find most common words from cleaned paper titles.
    """

    all_titles = " ".join(df["title_clean"].dropna().astype(str))

    words = re.findall(r"\b[a-zA-Z]{3,}\b", all_titles)

    stop_words = {
        "the", "and", "for", "with", "from", "this", "that",
        "using", "based", "towards", "into", "are", "was",
        "retrieval", "augmented", "generation"
    }

    filtered_words = [word for word in words if word not in stop_words]

    word_counts = Counter(filtered_words)

    print(f"\nTop {top_n} words in paper titles:")
    for word, count in word_counts.most_common(top_n):
        print(word, ":", count)

    return word_counts.most_common(top_n)


if __name__ == "__main__":
    clean_file = "data/processed/rag_papers_clean.csv"

    papers_df = load_clean_data(clean_file)

    if papers_df is not None:
        analyze_papers_by_year(papers_df)
        get_top_words_from_titles(papers_df, top_n=20)