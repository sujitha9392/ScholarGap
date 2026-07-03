from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
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
    return df


def plot_papers_by_year(df, output_path):
    """
    Create a bar chart showing number of papers by year.
    """

    papers_by_year = df["year"].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    papers_by_year.plot(kind="bar")

    plt.title("Number of RAG Evaluation Papers by Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Papers")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)
    plt.close()

    print(f"Papers by year chart saved to: {output_path}")


def plot_top_title_words(df, output_path, top_n=15):
    """
    Create a bar chart of most common words in paper titles.
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

    top_words = word_counts.most_common(top_n)

    words = [item[0] for item in top_words]
    counts = [item[1] for item in top_words]

    plt.figure(figsize=(10, 6))
    plt.bar(words, counts)

    plt.title("Top Words in RAG Evaluation Paper Titles")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)
    plt.close()

    print(f"Top title words chart saved to: {output_path}")


if __name__ == "__main__":
    clean_file = "data/processed/rag_papers_clean.csv"

    df = load_clean_data(clean_file)

    if df is not None:
        plot_papers_by_year(df, "reports/figures/papers_by_year.png")
        plot_top_title_words(df, "reports/figures/top_title_words.png")