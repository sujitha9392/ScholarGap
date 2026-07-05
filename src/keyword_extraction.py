from pathlib import Path
import pandas as pd
import re
from collections import Counter


def load_clean_data(file_path):
    """
    Load cleaned paper data from CSV.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)

    print("Clean data loaded successfully")
    print(f"Rows: {df.shape[0]}")

    return df


def extract_keywords_from_abstracts(df, top_n=30):
    """
    Extract most common useful keywords from paper abstracts.
    """

    all_abstracts = " ".join(df["abstract_clean"].dropna().astype(str))

    words = re.findall(r"\b[a-zA-Z]{3,}\b", all_abstracts)

    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "are",
        "was", "were", "has", "have", "had", "our", "their", "into",
        "using", "based", "such", "can", "may", "also", "than", "then",
        "these", "those", "been", "more", "which", "while", "where",
        "between", "through", "during", "however", "therefore", "paper",
        "study", "approach", "method", "methods", "results", "show"
    }

    filtered_words = [word for word in words if word not in stop_words]

    keyword_counts = Counter(filtered_words)

    top_keywords = keyword_counts.most_common(top_n)

    print(f"\nTop {top_n} keywords from abstracts:")
    for word, count in top_keywords:
        print(word, ":", count)

    return top_keywords


def save_keywords_to_csv(keywords, output_path):
    """
    Save extracted keywords to CSV.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_keywords = pd.DataFrame(keywords, columns=["keyword", "count"])

    df_keywords.to_csv(output_path, index=False)

    print(f"\nKeywords saved to: {output_path}")


if __name__ == "__main__":
    clean_file = "data/processed/rag_papers_clean.csv"
    output_file = "data/processed/top_keywords.csv"

    papers_df = load_clean_data(clean_file)

    if papers_df is not None:
        keywords = extract_keywords_from_abstracts(papers_df, top_n=30)
        save_keywords_to_csv(keywords, output_file)