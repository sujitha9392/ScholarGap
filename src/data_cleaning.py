from pathlib import Path
import pandas as pd
import re


def clean_text(text):
    """
    Clean text by lowercasing, removing extra spaces, and removing unwanted symbols.
    """

    text = str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def clean_paper_data(input_path, output_path):
    """
    Load raw paper data, clean it, and save processed data.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"File not found: {input_path}")
        return None

    df = pd.read_csv(input_path)

    print("Raw data loaded")
    print(f"Rows before cleaning: {df.shape[0]}")

    df = df.dropna(subset=["title", "abstract"])

    df = df.drop_duplicates(subset=["title"])

    df["title_clean"] = df["title"].apply(clean_text)
    df["abstract_clean"] = df["abstract"].apply(clean_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("Data cleaning completed")
    print(f"Rows after cleaning: {df.shape[0]}")
    print(f"Clean data saved to: {output_path}")

    return df


if __name__ == "__main__":
    raw_file = "data/raw/rag_papers_raw.csv"
    clean_file = "data/processed/rag_papers_clean.csv"

    clean_paper_data(raw_file, clean_file)