from pathlib import Path
import pandas as pd


def load_paper_data(file_path):
    """
    Load research paper data from a CSV file.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)

    print("Data loaded successfully")
    print("------------------------")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    return df


def inspect_paper_data(df):
    """
    Inspect basic information about the paper dataset.
    """

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 papers:")
    print(df[["title", "year", "url"]].head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate titles:")
    print(df["title"].duplicated().sum())

    print("\nPapers by year:")
    print(df["year"].value_counts().sort_index())


if __name__ == "__main__":
    data_path = "data/raw/rag_papers_raw.csv"

    papers_df = load_paper_data(data_path)

    if papers_df is not None:
        inspect_paper_data(papers_df)