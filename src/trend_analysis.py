import pandas as pd
from pathlib import Path


TREND_TERMS = [
    "evaluation",
    "hallucination",
    "faithfulness",
    "retrieval",
    "benchmark",
    "question",
    "answering",
    "knowledge",
    "reasoning",
    "metrics"
]


def analyze_trends(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    trend_rows = []

    for year in sorted(df["year"].unique()):
        year_df = df[df["year"] == year]

        all_abstracts = " ".join(
            year_df["abstract_clean"].dropna().astype(str)
        )

        for term in TREND_TERMS:
            count = all_abstracts.count(term)

            trend_rows.append({
                "year": year,
                "term": term,
                "count": count
            })

    trend_df = pd.DataFrame(trend_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trend_df.to_csv(output_path, index=False)

    print("Trend analysis completed")
    print(trend_df)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    analyze_trends(
        "data/processed/rag_papers_clean.csv",
        "data/processed/trend_analysis.csv"
    )