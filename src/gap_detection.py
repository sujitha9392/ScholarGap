import pandas as pd
from pathlib import Path


GAP_PATTERNS = [
    "limitation",
    "limitations",
    "future work",
    "challenge",
    "challenges",
    "not fully explored",
    "open problem",
    "further research",
    "lack of",
    "underexplored",
    "difficult",
    "unclear"
]


def detect_research_gaps(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    gap_rows = []

    for _, row in df.iterrows():
        abstract = str(row["abstract_clean"])

        matched_patterns = []

        for pattern in GAP_PATTERNS:
            if pattern in abstract:
                matched_patterns.append(pattern)

        if matched_patterns:
            gap_rows.append({
                "title": row["title"],
                "year": row["year"],
                "matched_gap_terms": ", ".join(matched_patterns),
                "url": row["url"]
            })

    gap_df = pd.DataFrame(gap_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gap_df.to_csv(output_path, index=False)

    print("Research gap detection completed")
    print(gap_df)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    detect_research_gaps(
        "data/processed/rag_papers_clean.csv",
        "data/processed/research_gaps.csv"
    )