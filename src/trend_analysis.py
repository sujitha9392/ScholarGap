import pandas as pd


def get_papers_by_year(df):
    """
    Count papers year-wise.
    """

    if df.empty:
        return pd.DataFrame(columns=["year", "paper_count"])

    return (
        df.groupby("year")
        .size()
        .reset_index(name="paper_count")
        .sort_values("year")
    )


def get_keyword_trends(df, keywords):
    """
    Count keyword occurrence year-wise.
    """

    trend_data = []

    for year in sorted(df["year"].unique()):
        year_df = df[df["year"] == year]

        for keyword in keywords:
            count = year_df["combined_text"].str.contains(
                keyword,
                case=False,
                na=False,
                regex=False
            ).sum()

            trend_data.append({
                "year": year,
                "keyword": keyword,
                "count": int(count)
            })

    return pd.DataFrame(trend_data)


def interpret_trends(trend_df):
    """
    Give simple trend interpretation.
    """

    results = []

    for keyword in trend_df["keyword"].unique():
        temp = trend_df[trend_df["keyword"] == keyword].sort_values("year")

        if len(temp) < 2:
            continue

        first_count = temp.iloc[0]["count"]
        latest_count = temp.iloc[-1]["count"]
        growth = latest_count - first_count

        if growth > 0:
            status = "Increasing"
        elif growth < 0:
            status = "Decreasing"
        else:
            status = "Stable"

        results.append({
            "keyword": keyword,
            "first_year_count": int(first_count),
            "latest_year_count": int(latest_count),
            "growth": int(growth),
            "trend_status": status
        })

    return pd.DataFrame(results)