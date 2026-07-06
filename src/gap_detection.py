import pandas as pd


def detect_research_gaps(trend_df):
    """
    Detect possible research gaps.

    Logic:
    Less total papers + positive growth = emerging research gap.
    """

    if trend_df.empty:
        return pd.DataFrame()

    results = []

    total_counts = trend_df.groupby("keyword")["count"].sum()
    low_cutoff = max(3, total_counts.quantile(0.35))

    for keyword in trend_df["keyword"].unique():
        temp = trend_df[trend_df["keyword"] == keyword].sort_values("year")

        total_papers = temp["count"].sum()
        first_count = temp.iloc[0]["count"]
        latest_count = temp.iloc[-1]["count"]
        growth = latest_count - first_count

        if total_papers <= low_cutoff and growth > 0:
            gap_type = "Emerging Research Gap"
            explanation = "This topic has low research count but is growing in recent years."
        elif total_papers <= low_cutoff:
            gap_type = "Less Explored Topic"
            explanation = "This topic has fewer papers compared to other topics."
        elif growth > 0:
            gap_type = "Trending Topic"
            explanation = "This topic is growing compared to previous years."
        elif growth < 0:
            gap_type = "Declining Topic"
            explanation = "This topic is decreasing compared to previous years."
        else:
            gap_type = "Stable / Saturated Topic"
            explanation = "This topic is not showing strong growth."

        results.append({
            "keyword": keyword,
            "total_papers": int(total_papers),
            "first_year_count": int(first_count),
            "latest_year_count": int(latest_count),
            "growth": int(growth),
            "gap_type": gap_type,
            "explanation": explanation
        })

    gap_df = pd.DataFrame(results)
    gap_df = gap_df.sort_values(by=["gap_type", "growth"], ascending=[True, False])

    return gap_df.reset_index(drop=True)