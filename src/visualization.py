import plotly.express as px


def papers_by_year_chart(papers_by_year):
    fig = px.bar(
        papers_by_year,
        x="year",
        y="paper_count",
        text="paper_count",
        title="Papers by Year"
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Year",
        yaxis_title="Number of Papers"
    )

    return fig


def keyword_bar_chart(keyword_df):
    fig = px.bar(
        keyword_df.sort_values("score"),
        x="score",
        y="keyword",
        orientation="h",
        title="Top Keywords"
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="TF-IDF Score",
        yaxis_title="Keyword"
    )

    return fig


def keyword_trend_chart(trend_df):
    fig = px.line(
        trend_df,
        x="year",
        y="count",
        color="keyword",
        markers=True,
        title="Keyword Trend Over Years"
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Year",
        yaxis_title="Keyword Count"
    )

    return fig