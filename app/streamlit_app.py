import streamlit as st
import pandas as pd
from pathlib import Path


st.set_page_config(
    page_title="ScholarGap",
    layout="wide"
)


st.title("ScholarGap: Research Gap Discovery and Trend Prediction")

st.write(
    "ScholarGap analyzes research papers to discover keywords, trends, and possible research gaps."
)


clean_data_path = Path("data/processed/rag_papers_clean.csv")
keywords_path = Path("data/processed/top_keywords.csv")
trends_path = Path("data/processed/trend_analysis.csv")
gaps_path = Path("data/processed/research_gaps.csv")


st.sidebar.title("Navigation")

section = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Research Papers",
        "Keywords",
        "Trend Analysis",
        "Research Gaps"
    ]
)


if not clean_data_path.exists():
    st.error("Clean data file not found. Run data cleaning first.")
    st.stop()


df = pd.read_csv(clean_data_path)


if section == "Overview":
    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Papers", df.shape[0])
    col2.metric("Total Years", df["year"].nunique())
    col3.metric("Latest Year", df["year"].max())

    st.subheader("Papers by Year")

    papers_by_year = df["year"].value_counts().sort_index()

    st.bar_chart(papers_by_year)


elif section == "Research Papers":
    st.header("Research Paper Dataset")

    st.dataframe(
        df[["title", "year", "authors", "url"]]
    )


elif section == "Keywords":
    st.header("Top Keywords")

    if keywords_path.exists():
        keywords_df = pd.read_csv(keywords_path)

        st.dataframe(keywords_df)

        st.subheader("Top 15 Keywords")

        top_keywords = keywords_df.head(15)
        st.bar_chart(top_keywords.set_index("keyword")["count"])

    else:
        st.warning("Keyword file not found. Run keyword extraction first.")


elif section == "Trend Analysis":
    st.header("Trend Analysis")

    if trends_path.exists():
        trends_df = pd.read_csv(trends_path)

        selected_term = st.selectbox(
            "Select a research term",
            sorted(trends_df["term"].unique())
        )

        term_df = trends_df[trends_df["term"] == selected_term]
        term_df = term_df.sort_values("year")

        st.line_chart(term_df.set_index("year")["count"])

        st.dataframe(term_df)

    else:
        st.warning("Trend analysis file not found. Run trend analysis first.")


elif section == "Research Gaps":
    st.header("Possible Research Gaps")

    if gaps_path.exists():
        gaps_df = pd.read_csv(gaps_path)

        if gaps_df.empty:
            st.info("No gap pattern matches found in the current dataset.")
        else:
            st.dataframe(gaps_df)

    else:
        st.warning("Research gap file not found. Run gap detection first.")