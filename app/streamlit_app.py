from pathlib import Path
import sys

import streamlit as st
import pandas as pd
import plotly.express as px


# ============================================================
# Project Root Setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


# ============================================================
# Import Project Modules
# ============================================================

from src.data_collection import fetch_arxiv_papers
from src.data_cleaning import clean_papers_data
from src.keyword_extraction import extract_top_keywords
from src.trend_analysis import get_papers_by_year, get_keyword_trends, interpret_trends
from src.gap_detection import detect_research_gaps
from src.topic_modeling import create_topic_model
from src.forecasting import forecast_keyword_trends
from src.semantic_search import semantic_search_papers
from src.rag_qa import generate_rag_answer

from src.data_loading import (
    create_project_folders,
    save_csv,
    save_text,
    RAW_DIR,
    PROCESSED_DIR
)

from src.visualization import (
    papers_by_year_chart,
    keyword_bar_chart,
    keyword_trend_chart
)


# ============================================================
# Streamlit Page Configuration
# ============================================================

st.set_page_config(
    page_title="ScholarGap",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CSS Styling
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 44px;
        font-weight: 900;
        color: white;
        line-height: 1.2;
        margin-bottom: 10px;
    }

    .sub-title {
        font-size: 18px;
        color: #d1d5db;
        margin-bottom: 35px;
    }

    .section-title {
        font-size: 32px;
        font-weight: 800;
        color: white;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .metric-box {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 18px;
        padding: 22px;
    }

    .metric-label {
        color: #d1d5db;
        font-size: 15px;
        font-weight: 600;
    }

    .metric-value {
        color: white;
        font-size: 40px;
        font-weight: 900;
    }

    .gap-card {
        background-color: #111827;
        border-left: 5px solid #38bdf8;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 14px;
    }

    .answer-box {
        background-color: #111827;
        border: 1px solid #374151;
        border-left: 5px solid #22c55e;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 14px;
        color: white;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Create Required Folders
# ============================================================

create_project_folders()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Research Papers",
        "Keywords",
        "Topic Modeling",
        "Trend Analysis",
        "Trend Forecasting",
        "Semantic Search",
        "RAG Paper Q&A",
        "Research Gaps",
        "About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.header("Search Settings")

query = st.sidebar.text_input(
    "Research Topic",
    value="retrieval augmented generation"
)

start_year = st.sidebar.selectbox(
    "Start Year",
    [2020, 2021, 2022, 2023, 2024, 2025],
    index=0
)

end_year = st.sidebar.selectbox(
    "End Year",
    [2021, 2022, 2023, 2024, 2025, 2026],
    index=5
)

papers_per_year = st.sidebar.slider(
    "Papers Per Year",
    min_value=10,
    max_value=100,
    value=50,
    step=10
)

num_topics = st.sidebar.slider(
    "Number of Topics",
    min_value=3,
    max_value=10,
    value=6,
    step=1
)

fetch_button = st.sidebar.button("Fetch and Analyze Papers")


# ============================================================
# Session State
# ============================================================

if "clean_df" not in st.session_state:
    st.session_state.clean_df = pd.DataFrame()

if "keyword_df" not in st.session_state:
    st.session_state.keyword_df = pd.DataFrame()

if "trend_df" not in st.session_state:
    st.session_state.trend_df = pd.DataFrame()

if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = pd.DataFrame()

if "gap_df" not in st.session_state:
    st.session_state.gap_df = pd.DataFrame()

if "topic_summary_df" not in st.session_state:
    st.session_state.topic_summary_df = pd.DataFrame()

if "paper_topics_df" not in st.session_state:
    st.session_state.paper_topics_df = pd.DataFrame()


# ============================================================
# Fetch and Analyze Papers
# ============================================================

if fetch_button:
    with st.spinner("Fetching papers from arXiv and analyzing..."):

        raw_df = fetch_arxiv_papers(
            query=query,
            start_year=start_year,
            end_year=end_year,
            papers_per_year=papers_per_year
        )

        clean_df = clean_papers_data(raw_df)

        if clean_df.empty:
            st.error("No papers found. Try another research topic.")
            st.stop()

        keyword_df = extract_top_keywords(clean_df, top_n=25)

        if keyword_df.empty:
            st.error("Keywords could not be extracted. Try another research topic.")
            st.stop()

        top_keywords = keyword_df["keyword"].head(8).tolist()

        trend_df = get_keyword_trends(clean_df, top_keywords)

        forecast_df = forecast_keyword_trends(trend_df)

        gap_df = detect_research_gaps(trend_df)

        safe_num_topics = min(num_topics, len(clean_df))

        topic_summary_df, paper_topics_df = create_topic_model(
            clean_df,
            num_topics=safe_num_topics,
            words_per_topic=8
        )

        st.session_state.clean_df = clean_df
        st.session_state.keyword_df = keyword_df
        st.session_state.trend_df = trend_df
        st.session_state.forecast_df = forecast_df
        st.session_state.gap_df = gap_df
        st.session_state.topic_summary_df = topic_summary_df
        st.session_state.paper_topics_df = paper_topics_df

        save_csv(raw_df, RAW_DIR / "papers_raw.csv")
        save_csv(clean_df, PROCESSED_DIR / "papers_clean.csv")
        save_csv(keyword_df, PROCESSED_DIR / "top_keywords.csv")
        save_csv(trend_df, PROCESSED_DIR / "trend_analysis.csv")
        save_csv(forecast_df, PROCESSED_DIR / "trend_forecast.csv")
        save_csv(gap_df, PROCESSED_DIR / "research_gaps.csv")
        save_csv(topic_summary_df, PROCESSED_DIR / "topic_summary.csv")
        save_csv(paper_topics_df, PROCESSED_DIR / "paper_topics.csv")

        save_text(query, PROCESSED_DIR / "latest_topic.txt")
        save_text(end_year, PROCESSED_DIR / "latest_year.txt")

        st.success("Papers fetched and analysis completed successfully.")


# ============================================================
# Load Data from Session State
# ============================================================

df = st.session_state.clean_df
keyword_df = st.session_state.keyword_df
trend_df = st.session_state.trend_df
forecast_df = st.session_state.forecast_df
gap_df = st.session_state.gap_df
topic_summary_df = st.session_state.topic_summary_df
paper_topics_df = st.session_state.paper_topics_df


# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div class="main-title">
        ScholarGap: AI Research Intelligence System
    </div>
    <div class="sub-title">
        Research Gap Discovery and Trend Prediction using NLP, Machine Learning, Topic Modeling, Forecasting, Semantic Search, RAG, and Data Science
    </div>
    """,
    unsafe_allow_html=True
)


if df.empty:
    st.info("Select your topic and click 'Fetch and Analyze Papers' from the sidebar.")
    st.stop()


# ============================================================
# Page 1: Overview
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">Project Overview</div>',
        unsafe_allow_html=True
    )

    total_papers = len(df)
    total_years = df["year"].nunique()
    latest_year = df["year"].max()
    total_topics = topic_summary_df["topic_id"].nunique() if not topic_summary_df.empty else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Total Papers</div>
                <div class="metric-value">{total_papers}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Total Years</div>
                <div class="metric-value">{total_years}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Latest Year</div>
                <div class="metric-value">{latest_year}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Discovered Topics</div>
                <div class="metric-value">{total_topics}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Papers by Year")

    papers_by_year = get_papers_by_year(df)
    fig = papers_by_year_chart(papers_by_year)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Dataset Preview")

    st.dataframe(
        df[["title", "published_date", "year", "category", "paper_link"]].head(10),
        use_container_width=True
    )


# ============================================================
# Page 2: Research Papers
# ============================================================

elif page == "Research Papers":

    st.markdown(
        '<div class="section-title">Collected Research Papers</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df[
            [
                "title",
                "abstract",
                "published_date",
                "year",
                "authors",
                "category",
                "paper_link"
            ]
        ],
        use_container_width=True
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Clean Papers CSV",
        data=csv,
        file_name="scholargap_clean_papers.csv",
        mime="text/csv"
    )


# ============================================================
# Page 3: Keywords
# ============================================================

elif page == "Keywords":

    st.markdown(
        '<div class="section-title">Keyword Extraction using TF-IDF</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This section extracts important keywords from research paper titles and abstracts
        using TF-IDF. Higher score means the keyword is more important in the collected papers.
        """
    )

    st.dataframe(keyword_df, use_container_width=True)

    fig = keyword_bar_chart(keyword_df)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page 4: Topic Modeling
# ============================================================

elif page == "Topic Modeling":

    st.markdown(
        '<div class="section-title">Topic Modeling using LDA</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Topic modeling is an NLP technique used to discover hidden research themes
        from paper titles and abstracts. LDA groups similar research papers into topics
        based on repeated word patterns.
        """
    )

    st.markdown("### Discovered Research Topics")

    st.dataframe(topic_summary_df, use_container_width=True)

    if not topic_summary_df.empty:
        fig = px.bar(
            topic_summary_df,
            x="topic_id",
            y="paper_count",
            text="paper_count",
            title="Number of Papers in Each Topic"
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Topic ID",
            yaxis_title="Paper Count"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Paper-wise Topic Assignment")

    if not paper_topics_df.empty:
        st.dataframe(
            paper_topics_df[
                [
                    "title",
                    "year",
                    "dominant_topic",
                    "topic_confidence",
                    "paper_link"
                ]
            ],
            use_container_width=True
        )


# ============================================================
# Page 5: Trend Analysis
# ============================================================

elif page == "Trend Analysis":

    st.markdown(
        '<div class="section-title">Trend Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This section compares keyword frequency across previous years and current year.
        If the count increases year by year, the topic is considered trending.
        """
    )

    st.dataframe(trend_df, use_container_width=True)

    fig = keyword_trend_chart(trend_df)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Trend Interpretation")

    trend_summary = interpret_trends(trend_df)

    st.dataframe(trend_summary, use_container_width=True)

    for _, row in trend_summary.iterrows():
        keyword = row["keyword"]
        status = row["trend_status"]
        growth = row["growth"]

        if status == "Increasing":
            st.success(f"{keyword} is increasing compared to previous years. Growth: {growth}")
        elif status == "Decreasing":
            st.warning(f"{keyword} is decreasing compared to previous years. Growth: {growth}")
        else:
            st.info(f"{keyword} is stable compared to previous years.")


# ============================================================
# Page 6: Trend Forecasting
# ============================================================

elif page == "Trend Forecasting":

    st.markdown(
        '<div class="section-title">Trend Forecasting</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This section uses Linear Regression to forecast next-year keyword growth.
        It predicts which research topics may increase, decrease, or stay stable.
        """
    )

    st.dataframe(forecast_df, use_container_width=True)

    if not forecast_df.empty:
        fig = px.bar(
            forecast_df,
            x="keyword",
            y="predicted_growth",
            color="forecast_status",
            title="Predicted Keyword Growth for Next Year"
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Keyword",
            yaxis_title="Predicted Growth"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Forecast Interpretation")

        for _, row in forecast_df.head(10).iterrows():
            keyword = row["keyword"]
            forecast_year = row["forecast_year"]
            status = row["forecast_status"]
            growth = row["predicted_growth"]

            if status == "Expected to Grow":
                st.success(
                    f"{keyword} is expected to grow in {forecast_year}. Predicted growth: {growth}"
                )
            elif status == "Expected to Decline":
                st.warning(
                    f"{keyword} may decline in {forecast_year}. Predicted growth: {growth}"
                )
            else:
                st.info(
                    f"{keyword} may stay stable in {forecast_year}. Predicted growth: {growth}"
                )

    else:
        st.info("Forecast data is not available. Please fetch papers again.")


# ============================================================
# Page 7: Semantic Search
# ============================================================

elif page == "Semantic Search":

    st.markdown(
        '<div class="section-title">Semantic Paper Search</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This section uses Sentence Transformers to search research papers based on meaning.
        It can find relevant papers even when the exact keywords are different.
        """
    )

    search_query = st.text_input(
        "Enter your semantic search query",
        value="hallucination reduction in retrieval augmented generation"
    )

    top_k = st.slider(
        "Number of relevant papers to show",
        min_value=3,
        max_value=10,
        value=5,
        step=1
    )

    if st.button("Search Relevant Papers"):

        with st.spinner("Searching papers using semantic similarity..."):

            semantic_results = semantic_search_papers(
                df,
                user_query=search_query,
                top_k=top_k
            )

        if semantic_results.empty:
            st.warning("No semantic search results found.")
        else:
            st.success("Semantic search completed successfully.")

            st.dataframe(
                semantic_results,
                use_container_width=True
            )

            st.markdown("### Top Matching Papers")

            for _, row in semantic_results.iterrows():
                similarity_score = round(float(row["similarity_score"]), 4)

                st.markdown(
                    f"""
                    <div class="gap-card">
                        <b>Rank:</b> {row['rank']}<br>
                        <b>Similarity Score:</b> {similarity_score}<br>
                        <b>Title:</b> {row['title']}<br>
                        <b>Year:</b> {row['year']}<br>
                        <b>Category:</b> {row['category']}<br>
                        <b>Paper Link:</b> {row['paper_link']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# Page 8: RAG Paper Q&A
# ============================================================

elif page == "RAG Paper Q&A":

    st.markdown(
        '<div class="section-title">RAG Paper Q&A</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Ask a question about the collected research papers.
        The system retrieves relevant papers using semantic search and generates
        an evidence-based answer from paper abstracts.
        """
    )

    user_question = st.text_input(
        "Ask your question",
        value="What are the main research gaps in retrieval augmented generation?"
    )

    rag_top_k = st.slider(
        "Number of papers to use for answer",
        min_value=3,
        max_value=10,
        value=5,
        step=1
    )

    if st.button("Generate RAG Answer"):

        with st.spinner("Retrieving papers and generating answer..."):

            answer, retrieved_papers = generate_rag_answer(
                df,
                question=user_question,
                top_k=rag_top_k
            )

        st.markdown("### Generated Answer")

        st.markdown(
            f"""
            <div class="answer-box">
{answer}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Retrieved Papers Used")

        if retrieved_papers.empty:
            st.warning("No papers were retrieved.")
        else:
            st.dataframe(
                retrieved_papers,
                use_container_width=True
            )


# ============================================================
# Page 9: Research Gaps
# ============================================================

elif page == "Research Gaps":

    st.markdown(
        '<div class="section-title">Research Gap Discovery</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This section identifies possible research gaps using keyword count,
        year-wise growth, and topic coverage.
        """
    )

    st.dataframe(gap_df, use_container_width=True)

    st.markdown("### Suggested Research Gaps")

    if "gap_type" not in gap_df.columns:
        st.info("Gap analysis output is not available. Please fetch papers again.")
    else:
        suggested_gaps = gap_df[
            gap_df["gap_type"].isin(["Emerging Research Gap", "Less Explored Topic"])
        ]

        if suggested_gaps.empty:
            st.info("No strong research gap found. Try another topic or increase papers per year.")
        else:
            for _, row in suggested_gaps.head(10).iterrows():
                explanation = row.get(
                    "explanation",
                    "This topic may have low coverage or possible future research opportunity."
                )

                st.markdown(
                    f"""
                    <div class="gap-card">
                        <b>Possible Gap:</b> {row['keyword']}<br>
                        <b>Total Papers:</b> {row['total_papers']}<br>
                        <b>Growth:</b> {row['growth']}<br>
                        <b>Gap Type:</b> {row['gap_type']}<br>
                        <b>Explanation:</b> {explanation}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# Page 10: About Project
# ============================================================

elif page == "About Project":

    st.markdown(
        '<div class="section-title">About ScholarGap</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ### What is ScholarGap?

        ScholarGap is an AI Research Intelligence System that analyzes research papers
        to discover research trends and possible research gaps.

        ### Problem Statement

        Students and researchers often struggle to find good research topics.
        They do not know which areas are trending, saturated, or less explored.

        ScholarGap solves this problem by using NLP and machine learning techniques
        to analyze research papers.

        ### Data Source

        The project uses arXiv research paper data.

        ### Main Features

        - Fetches papers from arXiv
        - Cleans title and abstract text
        - Extracts keywords using TF-IDF
        - Performs topic modeling using LDA
        - Compares keyword frequency year-wise
        - Forecasts next-year keyword growth using Linear Regression
        - Performs semantic search using Sentence Transformers
        - Generates RAG-style answers from retrieved paper abstracts
        - Finds increasing, decreasing, and stable research topics
        - Detects possible research gaps
        - Shows results in an interactive dashboard

        ### Technologies Used

        - Python
        - Streamlit
        - Pandas
        - Regex
        - Scikit-learn
        - TF-IDF
        - LDA Topic Modeling
        - Linear Regression
        - Sentence Transformers
        - Semantic Search
        - RAG-style Q&A
        - Plotly
        - arXiv API

        ### Interview Explanation

        ScholarGap is a Data Science and NLP project that collects research papers
        from arXiv and analyzes their titles, abstracts, categories, and publication dates.
        It uses TF-IDF for keyword extraction, LDA for topic modeling, year-wise trend
        analysis, Linear Regression for forecasting future topic growth, Sentence
        Transformers for semantic paper search, and a RAG-style Q&A module to generate
        evidence-based answers from retrieved paper abstracts. The final output is an
        interactive dashboard that helps researchers understand trending, declining,
        saturated, and less-explored research areas.
        """
    )