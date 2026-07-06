import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def create_topic_model(df, num_topics=6, words_per_topic=8):
    """
    Creates research topics from paper titles and abstracts using LDA.

    LDA means Latent Dirichlet Allocation.
    It is an NLP technique used for topic modeling.
    """

    if df.empty or "combined_text" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    texts = df["combined_text"].fillna("").tolist()

    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=1500,
        ngram_range=(1, 2),
        min_df=1
    )

    document_term_matrix = vectorizer.fit_transform(texts)

    lda_model = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42,
        learning_method="batch"
    )

    topic_distribution = lda_model.fit_transform(document_term_matrix)

    feature_names = vectorizer.get_feature_names_out()

    topic_rows = []

    for topic_id, topic_weights in enumerate(lda_model.components_):
        top_word_indices = topic_weights.argsort()[-words_per_topic:][::-1]
        top_words = [feature_names[i] for i in top_word_indices]

        topic_rows.append({
            "topic_id": topic_id + 1,
            "topic_keywords": ", ".join(top_words)
        })

    topic_summary_df = pd.DataFrame(topic_rows)

    paper_topics_df = df.copy()
    paper_topics_df["dominant_topic"] = topic_distribution.argmax(axis=1) + 1
    paper_topics_df["topic_confidence"] = topic_distribution.max(axis=1)

    topic_counts = (
        paper_topics_df["dominant_topic"]
        .value_counts()
        .reset_index()
    )

    topic_counts.columns = ["topic_id", "paper_count"]

    topic_summary_df = topic_summary_df.merge(
        topic_counts,
        on="topic_id",
        how="left"
    )

    topic_summary_df["paper_count"] = topic_summary_df["paper_count"].fillna(0).astype(int)

    return topic_summary_df, paper_topics_df