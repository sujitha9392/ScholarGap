import re
import pandas as pd
from src.semantic_search import semantic_search_papers


def split_sentences(text):
    """
    Split abstract text into sentences.
    """

    if pd.isna(text):
        return []

    sentences = re.split(r"(?<=[.!?])\s+", str(text))

    cleaned_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()

        if len(sentence) > 30:
            cleaned_sentences.append(sentence)

    return cleaned_sentences


def generate_rag_answer(df, question, top_k=5):
    """
    Generate a simple RAG-style answer.

    RAG means Retrieval Augmented Generation.

    This function:
    1. Retrieves relevant papers using semantic search
    2. Extracts useful evidence from abstracts
    3. Creates an answer based on retrieved evidence
    """

    if df.empty:
        return "No papers available. Please fetch papers first.", pd.DataFrame()

    if question.strip() == "":
        return "Please enter a question.", pd.DataFrame()

    retrieved_papers = semantic_search_papers(
        df,
        user_query=question,
        top_k=top_k
    )

    if retrieved_papers.empty:
        return "No relevant papers found for this question.", pd.DataFrame()

    question_words = set(
        re.sub(r"[^a-zA-Z0-9\s]", " ", question.lower()).split()
    )

    evidence_rows = []

    for _, paper in retrieved_papers.iterrows():

        abstract = paper["abstract"]
        sentences = split_sentences(abstract)

        for sentence in sentences:

            sentence_words = set(
                re.sub(r"[^a-zA-Z0-9\s]", " ", sentence.lower()).split()
            )

            overlap_score = len(question_words.intersection(sentence_words))

            evidence_rows.append(
                {
                    "sentence": sentence,
                    "score": overlap_score,
                    "paper_title": paper["title"],
                    "year": paper["year"],
                    "paper_link": paper["paper_link"]
                }
            )

    evidence_df = pd.DataFrame(evidence_rows)

    if evidence_df.empty:
        answer = (
            "Relevant papers were found, but strong evidence sentences "
            "could not be extracted from the abstracts."
        )
        return answer, retrieved_papers

    evidence_df = evidence_df.sort_values(
        by="score",
        ascending=False
    ).head(5)

    answer_lines = []

    answer_lines.append("Based on the retrieved research papers:")
    answer_lines.append("")

    answer_lines.append("Main findings:")
    answer_lines.append("")

    for _, row in evidence_df.iterrows():
        answer_lines.append(f"- {row['sentence']}")

    answer_lines.append("")
    answer_lines.append("Most relevant papers used:")
    answer_lines.append("")

    for _, row in retrieved_papers.head(3).iterrows():
        answer_lines.append(f"- {row['title']} ({row['year']})")

    answer = "\n".join(answer_lines)

    return answer, retrieved_papers