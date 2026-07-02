import requests
import pandas as pd
import xml.etree.ElementTree as ET
from urllib.parse import quote


def fetch_arxiv_papers(query, max_results=50):
    """
    Fetch research papers from arXiv API based on a search query.
    """

    base_url = "http://export.arxiv.org/api/query"

    encoded_query = quote(query)

    url = (
        f"{base_url}?"
        f"search_query=all:{encoded_query}"
        f"&start=0"
        f"&max_results={max_results}"
        f"&sortBy=submittedDate"
        f"&sortOrder=descending"
    )

    response = requests.get(url)

    if response.status_code != 200:
        print("Failed to fetch papers")
        return []

    root = ET.fromstring(response.content)

    namespace = {
        "atom": "http://www.w3.org/2005/Atom"
    }

    papers = []

    for entry in root.findall("atom:entry", namespace):
        title = entry.find("atom:title", namespace).text.strip().replace("\n", " ")
        abstract = entry.find("atom:summary", namespace).text.strip().replace("\n", " ")
        published_date = entry.find("atom:published", namespace).text.strip()
        year = published_date[:4]

        authors_list = []

        for author in entry.findall("atom:author", namespace):
            name = author.find("atom:name", namespace).text
            authors_list.append(name)

        authors = ", ".join(authors_list)

        paper_url = entry.find("atom:id", namespace).text.strip()

        paper = {
            "title": title,
            "abstract": abstract,
            "published_date": published_date,
            "year": year,
            "authors": authors,
            "url": paper_url
        }

        papers.append(paper)

    return papers


def save_papers_to_csv(papers, file_path):
    """
    Save collected paper data into a CSV file.
    """

    df = pd.DataFrame(papers)

    df.to_csv(file_path, index=False)

    print(f"Saved {len(df)} papers to {file_path}")


if __name__ == "__main__":
    search_query = "retrieval augmented generation evaluation"

    papers = fetch_arxiv_papers(search_query, max_results=50)

    save_papers_to_csv(papers, "data/raw/rag_papers_raw.csv")