import time
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote
import pandas as pd


def fetch_arxiv_papers(query, start_year=2020, end_year=2026, papers_per_year=50):
    """
    Fetch research papers from arXiv year-wise.
    """

    all_papers = []

    for year in range(start_year, end_year + 1):
        search_query = quote(
            f'all:"{query}" AND submittedDate:[{year}01010000 TO {year}12312359]'
        )

        url = (
            "http://export.arxiv.org/api/query?"
            f"search_query={search_query}"
            f"&start=0"
            f"&max_results={papers_per_year}"
            f"&sortBy=submittedDate"
            f"&sortOrder=descending"
        )

        response = requests.get(url, timeout=30)

        if response.status_code != 200:
            print(f"Failed to fetch papers for {year}")
            continue

        root = ET.fromstring(response.content)

        namespace = {
            "atom": "http://www.w3.org/2005/Atom"
        }

        entries = root.findall("atom:entry", namespace)

        for entry in entries:
            title = entry.find("atom:title", namespace)
            abstract = entry.find("atom:summary", namespace)
            published = entry.find("atom:published", namespace)
            paper_link = entry.find("atom:id", namespace)

            authors = []
            for author in entry.findall("atom:author", namespace):
                name = author.find("atom:name", namespace)
                if name is not None:
                    authors.append(name.text)

            categories = []
            for category in entry.findall("atom:category", namespace):
                term = category.attrib.get("term")
                if term:
                    categories.append(term)

            all_papers.append({
                "title": title.text.replace("\n", " ").strip() if title is not None else "",
                "abstract": abstract.text.replace("\n", " ").strip() if abstract is not None else "",
                "published_date": published.text[:10] if published is not None else "",
                "year": year,
                "authors": ", ".join(authors),
                "category": ", ".join(categories),
                "paper_link": paper_link.text if paper_link is not None else ""
            })

        time.sleep(1)

    return pd.DataFrame(all_papers)