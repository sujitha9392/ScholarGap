import time
import requests
import xml.etree.ElementTree as ET
import pandas as pd


def build_arxiv_query(query, year):
    """
    Build a safe arXiv query.

    Example:
    retrieval augmented generation

    Output:
    all:retrieval AND all:augmented AND all:generation AND submittedDate:[202301010000 TO 202312312359]
    """

    words = query.lower().strip().split()

    search_parts = []

    for word in words:
        clean_word = "".join(char for char in word if char.isalnum())
        if clean_word:
            search_parts.append(f"all:{clean_word}")

    if not search_parts:
        text_query = "all:machine learning"
    else:
        text_query = " AND ".join(search_parts)

    date_filter = f"submittedDate:[{year}01010000 TO {year}12312359]"

    final_query = f"{text_query} AND {date_filter}"

    return final_query


def fetch_arxiv_papers(query, start_year=2020, end_year=2026, papers_per_year=50):
    """
    Fetch research papers from arXiv year-wise.

    Safe version:
    - Handles timeout
    - Handles connection errors
    - Retries each year
    - Skips failed year instead of crashing Streamlit
    - Returns whatever papers are available
    """

    all_papers = []

    base_url = "https://export.arxiv.org/api/query"

    headers = {
        "User-Agent": "ScholarGap/1.0 student research project"
    }

    for year in range(start_year, end_year + 1):

        search_query = build_arxiv_query(query, year)

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": papers_per_year,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }

        print(f"Fetching papers for year: {year}")
        print(f"Search query: {search_query}")

        success = False

        for attempt in range(1, 4):

            try:
                response = requests.get(
                    base_url,
                    params=params,
                    headers=headers,
                    timeout=(10, 120)
                )

                if response.status_code != 200:
                    print(
                        f"Attempt {attempt}: Failed for {year}. "
                        f"Status code: {response.status_code}"
                    )
                    time.sleep(5)
                    continue

                root = ET.fromstring(response.content)

                namespace = {
                    "atom": "http://www.w3.org/2005/Atom"
                }

                entries = root.findall("atom:entry", namespace)

                print(f"Found {len(entries)} papers for {year}")

                for entry in entries:

                    title = entry.find("atom:title", namespace)
                    abstract = entry.find("atom:summary", namespace)
                    published = entry.find("atom:published", namespace)
                    paper_link = entry.find("atom:id", namespace)

                    authors = []

                    for author in entry.findall("atom:author", namespace):
                        name = author.find("atom:name", namespace)
                        if name is not None and name.text:
                            authors.append(name.text)

                    categories = []

                    for category in entry.findall("atom:category", namespace):
                        term = category.attrib.get("term")
                        if term:
                            categories.append(term)

                    paper_data = {
                        "title": title.text.replace("\n", " ").strip()
                        if title is not None and title.text
                        else "",

                        "abstract": abstract.text.replace("\n", " ").strip()
                        if abstract is not None and abstract.text
                        else "",

                        "published_date": published.text[:10]
                        if published is not None and published.text
                        else "",

                        "year": year,

                        "authors": ", ".join(authors),

                        "category": ", ".join(categories),

                        "paper_link": paper_link.text
                        if paper_link is not None and paper_link.text
                        else ""
                    }

                    all_papers.append(paper_data)

                success = True
                break

            except requests.exceptions.ReadTimeout:
                print(f"Attempt {attempt}: arXiv timeout for year {year}. Retrying...")
                time.sleep(8)

            except requests.exceptions.ConnectionError:
                print(f"Attempt {attempt}: Connection error for year {year}. Retrying...")
                time.sleep(8)

            except requests.exceptions.RequestException as error:
                print(f"Attempt {attempt}: Request error for year {year}: {error}")
                time.sleep(8)

            except Exception as error:
                print(f"Attempt {attempt}: Unexpected error for year {year}: {error}")
                time.sleep(8)

        if not success:
            print(f"Skipped year {year} because arXiv did not respond.")

        time.sleep(3)

    df = pd.DataFrame(all_papers)

    return df