import os
import time
import arxiv

# Configuration
MAX_RESULTS = 10                     # Maximum number of results to fetch
TIMEOUT = 5                          # Seconds to wait between downloads

def _create_download_dir(download_dir):
    """
    Ensure the download directory exists.

    If the directory does not exist, create it.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

def _download_papers(query: str, max_results: int, download_dir: str, timeout: int):
    """
    Helper function to download papers from arXiv based on the search query.
    """
    _create_download_dir(download_dir)

    # Perform the search
    print(f"Searching for papers with query: '{query}'...")
    search = arxiv.Search(
        query=query,
        max_results=max_results,  # Ensure max_results is an integer
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    client = arxiv.Client()

    # Download each paper
    for result in client.results(search):
        title = result.title.replace("/", "-")  # Clean title for filename
        download_path = os.path.join(download_dir, f"{title}.pdf")
        
        if not os.path.exists(download_path):
            print(f"Downloading: {title}")
            try:
                # Download the PDF
                result.download_pdf(filename=download_path)
                print(f"Saved to: {download_path}")
            except Exception as e:
                print(f"Failed to download {title}: {e}")
            time.sleep(timeout)  # Wait to avoid overwhelming the server
        else:
            print(f"Already downloaded: {title}")

def download_arxiv_papers(query: str, download_dir: str, max_results: int = 10, timeout: int = 5):
    """
    Download papers from arXiv based on the search query.

    :param query: Search query for arXiv
    :param max_results: Maximum number of results to fetch
    :param download_dir: Directory to save the downloaded papers
    :param timeout: Seconds to wait between downloads
    """
    _download_papers(query, int(max_results), download_dir, timeout)
