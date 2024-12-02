import arxiv
import time
import os

# Configuration
SEARCH_QUERY = "Federated Learning"  # Keywords for the search
DOWNLOAD_DIR = ".arxiv_papers"       # Directory to save the downloaded papers
MAX_RESULTS = 10                     # Maximum number of results to fetch
TIMEOUT = 5                          # Seconds to wait between downloads

def create_download_dir(directory):
    """
    Ensure the download directory exists.

    If the directory does not exist, create it.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_papers(query, max_results, download_dir, timeout):
    """
    Search and download papers from ArXiv.

    :param query: Keywords for the search
    :param max_results: Maximum number of results to fetch
    :param download_dir: Directory to save the downloaded papers
    :param timeout: Seconds to wait between downloads
    """
    create_download_dir(download_dir)

    # Perform the search
    print(f"Searching for papers with query: '{query}'...")
    search = arxiv.Search(
        query=query,
        max_results=max_results,
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

if __name__ == "__main__":
    download_papers(SEARCH_QUERY, MAX_RESULTS, DOWNLOAD_DIR, TIMEOUT)