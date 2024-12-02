import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json

# Configuration
INPUT_DIR = ".arxiv_papers"
OUTPUT_FILE = "./dataset.jsonl"

def parse_pdfs(path: str) -> list[Document]:
    """
    Load PDFs from a given directory using PyPDFDirectoryLoader.

    :param path: The path to the directory containing the PDFs.
    :return: A list of Document objects, each representing a page of a PDF.
    """
    print(f"Loading PDFs from: {path}")
    pdf_document_loader = PyPDFDirectoryLoader(path)
    documents = pdf_document_loader.load()
    return documents

def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into smaller chunks. This is done using a
    RecursiveCharacterTextSplitter, which splits the text into chunks
    based on a specified chunk size and overlap.

    The RecursiveTextSplitter splits the text into chunks in a recursive
    manner until the chunk size is reached, or until the entire text is
    processed.

    Args:
        documents (list[Document]): The list of documents to split.

    Returns:
        list[Document]: The list of documents, split into smaller chunks.
    """
    # Split the documents into smaller chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    # Split the documents and return the list.
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    """
    Calculate the chunk ID for each document.

    The chunk ID is a string that contains the source of the document,
    the page number, and the chunk index. The format of the string is
    "data/document.pdf:6:2". This is used to identify the chunk in the
    dataset.

    :param chunks: The list of documents.
    :return: The list of documents with the chunk ID added to the meta-data.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        # Get the source and page number from the meta-data.
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add the chunk ID to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def save_to_jsonl(documents: list[Document], output_file: str) -> None:
    """
    Save documents with metadata to a JSONL file.

    The documents are saved as a JSONL file, which is a file where each line
    is a JSON object. The JSON object contains the text of the page and the
    meta-data associated with the page.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in documents:
            # Create a JSON object for the page.
            entry = {
                # The text of the page.
                "text": doc.page_content,
                # The meta-data associated with the page.
                "metadata": doc.metadata
            }
            f.write(json.dumps(entry) + "\n")
    print(f"Saved dataset to: {output_file}")

if __name__ == "__main__":
    raw_documents = parse_pdfs(INPUT_DIR)
    chunked_documents = split_documents(raw_documents)
    save_to_jsonl(chunked_documents, OUTPUT_FILE)