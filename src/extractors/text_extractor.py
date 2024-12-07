from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextExtractor:
    def __init__(self, chunk_size=800, chunk_overlap=80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def extract(self, pdf_path):
        """Extract and process text from PDF"""
        loader = PyPDFDirectoryLoader(pdf_path)
        documents = loader.load()
        
        # Split into manageable chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(documents) 