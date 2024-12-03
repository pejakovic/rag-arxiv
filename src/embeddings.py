import os
import json
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

class FAISSDatabase:
    def __init__(self, index_file="./faiss_index", model_name="all-MiniLM-L6-v2"):
        """
        Initialize a FAISS database for storing embeddings and metadata.

        :param index_file: Path to the FAISS index file.
        :param model_name: Name of the Hugging Face model used for generating embeddings.
        """
        self.index_file = index_file
        self.embedding_model = SentenceTransformer(model_name, device="cpu")
        self.index = None
        self.texts = []
        self.metadata = []

    def save_to_database(self, dataset_file):
        """
        Save a dataset (in JSONL format) to a FAISS index.
        """
        # Load the dataset
        with open(dataset_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        # Extract texts and metadata
        self.texts = [item["text"] for item in data]
        self.metadata = [item["metadata"] for item in data]

        # Use HuggingFaceEmbeddings with model parameters to set device
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        embeddings = hf_embeddings.embed_documents(self.texts)

        # Create FAISS index using the computed embeddings
        self.index = FAISS.from_embeddings(
            text_embeddings=list(zip(self.texts, embeddings)),
            embedding=hf_embeddings,
            metadatas=self.metadata
        )

        print(f"FAISS index created with {self.index.index.ntotal} entries.")

        # Save the index and auxiliary data
        self.index.save_local(self.index_file)
        with open(f"{self.index_file}_texts.json", "w", encoding="utf-8") as f:
            json.dump(self.texts, f)
        with open(f"{self.index_file}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)

        print(f"Index and data saved to {self.index_file}.")

    def load_database(self):
        """
        Load the FAISS index and auxiliary data.
        """
        if not os.path.exists(self.index_file):
            raise FileNotFoundError("FAISS index file not found. Please save to the database first.")
        
        # Use HuggingFaceEmbeddings with model parameters to set device
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.index = FAISS.load_local(self.index_file, hf_embeddings, allow_dangerous_deserialization=True)

        # Load texts and metadata
        with open(f"{self.index_file}_texts.json", "r", encoding="utf-8") as f:
            self.texts = json.load(f)
        with open(f"{self.index_file}_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"Index and data loaded from {self.index_file}.")

    def search_index(self, query, top_k=5):
        """
        Search the index for similar texts to the query.
        """
        if self.index is None:
            raise ValueError("Index not loaded. Please load the database first.")
            
        results = self.index.similarity_search_with_score(query, k=top_k)
        return [{"text": doc.page_content, "metadata": doc.metadata, "score": score} 
                for doc, score in results]

# Example usage
if __name__ == "__main__":
    db = FAISSDatabase()
    dataset_file = "dataset.jsonl"
    db.save_to_database(dataset_file)
    query = "What is Federated Learning?"
    results = db.search_index(query)
    for result in results:
        print(f"Text: {result['text']}\nMetadata: {result['metadata']}\n")