import logging
from pathlib import Path
from rag.multimodal_arxiv_rag import MultimodalArXivRAG, MultimodalTrainingConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize paths
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data"
    model_dir = base_dir / "models"
    
    # Create necessary directories
    data_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    (data_dir / "arxiv_papers").mkdir(exist_ok=True)
    
    # Initialize the RAG system
    rag = MultimodalArXivRAG(
        model_path="gpt2",  # Starting with GPT-2 as base model
        index_path=str(data_dir / "multimodal_index"),
        training_config=MultimodalTrainingConfig(
            learning_rate=2e-5,
            num_train_epochs=3,
            output_dir=str(model_dir / "multimodal_finetuned"),
            logging_dir=str(base_dir / "logs" / "multimodal_training")
        )
    )
    
    # Download and process arXiv papers
    logger.info("Downloading and processing papers...")
    documents = rag.initialize_from_arxiv(
        search_query="multimodal deep learning",
        papers_dir=str(data_dir / "arxiv_papers"),
        max_results=5,  # Start with a small number for testing
        dataset_path=str(data_dir / "multimodal_dataset.jsonl")
    )
    
    # Fine-tune the model
    logger.info("Starting fine-tuning...")
    rag.fine_tune(str(data_dir / "multimodal_dataset.jsonl"))
    
    # Try some queries
    test_queries = [
        "What are the main approaches to multimodal deep learning?",
        "Explain the relationship between text and images in multimodal systems",
        "What are the challenges in multimodal data processing?"
    ]
    
    logger.info("Testing queries...")
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        response = rag.query(query)
        
        print(f"\nAnswer: {response['answer']}")
        print("\nRelevant Tables:")
        for table in response['multimodal_context']['tables']:
            print(f"- {table['content']}")
        print("\nRelevant Charts:")
        for chart in response['multimodal_context']['charts']:
            print(f"- {chart['elements']}")

if __name__ == "__main__":
    main() 