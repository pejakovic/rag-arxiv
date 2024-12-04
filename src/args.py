import argparse
from dataclasses import dataclass
from typing import Optional
from utils import PathManager

# Initialize PathManager
path_manager = PathManager()

@dataclass
class RAGArguments:
    # Mode selection
    mode: str  # 'train' or 'inference'
    
    # Training related arguments
    dataset_path: Optional[str]
    train_batch_size: int
    num_train_epochs: int
    learning_rate: float
    output_dir: str
    
    # Inference related arguments
    index_file: str
    model_path: str
    max_length: int
    max_new_tokens: int
    temperature: float
    top_k: int
    log_file: str
    query: Optional[str]
    
    # Data collection arguments
    init: bool
    search_query: Optional[str]
    papers_dir: Optional[str]
    model_output_dir: str

def parse_rag_args() -> RAGArguments:
    """
    Parse command line arguments for the RAG system.
    
    Example usage:
        # Initialize and download papers
        python rag_arxiv.py --mode init --search-query "federated learning" --papers-dir "./papers"
        
        # Train the model
        python rag_arxiv.py --mode train --dataset-path "./data/dataset.jsonl" --output-dir "./models/finetuned"
        
        # Run inference
        python rag_arxiv.py --mode inference --model-path "./models/finetuned" --query "What is federated learning?"
    """
    parser = argparse.ArgumentParser(description="RAG System for arXiv Papers")
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=['init', 'train', 'inference'],
        required=True,
        help="Operation mode: 'init' for data collection, 'train' for fine-tuning, 'inference' for querying"
    )
    
    # Training related arguments
    train_group = parser.add_argument_group('Training Arguments')
    train_group.add_argument(
        "--dataset-path",
        type=str,
        default="data/dataset.jsonl",
        help="Path to the training dataset (JSONL format)"
    )
    train_group.add_argument(
        "--train-batch-size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    train_group.add_argument(
        "--num-train-epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for training"
    )
    train_group.add_argument(
        "--output-dir",
        type=str,
        default="models/fine_tuned",
        help="Directory to save the fine-tuned model"
    )
    
    # Inference related arguments
    inference_group = parser.add_argument_group('Inference Arguments')
    inference_group.add_argument(
        "--index-file",
        type=str,
        default="data/faiss_index",
        help="Path to the FAISS index file"
    )
    inference_group.add_argument(
        "--model-path",
        type=str,
        default="models/fine_tuned",
        help="Path to the fine-tuned model"
    )
    inference_group.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum length of the generated text"
    )
    inference_group.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    inference_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for text generation"
    )
    inference_group.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of documents to retrieve"
    )
    inference_group.add_argument(
        "--log-file",
        type=str,
        default="logs/rag_system.log",
        help="Path to the log file"
    )
    inference_group.add_argument(
        "--query",
        type=str,
        help="Query to process (if not provided, will run in interactive mode)"
    )
    
    # Data collection arguments
    init_group = parser.add_argument_group('Data Collection Arguments')
    init_group.add_argument(
        "--search-query",
        type=str,
        help="Search query for downloading papers"
    )
    init_group.add_argument(
        "--papers-dir",
        type=str,
        default="data/arxiv_papers",
        help="Directory to save downloaded papers"
    )
    init_group.add_argument(
        "--model-output-dir",
        type=str,
        default="models/fine_tuned",
        help="Directory to save the fine-tuned model during initialization"
    )
    
    args = parser.parse_args()
    
    # Convert paths to absolute paths
    abs_paths = {
        'dataset_path': path_manager.get_abs_path(args.dataset_path),
        'output_dir': path_manager.get_abs_path(args.output_dir, create_dirs=True),
        'index_file': path_manager.get_abs_path(args.index_file),
        'model_path': path_manager.get_abs_path(args.model_path),
        'log_file': path_manager.get_abs_path(args.log_file, create_dirs=True),
        'papers_dir': path_manager.get_abs_path(args.papers_dir, create_dirs=True),
        'model_output_dir': path_manager.get_abs_path(args.model_output_dir, create_dirs=True)
    }
    
    # Validate mode-specific required arguments
    if args.mode == 'init' and not args.search_query:
        parser.error("--search-query is required when mode is 'init'")
    elif args.mode == 'inference' and not (args.model_path and args.index_file):
        parser.error("--model-path and --index-file are required when mode is 'inference'")
    
    return RAGArguments(
        mode=args.mode,
        dataset_path=str(abs_paths['dataset_path']),
        train_batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        output_dir=str(abs_paths['output_dir']),
        index_file=str(abs_paths['index_file']),
        model_path=str(abs_paths['model_path']),
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        log_file=str(abs_paths['log_file']),
        query=args.query,
        init=args.mode == 'init',
        search_query=args.search_query,
        papers_dir=str(abs_paths['papers_dir']),
        model_output_dir=str(abs_paths['model_output_dir'])
    )

@dataclass
class DownloadArguments:
    search_query: str
    max_results: int
    output_dir: str
    start_date: Optional[str]
    end_date: Optional[str]

def parse_download_args() -> DownloadArguments:
    """
    Parse command line arguments for the arXiv paper downloader.
    
    Returns:
        DownloadArguments: Parsed arguments for the download system
    """
    parser = argparse.ArgumentParser(description="Download arXiv Papers")
    
    parser.add_argument(
        "--search-query",
        type=str,
        required=True,
        help="Search query for arXiv papers"
    )
    
    parser.add_argument(
        "--max-results",
        type=int,
        default=100,
        help="Maximum number of papers to download"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./papers",
        help="Directory to save downloaded papers"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for paper search (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for paper search (YYYY-MM-DD)"
    )
    
    args = parser.parse_args()
    
    return DownloadArguments(
        search_query=args.search_query,
        max_results=args.max_results,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date
    )

def main():
    """
    Example usage of the argument parsers
    """
    # Example for RAG arguments
    rag_args = parse_rag_args()
    print("\nRAG Arguments:")
    print(f"Index File: {rag_args.index_file}")
    print(f"Model Path: {rag_args.model_path}")
    print(f"Max Length: {rag_args.max_length}")
    print(f"Temperature: {rag_args.temperature}")
    
    # Example for Download arguments
    download_args = parse_download_args()
    print("\nDownload Arguments:")
    print(f"Search Query: {download_args.search_query}")
    print(f"Max Results: {download_args.max_results}")
    print(f"Output Directory: {download_args.output_dir}")

if __name__ == "__main__":
    main() 