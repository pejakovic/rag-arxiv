import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from time import time
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from args import parse_rag_args
from downloader import download_arxiv_papers
from doc_parser import parse_pdfs_to_chunks
from finetuner import fine_tune_model
from embeddings import FAISSDatabase
from utils import PathManager

# Initialize PathManager
path_manager = PathManager()

# Configure logging
def setup_logging(log_file: str):
    """Setup logging with absolute path"""
    log_path = path_manager.get_abs_path(log_file, create_dirs=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_path)),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

@dataclass
class QueryResult:
    query: str
    response: str
    source_documents: list
    execution_time: float
    error: Optional[str] = None

class RAGSystem:
    def __init__(self, args):
        """
        Initialize the RAG system with the given arguments.

        :param args: RAGArguments object containing all configuration
        """
        self.args = args
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.logger = setup_logging(args.log_file)
        
        # Log absolute paths
        self.logger.info(f"Using index file: {args.index_file}")
        self.logger.info(f"Using model path: {args.model_path}")
        self.logger.info(
            f"Initialized RAG system with parameters: max_length={args.max_length}, "
            f"max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, "
            f"top_k={args.top_k}"
        )

    def load_components(self):
        """Load all necessary components for the RAG system"""
        try:
            # Load FAISS index
            self.logger.info("Loading FAISS index...")
            hf_embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                # model_kwargs={'device': 'cpu'}
            )
            self.vectorstore = FAISS.load_local(
                self.args.index_file, 
                hf_embeddings, 
                allow_dangerous_deserialization=True
            )
            
            # Load fine-tuned LLM
            self.logger.info("Loading fine-tuned LLM...")
            model = AutoModelForCausalLM.from_pretrained(self.args.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
            generation_pipeline = pipeline(
                "text-generation", 
                model=model, 
                tokenizer=tokenizer,
                max_length=self.args.max_length,
                max_new_tokens=self.args.max_new_tokens,
                temperature=self.args.temperature
            )
            self.llm = HuggingFacePipeline(pipeline=generation_pipeline)
            
            # Build QA chain
            self.logger.info("Building RetrievalQA chain...")
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": self.args.top_k}
                ),
                return_source_documents=True
            )
            self.logger.info("RAG system successfully loaded!")
            
        except Exception as e:
            self.logger.error(f"Error loading RAG components: {str(e)}")
            raise

    def ask_question(self, query: str) -> QueryResult:
        """Process a query and return the result"""
        if not self.qa_chain:
            raise ValueError("RAG system not initialized. Call load_components() first.")
            
        start_time = time()
        try:
            self.logger.info(f"Processing query: {query}")
            result = self.qa_chain.invoke({"query": query})
            
            execution_time = time() - start_time
            self.logger.info(f"Query processed in {execution_time:.2f} seconds")
            
            return QueryResult(
                query=query,
                response=result.get('result', ''),
                source_documents=result.get('source_documents', []),
                execution_time=execution_time
            )
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self.logger.error(error_msg)
            return QueryResult(
                query=query,
                response="",
                source_documents=[],
                execution_time=time() - start_time,
                error=error_msg
            )

def initialize_system(args):
    """Initialize the system by downloading papers and creating the database"""
    print(f"\nDownloading papers for query: {args.search_query}")
    download_arxiv_papers(args.search_query, args.papers_dir)
    print(f"Papers downloaded to: {args.papers_dir}")

    print("\nParsing PDFs and creating chunks...")
    parse_pdfs_to_chunks(args.papers_dir, args.dataset_path)

def run_inference(rag_system, args):
    """Run the system in inference mode"""
    if args.query:
        # Single query mode
        result = rag_system.ask_question(args.query)
        print_results(result)
    else:
        # Interactive mode
        print("\nEnter your questions (type 'exit' to quit):")
        while True:
            query = input("\nQuestion: ").strip()
            if query.lower() == 'exit':
                break
            result = rag_system.ask_question(query)
            print_results(result)

def print_results(result: QueryResult):
    """Print query results in a formatted way"""
    print("\nQuery Results:")
    print(f"Response: {result.response}")
    print(f"Execution Time: {result.execution_time:.2f} seconds")
    
    if result.error:
        print(f"Error: {result.error}")
    
    if result.source_documents:
        print("\nSource Documents:")
        for i, doc in enumerate(result.source_documents, 1):
            print(f"\n{i}. {doc.metadata.get('title', 'Unknown Title')}")
            print(f"Content: {doc.page_content[:200]}...")

def main():
    """Main function to run the RAG system"""
    args = parse_rag_args()
    
    try:
        if args.mode == 'init':
            initialize_system(args)
        elif args.mode == 'train':
            print("\nFine-tuning the model...")
            fine_tune_model(args.dataset_path, args.model_output_dir)
            print("\nCreating FAISS database...")
            db = FAISSDatabase()
            db.save_to_database(args.dataset_path)
            print("System initialization complete!")
        else:  # inference mode
            rag_system = RAGSystem(args)
            rag_system.load_components()
            run_inference(rag_system, args)
            
    except Exception as e:
        logger = setup_logging(args.log_file)
        logger.error(f"Main execution error: {str(e)}")
        raise

if __name__ == "__main__":
    '''
    # Initialize the system
    python src/rag_arxiv.py --mode init --search-query "federated learning" --papers-dir "data/papers" --dataset-path "data/dataset.jsonl"

    # Train the model
    python src/rag_arxiv.py --mode train --dataset-path "data/dataset.jsonl" --model_output_dir "models/fine_tuned"

    # Run inference (single query)
    python src/rag_arxiv.py --mode inference --query "What is federated learning?"

    # Run inference (interactive mode)
    python src/rag_arxiv.py --mode inference
    '''

    main()