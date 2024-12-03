import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from time import time
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    query: str
    response: str
    source_documents: list
    execution_time: float
    error: Optional[str] = None

class RAGSystem:
    def __init__(self, 
                 index_file: str = "./faiss_index", 
                 model_path: str = "./fine_tuned_model",
                 max_length: int = 1024,
                 max_new_tokens: int = 512,
                 temperature: float = 0.7,
                 top_k: int = 4):
        """
        Initialize the RAG system with the given parameters.

        :param index_file: Path to the FAISS index file.
        :param model_path: Path to the fine-tuned model.
        :param max_length: Maximum length of the generated text.
        :param max_new_tokens: Maximum number of new tokens to generate.
        :param temperature: Sampling temperature for the generated text.
        :param top_k: Number of top-k tokens to consider for the generated text.
        """
        self.index_file = index_file
        self.model_path = model_path
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        logger.info(f"Initialized RAG system with parameters: max_length={max_length}, "
                   f"max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k}")

    def load_components(self):
        """
        Load the components of the RAG system.
        """
        try:
            # Load FAISS index
            logger.info("Loading FAISS index...")
            hf_embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            self.vectorstore = FAISS.load_local(
                self.index_file, 
                hf_embeddings, 
                allow_dangerous_deserialization=True
            )
            
            # Load fine-tuned LLM
            logger.info("Loading fine-tuned LLM...")
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            generation_pipeline = pipeline(
                "text-generation", 
                model=model, 
                tokenizer=tokenizer,
                max_length=self.max_length,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature
            )
            self.llm = HuggingFacePipeline(pipeline=generation_pipeline)
            
            # Build QA chain
            logger.info("Building RetrievalQA chain...")
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": self.top_k}
                ),
                return_source_documents=True
            )
            logger.info("RAG system successfully loaded!")
            
        except Exception as e:
            logger.error(f"Error loading RAG components: {str(e)}")
            raise

    def ask_question(self, query: str) -> QueryResult:
        """
        Processes a query using the RAG system and returns the result.

        This function checks if the RAG system is properly initialized, processes
        the given query using the RetrievalQA chain, and returns the response,
        source documents, and execution time. If an error occurs during processing,
        it logs the error and returns an error message within the QueryResult.

        :param query: The query string to process.
        :return: A QueryResult object containing the query, response, source documents,
                execution time, and any error encountered.
        :raises ValueError: If the RAG system is not initialized.
        """
        if not self.qa_chain:
            raise ValueError("RAG system not initialized. Call load_components() first.")
            
        start_time = time()
        try:
            logger.info(f"Processing query: {query}")
            result = self.qa_chain({"query": query})
            
            execution_time = time() - start_time
            logger.info(f"Query processed in {execution_time:.2f} seconds")
            
            return QueryResult(
                query=query,
                response=result.get('result', ''),
                source_documents=result.get('source_documents', []),
                execution_time=execution_time
            )
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            return QueryResult(
                query=query,
                response="",
                source_documents=[],
                execution_time=time() - start_time,
                error=error_msg
            )

def main():
    # Initialize and use the RAG system
    """
    Initialize and use the RAG system to ask a question and print the response,
    source documents, and any errors encountered.

    :raises Exception: If an error occurs during main execution.
    """
    rag_system = RAGSystem()
    
    try:
        # Step 1: Load Components
        rag_system.load_components()
        
        # Step 2: Ask Questions
        query = "What is Federated Learning?"
        result = rag_system.ask_question(query)
        
        # Print the Response
        print("\nQuery Results:")
        print(f"Query: {result.query}")
        print(f"Response: {result.response}")
        print(f"Execution Time: {result.execution_time:.2f} seconds")
        if result.error:
            print(f"Error: {result.error}")
        
        # Print source documents if available
        if result.source_documents:
            print("\nSource Documents:")
            for i, doc in enumerate(result.source_documents, 1):
                print(f"\n{i}. {doc.metadata.get('title', 'Unknown Title')}")
                print(f"Content: {doc.page_content[:200]}...")
                
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
