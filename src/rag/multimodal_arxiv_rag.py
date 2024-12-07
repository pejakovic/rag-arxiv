from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

from multimodal_processor import MultimodalProcessor
from embeddings.multimodal_embedder import MultimodalEmbedder
from embeddings.cross_modal_attention import CrossModalAttention
from downloader import download_arxiv_papers
from doc_parser import parse_pdfs_to_chunks
from dataset_handler import get_training_and_validation_datasets
from monitoring import (
    monitor_time, monitor_errors, QUERY_COUNTER, 
    QUERY_DURATION, DOC_RETRIEVAL_DURATION
)

@dataclass
class MultimodalTrainingConfig:
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    output_dir: str = "models/multimodal_finetuned"
    logging_dir: str = "logs/multimodal_training"

class MultimodalArXivRAG:
    def __init__(
        self, 
        model_path: str, 
        index_path: str,
        training_config: Optional[MultimodalTrainingConfig] = None
    ):
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize processors and embedders
        self.processor = MultimodalProcessor()
        self.embedder = MultimodalEmbedder()
        self.cross_attention = CrossModalAttention()
        
        # Initialize RAG components
        self.retriever = self._init_retriever(index_path)
        self.generator = self._init_generator(model_path)
        self.qa_chain = self._create_qa_chain()
        
        # Training configuration
        self.training_config = training_config or MultimodalTrainingConfig()
        
        # Model and tokenizer for fine-tuning
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    def initialize_from_arxiv(
        self, 
        search_query: str,
        papers_dir: str,
        max_results: int = 100,
        dataset_path: str = "data/dataset.jsonl"
    ):
        """Initialize the system by downloading and processing arXiv papers"""
        self.logger.info(f"Downloading papers for query: {search_query}")
        
        # Download papers
        download_arxiv_papers(
            query=search_query,
            download_dir=papers_dir,
            max_results=max_results
        )
        
        # Process PDFs and create multimodal chunks
        self.logger.info("Processing PDFs and creating multimodal chunks...")
        documents = []
        for pdf_path in Path(papers_dir).glob("*.pdf"):
            content, embeddings = self.processor.process_document(str(pdf_path))
            documents.append({
                'content': content,
                'embeddings': embeddings,
                'source': str(pdf_path)
            })
            
        # Save processed documents
        self._save_processed_documents(documents, dataset_path)
        
        return documents
    
    def fine_tune(self, dataset_path: str):
        """Fine-tune the model on multimodal data"""
        self.logger.info("Starting multimodal fine-tuning...")
        
        # Load and prepare datasets
        train_dataset, eval_dataset = get_training_and_validation_datasets(dataset_path)
        
        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            learning_rate=self.training_config.learning_rate,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            logging_dir=self.training_config.logging_dir,
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer
        )
        
        # Train the model
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.training_config.output_dir)
        
        # Update the generator with fine-tuned model
        self.generator = self._init_generator(self.training_config.output_dir)
        self.qa_chain = self._create_qa_chain()
        
        self.logger.info("Fine-tuning completed successfully!")
    
    @monitor_time(QUERY_DURATION)
    @monitor_errors
    def query(self, question: str) -> Dict[str, Any]:
        """Process a query with multimodal context"""
        QUERY_COUNTER.inc()
        
        with DOC_RETRIEVAL_DURATION.time():
            # Get relevant documents with multimodal context
            docs = self.retriever.similarity_search(question, k=4)
        
        # Extract multimodal elements from retrieved docs
        multimodal_context = self._extract_multimodal_context(docs)
        
        # Apply cross-modal attention
        enhanced_query = self._enhance_query_with_multimodal(
            question, multimodal_context)
        
        # Generate answer
        response = self.qa_chain({"query": enhanced_query})
        
        return {
            'answer': response['result'],
            'sources': self._format_sources(response['source_documents']),
            'multimodal_context': multimodal_context
        }
    
    # ... (keep existing helper methods from MultimodalRAG)
    
    def _save_processed_documents(self, documents: List[Dict], output_path: str):
        """Save processed documents to disk"""
        import json
        
        with open(output_path, 'w') as f:
            for doc in documents:
                # Convert embeddings to list for JSON serialization
                doc['embeddings'] = {
                    k: v.tolist() if hasattr(v, 'tolist') else v 
                    for k, v in doc['embeddings'].items()
                }
                f.write(json.dumps(doc) + '\n') 