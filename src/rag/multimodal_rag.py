from typing import Dict, List, Any
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

from multimodal_processor import MultimodalProcessor
from embeddings.multimodal_embedder import MultimodalEmbedder
from embeddings.cross_modal_attention import CrossModalAttention

class MultimodalRAG:
    def __init__(self, model_path: str, index_path: str):
        # Initialize processors and embedders
        self.processor = MultimodalProcessor()
        self.embedder = MultimodalEmbedder()
        self.cross_attention = CrossModalAttention()
        
        # Initialize RAG components
        self.retriever = self._init_retriever(index_path)
        self.generator = self._init_generator(model_path)
        self.qa_chain = self._create_qa_chain()
        
    def _init_retriever(self, index_path: str):
        """Initialize FAISS retriever with multimodal embeddings"""
        return FAISS.load_local(
            index_path,
            self.embedder,
            allow_dangerous_deserialization=True
        )
    
    def _init_generator(self, model_path: str):
        """Initialize language model for generation"""
        gen_pipeline = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=model_path
        )
        return HuggingFacePipeline(pipeline=gen_pipeline)
    
    def _create_qa_chain(self):
        """Create QA chain with multimodal retrieval"""
        return RetrievalQA.from_chain_type(
            llm=self.generator,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
    
    def process_document(self, pdf_path: str):
        """Process new document and add to knowledge base"""
        # Extract multimodal content
        content, embeddings = self.processor.process_document(pdf_path)
        
        # Add to vector store with multimodal embeddings
        self.retriever.add_embeddings(
            embeddings,
            metadatas=[{
                'source': pdf_path,
                'type': 'multimodal',
                'content': content
            }]
        )
        
    def query(self, question: str) -> Dict[str, Any]:
        """Process a query with multimodal context"""
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
    
    def _extract_multimodal_context(self, docs: List[Any]) -> Dict[str, Any]:
        """Extract relevant multimodal elements from retrieved documents"""
        context = {
            'text': [],
            'tables': [],
            'charts': [],
            'relationships': []
        }
        
        for doc in docs:
            metadata = doc.metadata
            if metadata['type'] == 'multimodal':
                content = metadata['content']
                # Add relevant content from each modality
                context['text'].extend(content['text_blocks'])
                context['tables'].extend(content['tables'])
                context['charts'].extend(content['charts'])
                context['relationships'].extend(content['relationships'])
        
        return context
    
    def _enhance_query_with_multimodal(self, query: str, context: Dict[str, Any]) -> str:
        """Enhance query with multimodal context using cross-attention"""
        # Convert query to features
        query_features = self.embedder.encode_text(query)
        
        # Get features for each modality
        text_features = self.embedder.encode_text(context['text'])
        table_features = self.embedder.encode_tables(context['tables'])
        chart_features = self.embedder.encode_charts(context['charts'])
        
        # Apply cross-modal attention
        enhanced_features, _ = self.cross_attention(
            query_features,
            torch.cat([text_features, table_features, chart_features], dim=0)
        )
        
        # Convert enhanced features back to query
        enhanced_query = self._features_to_query(enhanced_features, query)
        return enhanced_query
    
    def _features_to_query(self, features, original_query: str) -> str:
        """Convert enhanced features back to query text"""
        # Combine original query with enhanced context
        return f"{original_query} [Context: {self._format_context(features)}]"
    
    def _format_sources(self, docs: List[Any]) -> List[Dict[str, Any]]:
        """Format source documents with multimodal elements"""
        sources = []
        for doc in docs:
            source = {
                'text': doc.page_content,
                'metadata': doc.metadata
            }
            if 'content' in doc.metadata:
                source.update({
                    'tables': doc.metadata['content'].get('tables', []),
                    'charts': doc.metadata['content'].get('charts', []),
                    'relationships': doc.metadata['content'].get('relationships', [])
                })
            sources.append(source)
        return sources 