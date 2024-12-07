class MultimodalEmbedder:
    def __init__(self):
        # Initialize different embedding models
        self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.visual_embedder = ResNetFeatureExtractor()
        self.table_embedder = TableBERT()
        
    def create_embeddings(self, content):
        """Create embeddings that capture cross-modal relationships"""
        embeddings = {
            # Text embeddings
            'text': self.text_embedder.encode(content['text']),
            
            # Table embeddings with text context
            'tables': [
                self._embed_table_with_context(table, content['text'])
                for table in content['tables']
            ],
            
            # Chart embeddings with surrounding text
            'charts': [
                self._embed_chart_with_context(chart, content['text'])
                for chart in content['charts']
            ],
            
            # Cross-modal attention maps
            'cross_modal_attention': self._compute_cross_attention(content)
        }
        return embeddings 