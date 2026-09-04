"""
Embedding component for M2: Dense embeddings using SentenceTransformer.
"""

import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer


class DenseEmbedder:
    """Encode text chunks and queries to dense embeddings."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedder.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Encode document chunks.
        
        Args:
            chunks: List of chunk texts
            
        Returns:
            Normalized embeddings (N, embedding_dim)
        """
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        # L2 normalize for cosine similarity via inner product
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        return embeddings.astype(np.float32)
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a search query.
        
        Args:
            query: Query text
            
        Returns:
            Normalized embedding (embedding_dim,)
        """
        embedding = self.model.encode([query], convert_to_numpy=True)[0]
        # L2 normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding.astype(np.float32)
    
    @property
    def dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim
