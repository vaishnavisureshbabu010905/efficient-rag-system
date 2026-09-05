"""
Binary retrieval for M3: Search using Hamming distance on quantized embeddings.
"""

import numpy as np
from typing import List, Dict, Any
from .quantizer import BinaryQuantizer, hamming_distance


class BinaryRetriever:
    """Index and search quantized binary embeddings using Hamming distance."""
    
    def __init__(self, quantizer: BinaryQuantizer):
        """
        Initialize binary retriever.
        
        Args:
            quantizer: BinaryQuantizer instance
        """
        self.quantizer = quantizer
        self.binary_index = None
        self.chunk_ids = []
        self.chunk_metadata = []
    
    def build_index(self, chunks: List[Dict[str, Any]], binary_embeddings: np.ndarray) -> int:
        """
        Build index from binary embeddings.
        
        Args:
            chunks: List of chunk dicts with 'metadata'
            binary_embeddings: Quantized binary embeddings (N, dim//8)
            
        Returns:
            Number of indexed chunks
        """
        if binary_embeddings.shape[0] != len(chunks):
            raise ValueError("Embedding count mismatch with chunk count")
        
        self.binary_index = binary_embeddings
        self.chunk_ids = [c["metadata"]["chunk_id"] for c in chunks]
        self.chunk_metadata = [c["metadata"] for c in chunks]
        
        return len(chunks)
    
    def search(self, query_binary: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search using Hamming distance.
        
        Args:
            query_binary: Quantized query embedding (dim//8,)
            top_k: Number of results
            
        Returns:
            List of result dicts with chunk_id, metadata, and hamming_distance
        """
        if self.binary_index is None:
            raise RuntimeError("Index not built")
        
        # Compute Hamming distances
        distances = []
        for idx in range(len(self.chunk_ids)):
            dist = hamming_distance(query_binary, self.binary_index[idx])
            distances.append((dist, idx))
        
        # Sort by distance (ascending = most similar first)
        distances.sort(key=lambda x: x[0])
        
        results = []
        for dist, idx in distances[:top_k]:
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "metadata": self.chunk_metadata[idx],
                "hamming_distance": int(dist)
            })
        
        return results
