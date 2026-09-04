"""
Dense FAISS retriever for M2: Index and search embeddings.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Tuple, Any
from .embedding import DenseEmbedder


class DenseRetriever:
    """Build and search FAISS index of chunk embeddings."""
    
    def __init__(self, embedder: DenseEmbedder):
        """
        Initialize retriever.
        
        Args:
            embedder: DenseEmbedder instance
        """
        self.embedder = embedder
        self.index = None
        self.chunk_ids = []
        self.chunk_texts = []
        self.chunk_metadata = []
    
    def build_index(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Build FAISS index from chunks.
        
        Args:
            chunks: List of chunk dicts with 'text' and 'metadata'
            
        Returns:
            Number of indexed chunks
        """
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode_chunks(texts)
        
        # Create flat index for exact cosine search
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        
        self.chunk_ids = [c["metadata"]["chunk_id"] for c in chunks]
        self.chunk_texts = texts
        self.chunk_metadata = [c["metadata"] for c in chunks]
        
        return len(self.chunk_ids)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for top-k similar chunks.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of result dicts with text, metadata, and score
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        query_embedding = self.embedder.encode_query(query)
        scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # No result
                break
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "text": self.chunk_texts[idx],
                "metadata": self.chunk_metadata[idx],
                "score": float(score)
            })
        return results
    
    def save_index(self, output_dir: str) -> None:
        """
        Save index and metadata.
        
        Args:
            output_dir: Directory to save to
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(output_path / "index.faiss"))
        
        # Save metadata
        metadata = {
            "chunk_ids": self.chunk_ids,
            "chunk_texts": self.chunk_texts,
            "chunk_metadata": self.chunk_metadata
        }
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f)
    
    def load_index(self, input_dir: str) -> int:
        """
        Load index and metadata.
        
        Args:
            input_dir: Directory to load from
            
        Returns:
            Number of loaded chunks
        """
        input_path = Path(input_dir)
        
        # Load FAISS index
        self.index = faiss.read_index(str(input_path / "index.faiss"))
        
        # Load metadata
        with open(input_path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        self.chunk_ids = metadata["chunk_ids"]
        self.chunk_texts = metadata["chunk_texts"]
        self.chunk_metadata = metadata["chunk_metadata"]
        
        return len(self.chunk_ids)
