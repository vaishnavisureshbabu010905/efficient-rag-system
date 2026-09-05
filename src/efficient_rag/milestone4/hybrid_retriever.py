"""
Hybrid retrieval for M4: Combine dense (FAISS) and binary (Hamming) results.
"""

import numpy as np
from typing import List, Dict, Any, Literal
from collections import defaultdict


class HybridRetriever:
    """Fuse results from dense and binary retrievers."""
    
    def __init__(self, dense_retriever, binary_retriever, quantizer, embedding, 
                 fusion_strategy: Literal["rrf", "weighted_avg"] = "rrf",
                 dense_weight: float = 1.0, binary_weight: float = 1.0, k: int = 60):
        """
        Initialize hybrid retriever.
        
        Args:
            dense_retriever: M2 DenseRetriever instance
            binary_retriever: M3 BinaryRetriever instance
            quantizer: M3 BinaryQuantizer instance
            embedding: M2 DenseEmbedder instance
            fusion_strategy: "rrf" (reciprocal rank fusion) or "weighted_avg"
            dense_weight: Weight for dense results (RRF uses k parameter)
            binary_weight: Weight for binary results
            k: Constant for RRF (higher = more balanced, default 60)
        """
        self.dense_retriever = dense_retriever
        self.binary_retriever = binary_retriever
        self.quantizer = quantizer
        self.embedding = embedding
        self.fusion_strategy = fusion_strategy
        self.dense_weight = dense_weight
        self.binary_weight = binary_weight
        self.k = k
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid search combining dense and binary retrieval.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of fused results with chunk_id, metadata, scores
        """
        # Encode query
        query_dense = self.embedding.encode_query(query)
        query_binary = self.quantizer.quantize(query_dense.reshape(1, -1))[0]
        
        # Run both retrievals
        dense_results = self.dense_retriever.search(query, top_k=top_k*2)
        binary_results = self.binary_retriever.search(query_binary, top_k=top_k*2)
        
        # Fuse results
        if self.fusion_strategy == "rrf":
            fused = self._fuse_rrf(dense_results, binary_results, top_k)
        else:  # weighted_avg
            fused = self._fuse_weighted_avg(dense_results, binary_results, top_k)
        
        return fused
    
    def _fuse_rrf(self, dense_results: List[Dict], binary_results: List[Dict], 
                  top_k: int) -> List[Dict]:
        """Reciprocal Rank Fusion."""
        scores = defaultdict(float)
        
        # Score dense results
        for rank, result in enumerate(dense_results, 1):
            cid = result["chunk_id"]
            scores[cid] += (1.0 / (self.k + rank)) * self.dense_weight
        
        # Score binary results
        for rank, result in enumerate(binary_results, 1):
            cid = result["chunk_id"]
            scores[cid] += (1.0 / (self.k + rank)) * self.binary_weight
        
        # Get metadata
        all_metadata = {}
        for r in dense_results + binary_results:
            if r["chunk_id"] not in all_metadata:
                all_metadata[r["chunk_id"]] = r
        
        # Sort by fusion score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                "chunk_id": cid,
                "metadata": all_metadata[cid]["metadata"],
                "fusion_score": score,
                "rank": idx + 1
            }
            for idx, (cid, score) in enumerate(ranked[:top_k])
        ]
    
    def _fuse_weighted_avg(self, dense_results: List[Dict], binary_results: List[Dict],
                           top_k: int) -> List[Dict]:
        """Normalize scores and compute weighted average."""
        all_results = {}
        
        # Process dense results (normalize by max score)
        if dense_results:
            max_dense = max(r["score"] for r in dense_results)
            for rank, result in enumerate(dense_results, 1):
                cid = result["chunk_id"]
                if cid not in all_results:
                    all_results[cid] = {"metadata": result["metadata"], "scores": {}}
                norm_score = result["score"] / max_dense if max_dense > 0 else 0
                all_results[cid]["scores"]["dense"] = norm_score
        
        # Process binary results (invert and normalize Hamming: lower distance = higher score)
        if binary_results:
            max_hamming = max(r["hamming_distance"] for r in binary_results)
            max_hamming = max(1, max_hamming)  # Avoid division by zero
            
            for rank, result in enumerate(binary_results, 1):
                cid = result["chunk_id"]
                if cid not in all_results:
                    all_results[cid] = {"metadata": result["metadata"], "scores": {}}
                # Invert: low distance = high score
                norm_score = 1.0 - (result["hamming_distance"] / max_hamming)
                all_results[cid]["scores"]["binary"] = norm_score
        
        # Compute weighted average
        fused_scores = {}
        for cid, data in all_results.items():
            dense_score = data["scores"].get("dense", 0.0)
            binary_score = data["scores"].get("binary", 0.0)
            
            total_weight = self.dense_weight + self.binary_weight
            weighted = (dense_score * self.dense_weight + 
                       binary_score * self.binary_weight) / total_weight
            fused_scores[cid] = (weighted, data["metadata"])
        
        # Sort by fusion score
        ranked = sorted(fused_scores.items(), key=lambda x: x[1][0], reverse=True)
        
        return [
            {
                "chunk_id": cid,
                "metadata": metadata,
                "fusion_score": score,
                "rank": idx + 1
            }
            for idx, (cid, (score, metadata)) in enumerate(ranked[:top_k])
        ]
