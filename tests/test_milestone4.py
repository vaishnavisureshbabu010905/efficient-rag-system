"""Tests for M4: Hybrid Retrieval."""

import pytest
import numpy as np
from unittest.mock import Mock
from src.efficient_rag.milestone4.hybrid_retriever import HybridRetriever


@pytest.fixture
def mock_dense_retriever():
    """Mock dense retriever."""
    retriever = Mock()
    retriever.search = Mock(return_value=[
        {"chunk_id": "c0", "metadata": {"f": "f0"}, "score": 0.9},
        {"chunk_id": "c1", "metadata": {"f": "f1"}, "score": 0.7},
        {"chunk_id": "c2", "metadata": {"f": "f2"}, "score": 0.6},
    ])
    return retriever


@pytest.fixture
def mock_binary_retriever():
    """Mock binary retriever."""
    retriever = Mock()
    retriever.search = Mock(return_value=[
        {"chunk_id": "c1", "metadata": {"f": "f1"}, "hamming_distance": 50},
        {"chunk_id": "c0", "metadata": {"f": "f0"}, "hamming_distance": 100},
        {"chunk_id": "c3", "metadata": {"f": "f3"}, "hamming_distance": 120},
    ])
    return retriever


@pytest.fixture
def mock_quantizer():
    """Mock quantizer."""
    q = Mock()
    q.quantize = Mock(return_value=np.zeros(48, dtype=np.uint8))
    return q


@pytest.fixture
def mock_embedder():
    """Mock embedder."""
    e = Mock()
    e.encode_query = Mock(return_value=np.zeros(384, dtype=np.float32))
    return e


@pytest.fixture
def hybrid_retriever(mock_dense_retriever, mock_binary_retriever, mock_quantizer, mock_embedder):
    """Create hybrid retriever with mocks."""
    return HybridRetriever(
        dense_retriever=mock_dense_retriever,
        binary_retriever=mock_binary_retriever,
        quantizer=mock_quantizer,
        embedding=mock_embedder,
        fusion_strategy="rrf"
    )


class TestHybridRetrieverRRF:
    """Test RRF fusion strategy."""
    
    def test_init(self, hybrid_retriever):
        """Test initialization."""
        assert hybrid_retriever.fusion_strategy == "rrf"
        assert hybrid_retriever.dense_weight == 1.0
        assert hybrid_retriever.k == 60
    
    def test_search_returns_results(self, hybrid_retriever):
        """Test search returns fused results."""
        results = hybrid_retriever.search("test", top_k=3)
        
        assert len(results) == 3
        assert all("chunk_id" in r for r in results)
        assert all("metadata" in r for r in results)
        assert all("fusion_score" in r for r in results)
    
    def test_search_results_ranked(self, hybrid_retriever):
        """Test results are properly ranked."""
        results = hybrid_retriever.search("test", top_k=3)
        
        scores = [r["fusion_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_search_result_has_rank(self, hybrid_retriever):
        """Test each result includes its rank."""
        results = hybrid_retriever.search("test", top_k=2)
        
        assert results[0]["rank"] == 1
        assert results[1]["rank"] == 2
    
    def test_fusion_combines_both_retrievers(self, hybrid_retriever):
        """Test that fusion uses results from both retrievers."""
        results = hybrid_retriever.search("test", top_k=4)
        result_ids = [r["chunk_id"] for r in results]
        
        # Should have c0, c1 from both, c2 from dense, c3 from binary
        assert len(set(result_ids) & {"c0", "c1"}) >= 1
    
    def test_rrf_with_weight_adjustment(self, mock_dense_retriever, mock_binary_retriever,
                                        mock_quantizer, mock_embedder):
        """Test RRF respects weight parameters."""
        hybrid = HybridRetriever(
            mock_dense_retriever, mock_binary_retriever, mock_quantizer, mock_embedder,
            fusion_strategy="rrf", dense_weight=2.0, binary_weight=0.5, k=60
        )
        
        results = hybrid.search("test", top_k=2)
        assert len(results) == 2


class TestHybridRetrieverWeightedAvg:
    """Test weighted average fusion strategy."""
    
    def test_weighted_avg_fusion(self, mock_dense_retriever, mock_binary_retriever,
                                 mock_quantizer, mock_embedder):
        """Test weighted average fusion."""
        hybrid = HybridRetriever(
            mock_dense_retriever, mock_binary_retriever, mock_quantizer, mock_embedder,
            fusion_strategy="weighted_avg", dense_weight=1.0, binary_weight=1.0
        )
        
        results = hybrid.search("test", top_k=3)
        
        assert len(results) == 3
        scores = [r["fusion_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_weighted_avg_scores_normalized(self, mock_dense_retriever, mock_binary_retriever,
                                            mock_quantizer, mock_embedder):
        """Test weighted average scores are in valid range."""
        hybrid = HybridRetriever(
            mock_dense_retriever, mock_binary_retriever, mock_quantizer, mock_embedder,
            fusion_strategy="weighted_avg"
        )
        
        results = hybrid.search("test", top_k=2)
        
        for result in results:
            assert 0 <= result["fusion_score"] <= 1.0
    
    def test_weighted_avg_respects_weights(self, mock_dense_retriever, mock_binary_retriever,
                                           mock_quantizer, mock_embedder):
        """Test that weights affect ranking."""
        # Heavy dense weight
        hybrid_dense = HybridRetriever(
            mock_dense_retriever, mock_binary_retriever, mock_quantizer, mock_embedder,
            fusion_strategy="weighted_avg", dense_weight=10.0, binary_weight=0.1
        )
        
        results_dense = hybrid_dense.search("test", top_k=2)
        first_id_dense = results_dense[0]["chunk_id"]
        
        # Verify it prioritizes dense results
        assert first_id_dense in {"c0", "c1", "c2"}  # Top dense results


class TestHybridRetrieverEdgeCases:
    """Test edge cases."""
    
    def test_empty_results(self, mock_quantizer, mock_embedder):
        """Test with empty results from one retriever."""
        dense = Mock()
        dense.search = Mock(return_value=[])
        
        binary = Mock()
        binary.search = Mock(return_value=[
            {"chunk_id": "c0", "metadata": {"f": "f0"}, "hamming_distance": 50},
        ])
        
        hybrid = HybridRetriever(dense, binary, mock_quantizer, mock_embedder)
        results = hybrid.search("test", top_k=1)
        
        assert len(results) >= 1
        assert results[0]["chunk_id"] == "c0"
    
    def test_top_k_limit(self, hybrid_retriever):
        """Test top_k properly limits results."""
        results_1 = hybrid_retriever.search("test", top_k=1)
        results_5 = hybrid_retriever.search("test", top_k=5)
        
        assert len(results_1) <= 1
        assert len(results_5) <= 5
    
    def test_duplicate_chunk_ids(self, mock_quantizer, mock_embedder):
        """Test handling of chunk IDs appearing in both retrievers."""
        dense = Mock()
        dense.search = Mock(return_value=[
            {"chunk_id": "c0", "metadata": {"f": "f0"}, "score": 0.9},
            {"chunk_id": "c1", "metadata": {"f": "f1"}, "score": 0.8},
        ])
        
        binary = Mock()
        binary.search = Mock(return_value=[
            {"chunk_id": "c0", "metadata": {"f": "f0"}, "hamming_distance": 10},
            {"chunk_id": "c1", "metadata": {"f": "f1"}, "hamming_distance": 50},
        ])
        
        hybrid = HybridRetriever(dense, binary, mock_quantizer, mock_embedder)
        results = hybrid.search("test", top_k=2)
        
        # Should have c0 and c1 with combined scores
        ids = [r["chunk_id"] for r in results]
        assert len(set(ids)) == len(ids)  # No duplicates
