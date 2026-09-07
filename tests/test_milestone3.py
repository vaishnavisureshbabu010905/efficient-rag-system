"""Tests for M3: Binary Quantization."""

import pytest
import numpy as np
from efficient_rag.milestone3.quantizer import BinaryQuantizer, hamming_distance
from efficient_rag.milestone3.binary_retriever import BinaryRetriever


class TestBinaryQuantizer:
    """Test quantization component."""
    
    def test_init_valid_dim(self):
        """Test initialization with valid dimension."""
        q = BinaryQuantizer(dim=384)
        assert q.dim == 384
    
    def test_init_invalid_dim(self):
        """Test initialization rejects non-8-multiple dimensions."""
        with pytest.raises(ValueError, match="divisible by 8"):
            BinaryQuantizer(dim=385)
    
    def test_quantize_shape(self):
        """Test quantization output shape."""
        q = BinaryQuantizer(dim=384)
        embeddings = np.random.randn(5, 384).astype(np.float32)
        binary = q.quantize(embeddings)
        
        assert binary.shape == (5, 48)  # 384 / 8 = 48
        assert binary.dtype == np.uint8
    
    def test_quantize_deterministic(self):
        """Test quantization is deterministic."""
        q = BinaryQuantizer(dim=384)
        embeddings = np.array([[1.0, -1.0, 0.5, -0.5] + [0.0]*380], dtype=np.float32)
        
        binary1 = q.quantize(embeddings)
        binary2 = q.quantize(embeddings)
        
        assert np.array_equal(binary1, binary2)
    
    def test_quantize_threshold(self):
        """Test quantization uses 0 threshold."""
        q = BinaryQuantizer(dim=16)
        # Create simple test case: first 8 positive, next 8 negative
        embeddings = np.array([
            [1.0]*8 + [-1.0]*8
        ], dtype=np.float32)
        
        binary = q.quantize(embeddings)
        
        # First byte: 11111111 = 255
        # Second byte: 00000000 = 0
        assert binary[0, 0] == 255
        assert binary[0, 1] == 0
    
    def test_dequantize_shape(self):
        """Test dequantization output shape."""
        q = BinaryQuantizer(dim=384)
        binary = np.random.randint(0, 256, (5, 48), dtype=np.uint8)
        
        reconstructed = q.dequantize(binary)
        
        assert reconstructed.shape == (5, 384)
        assert reconstructed.dtype == np.float32
    
    def test_quantize_dequantize_roundtrip(self):
        """Test quantize->dequantize roundtrip."""
        q = BinaryQuantizer(dim=16)
        original = np.array([[1.0, -1.0, 0.5, -0.5]*4], dtype=np.float32)
        
        binary = q.quantize(original)
        reconstructed = q.dequantize(binary)
        
        # Values should be 0 or 1 (quantized)
        expected = (original > 0).astype(np.float32)
        assert np.allclose(reconstructed, expected)
    
    def test_quantize_dimension_validation(self):
        """Test quantization validates input dimension."""
        q = BinaryQuantizer(dim=384)
        wrong_dim = np.random.randn(5, 256).astype(np.float32)
        
        with pytest.raises(ValueError, match="Expected dim"):
            q.quantize(wrong_dim)
    
    def test_dequantize_dimension_validation(self):
        """Test dequantization validates input size."""
        q = BinaryQuantizer(dim=384)
        wrong_size = np.random.randint(0, 256, (5, 32), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            q.dequantize(wrong_size)


class TestHammingDistance:
    """Test Hamming distance computation."""
    
    def test_hamming_identical(self):
        """Test Hamming distance between identical vectors."""
        v1 = np.array([255, 255], dtype=np.uint8)
        v2 = np.array([255, 255], dtype=np.uint8)
        
        dist = hamming_distance(v1, v2)
        assert dist == 0
    
    def test_hamming_opposite(self):
        """Test Hamming distance between opposite vectors."""
        v1 = np.array([255], dtype=np.uint8)  # 11111111
        v2 = np.array([0], dtype=np.uint8)    # 00000000
        
        dist = hamming_distance(v1, v2)
        assert dist == 8
    
    def test_hamming_single_bit_diff(self):
        """Test Hamming distance with single bit difference."""
        v1 = np.array([255], dtype=np.uint8)  # 11111111
        v2 = np.array([254], dtype=np.uint8)  # 11111110
        
        dist = hamming_distance(v1, v2)
        assert dist == 1
    
    def test_hamming_symmetric(self):
        """Test Hamming distance is symmetric."""
        v1 = np.array([42, 127], dtype=np.uint8)
        v2 = np.array([84, 200], dtype=np.uint8)
        
        dist12 = hamming_distance(v1, v2)
        dist21 = hamming_distance(v2, v1)
        
        assert dist12 == dist21


class TestBinaryRetriever:
    """Test binary retrieval component."""
    
    @pytest.fixture
    def quantizer(self):
        return BinaryQuantizer(dim=384)
    
    @pytest.fixture
    def sample_chunks(self):
        return [
            {"metadata": {"chunk_id": "c0", "filename": "f0.txt"}},
            {"metadata": {"chunk_id": "c1", "filename": "f1.txt"}},
            {"metadata": {"chunk_id": "c2", "filename": "f2.txt"}},
        ]
    
    def test_build_index(self, quantizer, sample_chunks):
        """Test building binary index."""
        retriever = BinaryRetriever(quantizer)
        binary = np.random.randint(0, 256, (3, 48), dtype=np.uint8)
        
        num = retriever.build_index(sample_chunks, binary)
        
        assert num == 3
        assert len(retriever.chunk_ids) == 3
        assert len(retriever.chunk_metadata) == 3
    
    def test_search_returns_top_k(self, quantizer, sample_chunks):
        """Test search returns top-k results."""
        retriever = BinaryRetriever(quantizer)
        binary = np.random.randint(0, 256, (3, 48), dtype=np.uint8)
        retriever.build_index(sample_chunks, binary)
        
        query_binary = binary[0]  # Use first chunk as query
        results = retriever.search(query_binary, top_k=2)
        
        assert len(results) == 2
        assert results[0]["hamming_distance"] == 0  # Exact match
    
    def test_search_results_have_metadata(self, quantizer, sample_chunks):
        """Test results include metadata."""
        retriever = BinaryRetriever(quantizer)
        binary = np.random.randint(0, 256, (3, 48), dtype=np.uint8)
        retriever.build_index(sample_chunks, binary)
        
        results = retriever.search(binary[0], top_k=1)
        
        assert "chunk_id" in results[0]
        assert "metadata" in results[0]
        assert "hamming_distance" in results[0]
    
    def test_search_without_index_raises(self, quantizer):
        """Test search raises error without built index."""
        retriever = BinaryRetriever(quantizer)
        query = np.zeros(48, dtype=np.uint8)
        
        with pytest.raises(RuntimeError, match="Index not built"):
            retriever.search(query)
    
    def test_search_mismatch_chunks_embeddings(self, quantizer, sample_chunks):
        """Test dimension mismatch handling."""
        retriever = BinaryRetriever(quantizer)
        binary = np.random.randint(0, 256, (2, 48), dtype=np.uint8)  # Only 2, chunks has 3
        
        with pytest.raises(ValueError, match="mismatch"):
            retriever.build_index(sample_chunks, binary)
    
    def test_search_top_k_limit(self, quantizer, sample_chunks):
        """Test top_k limits results correctly."""
        retriever = BinaryRetriever(quantizer)
        binary = np.random.randint(0, 256, (3, 48), dtype=np.uint8)
        retriever.build_index(sample_chunks, binary)
        
        results_1 = retriever.search(binary[0], top_k=1)
        results_2 = retriever.search(binary[0], top_k=5)  # More than available
        
        assert len(results_1) == 1
        assert len(results_2) == 3
    
    def test_search_results_ordered_by_distance(self, quantizer, sample_chunks):
        """Test results are ordered by Hamming distance (ascending)."""
        retriever = BinaryRetriever(quantizer)
        binary = np.random.randint(0, 256, (3, 48), dtype=np.uint8)
        retriever.build_index(sample_chunks, binary)
        
        results = retriever.search(binary[0], top_k=3)
        
        distances = [r["hamming_distance"] for r in results]
        assert distances == sorted(distances)
