"""Tests for M2: Dense Embedding + FAISS Retrieval."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.efficient_rag.milestone2.embedding import DenseEmbedder
from src.efficient_rag.milestone2.retriever import DenseRetriever


@pytest.fixture
def mock_embedder():
    """Create a mock embedder for offline testing."""
    embedder = DenseEmbedder.__new__(DenseEmbedder)
    embedder.model = Mock()
    embedder.embedding_dim = 384
    
    def mock_encode(texts, convert_to_numpy=True):
        """Mock encode that returns normalized random embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = np.random.randn(len(texts), 384).astype(np.float32)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        return embeddings
    
    embedder.model.encode = mock_encode
    embedder.model.get_sentence_embedding_dimension = Mock(return_value=384)
    return embedder


class TestDenseEmbedder:
    """Test embedding component."""
    
    @patch('src.efficient_rag.milestone2.embedding.SentenceTransformer')
    def test_init(self, mock_st):
        """Test embedder initialization."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model
        
        embedder = DenseEmbedder()
        assert embedder.model is not None
        assert embedder.dim == 384
    
    def test_embedding_dimension(self, mock_embedder):
        """Test embedding dimension is correct."""
        assert mock_embedder.dim == 384
    
    def test_encode_single_chunk(self, mock_embedder):
        """Test encoding a single chunk."""
        text = "This is a test document."
        embedding = mock_embedder.encode_query(text)
        
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32
        # Check normalization
        norm = np.linalg.norm(embedding)
        assert 0.99 < norm < 1.01
    
    def test_encode_multiple_chunks(self, mock_embedder):
        """Test encoding multiple chunks."""
        texts = [
            "First chunk about machine learning.",
            "Second chunk about neural networks.",
            "Third chunk about deep learning."
        ]
        embeddings = mock_embedder.encode_chunks(texts)
        
        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32
        # Check normalization for all
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            assert 0.99 < norm < 1.01
    
    def test_embeddings_are_different(self, mock_embedder):
        """Test that different texts produce different embeddings."""
        emb1 = mock_embedder.encode_query("Machine learning")
        emb2 = mock_embedder.encode_query("Deep learning")
        
        # Should not be identical (with high probability for random embeddings)
        assert not np.allclose(emb1, emb2)


class TestDenseRetriever:
    """Test retrieval component."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            {
                "text": "Machine learning is a subset of artificial intelligence.",
                "metadata": {"chunk_id": "doc1_0", "filename": "doc1.txt"}
            },
            {
                "text": "Neural networks are inspired by biological systems.",
                "metadata": {"chunk_id": "doc1_1", "filename": "doc1.txt"}
            },
            {
                "text": "Deep learning uses multiple layers of neurons.",
                "metadata": {"chunk_id": "doc1_2", "filename": "doc1.txt"}
            },
        ]
    
    def test_build_index(self, sample_chunks, mock_embedder):
        """Test building FAISS index."""
        retriever = DenseRetriever(mock_embedder)
        
        num_indexed = retriever.build_index(sample_chunks)
        
        assert num_indexed == 3
        assert retriever.index is not None
        assert len(retriever.chunk_ids) == 3
        assert len(retriever.chunk_texts) == 3
        assert len(retriever.chunk_metadata) == 3
    
    def test_search_single_result(self, sample_chunks, mock_embedder):
        """Test searching for single result."""
        retriever = DenseRetriever(mock_embedder)
        retriever.build_index(sample_chunks)
        
        results = retriever.search("machine learning", top_k=1)
        
        assert len(results) == 1
        assert "text" in results[0]
        assert "metadata" in results[0]
        assert "score" in results[0]
        assert isinstance(results[0]["score"], (int, float))
    
    def test_search_multiple_results(self, sample_chunks, mock_embedder):
        """Test searching for multiple results."""
        retriever = DenseRetriever(mock_embedder)
        retriever.build_index(sample_chunks)
        
        results = retriever.search("neural networks", top_k=3)
        
        assert len(results) == 3
        # Results should be ordered by score (descending)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_search_preserves_metadata(self, sample_chunks, mock_embedder):
        """Test that metadata is preserved in results."""
        retriever = DenseRetriever(mock_embedder)
        retriever.build_index(sample_chunks)
        
        results = retriever.search("learning", top_k=1)
        
        assert results[0]["metadata"]["chunk_id"] in ["doc1_0", "doc1_1", "doc1_2"]
        assert results[0]["metadata"]["filename"] == "doc1.txt"
    
    def test_search_without_index_raises_error(self, mock_embedder):
        """Test that search without index raises error."""
        retriever = DenseRetriever(mock_embedder)
        
        with pytest.raises(RuntimeError, match="Index not built"):
            retriever.search("test query")
    
    def test_save_and_load_index(self, sample_chunks, mock_embedder, tmp_path):
        """Test saving and loading index."""
        retriever1 = DenseRetriever(mock_embedder)
        retriever1.build_index(sample_chunks)
        
        # Save
        index_dir = tmp_path / "index"
        retriever1.save_index(str(index_dir))
        
        assert (index_dir / "index.faiss").exists()
        assert (index_dir / "metadata.json").exists()
        
        # Load
        retriever2 = DenseRetriever(mock_embedder)
        num_loaded = retriever2.load_index(str(index_dir))
        
        assert num_loaded == 3
        assert retriever2.chunk_ids == retriever1.chunk_ids
        assert retriever2.chunk_texts == retriever1.chunk_texts
    
    def test_search_after_load(self, sample_chunks, mock_embedder, tmp_path):
        """Test that search works after loading."""
        retriever1 = DenseRetriever(mock_embedder)
        retriever1.build_index(sample_chunks)
        
        # Save
        index_dir = tmp_path / "index"
        retriever1.save_index(str(index_dir))
        
        # Load and search
        retriever2 = DenseRetriever(mock_embedder)
        retriever2.load_index(str(index_dir))
        results = retriever2.search("learning", top_k=1)
        
        assert len(results) == 1
        assert "score" in results[0]
        # Score can be positive or negative with random embeddings, just check it's a number
        assert isinstance(results[0]["score"], (int, float))
    
    def test_empty_chunks_handled(self, mock_embedder):
        """Test handling of empty chunk list."""
        retriever = DenseRetriever(mock_embedder)
        
        num_indexed = retriever.build_index([])
        assert num_indexed == 0
    
    def test_search_top_k_limit(self, sample_chunks, mock_embedder):
        """Test top_k parameter limits results."""
        retriever = DenseRetriever(mock_embedder)
        retriever.build_index(sample_chunks)
        
        results_1 = retriever.search("learning", top_k=1)
        results_2 = retriever.search("learning", top_k=2)
        results_3 = retriever.search("learning", top_k=10)  # More than available
        
        assert len(results_1) == 1
        assert len(results_2) == 2
        assert len(results_3) == 3  # Only 3 chunks available
