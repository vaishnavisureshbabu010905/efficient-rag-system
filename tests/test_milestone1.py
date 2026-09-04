"""Unit tests for DocumentProcessor (M1: Ingestion & Chunking)"""

import pytest
from pathlib import Path
from src.efficient_rag.milestone1 import DocumentProcessor


class TestDocumentProcessor:
    """Test suite for DocumentProcessor"""
    
    def test_init_valid(self):
        """Test valid initialization"""
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 50
    
    def test_init_invalid_overlap(self):
        """Test that overlap >= size raises ValueError"""
        with pytest.raises(ValueError, match="chunk_overlap"):
            DocumentProcessor(chunk_size=500, chunk_overlap=500)
        
        with pytest.raises(ValueError, match="chunk_overlap"):
            DocumentProcessor(chunk_size=500, chunk_overlap=600)
    
    def test_ingest_documents(self, sample_documents_dir, document_processor):
        """Test ingestion of PDF and TXT files"""
        documents = document_processor.ingest_documents(str(sample_documents_dir))
        
        assert len(documents) == 2, "Should ingest PDF and TXT"
        
        filenames = [doc["metadata"]["filename"] for doc in documents]
        assert "sample.txt" in filenames
        assert "sample.pdf" in filenames
    
    def test_ingest_nonexistent_folder(self, document_processor):
        """Test error handling for nonexistent folder"""
        with pytest.raises(FileNotFoundError):
            document_processor.ingest_documents("/nonexistent/folder")
    
    def test_process_txt(self, sample_documents_dir, document_processor):
        """Test TXT file extraction"""
        documents = document_processor.ingest_documents(str(sample_documents_dir))
        txt_doc = [d for d in documents if d["metadata"]["filename"] == "sample.txt"][0]
        
        assert len(txt_doc["content"]) > 0
        assert "sample text document" in txt_doc["content"]
    
    def test_process_pdf(self, sample_documents_dir, document_processor):
        """Test PDF file extraction"""
        documents = document_processor.ingest_documents(str(sample_documents_dir))
        pdf_doc = [d for d in documents if d["metadata"]["filename"] == "sample.pdf"][0]
        
        assert len(pdf_doc["content"]) > 0
        assert pdf_doc["metadata"]["pages"] == 2
        # Pages should be joined with \n\n
        assert "\n\n" in pdf_doc["content"]
    
    def test_metadata_extraction(self, sample_documents_dir, document_processor):
        """Test metadata extraction"""
        documents = document_processor.ingest_documents(str(sample_documents_dir))
        
        for doc in documents:
            metadata = doc["metadata"]
            assert "filename" in metadata
            assert "file_type" in metadata
            assert "file_size_bytes" in metadata
            assert "modified_date" in metadata
            
            if metadata["file_type"] == ".pdf":
                assert "pages" in metadata
    
    def test_chunk_documents(self, sample_documents_dir, document_processor):
        """Test chunking of documents"""
        documents = document_processor.ingest_documents(str(sample_documents_dir))
        chunks = document_processor.chunk_documents(documents)
        
        assert len(chunks) > 0, "Should create at least one chunk"
        
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert len(chunk["text"]) <= document_processor.chunk_size
    
    def test_stable_chunk_ids(self, sample_documents_dir, document_processor):
        """Test that chunk IDs are stable and document-specific"""
        documents = document_processor.ingest_documents(str(sample_documents_dir))
        chunks = document_processor.chunk_documents(documents)
        
        chunk_ids = [c["metadata"]["chunk_id"] for c in chunks]
        
        # IDs should be unique
        assert len(chunk_ids) == len(set(chunk_ids))
        
        # IDs should contain filename
        txt_chunks = [c for c in chunks if "sample.txt" in c["metadata"]["chunk_id"]]
        pdf_chunks = [c for c in chunks if "sample.pdf" in c["metadata"]["chunk_id"]]
        
        assert len(txt_chunks) > 0
        assert len(pdf_chunks) > 0
        
        # IDs should follow pattern: filename_number
        for chunk_id in chunk_ids:
            assert "_" in chunk_id
            parts = chunk_id.rsplit("_", 1)
            assert len(parts) == 2
            assert parts[1].isdigit()
    
    def test_chunk_positions(self, sample_documents_dir, document_processor):
        """Test that char_start and char_end are correct"""
        documents = document_processor.ingest_documents(str(sample_documents_dir))
        chunks = document_processor.chunk_documents(documents)
        
        for chunk in chunks:
            metadata = chunk["metadata"]
            assert metadata["char_end"] > metadata["char_start"]
            assert metadata["char_end"] - metadata["char_start"] <= document_processor.chunk_size
    
    def test_pipeline(self, sample_documents_dir, document_processor):
        """Test complete M1 pipeline"""
        chunks = document_processor.process_pipeline(str(sample_documents_dir), verbose=False)
        
        assert len(chunks) > 0
        
        # Verify all chunks have required structure
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"].get("chunk_id")
            assert chunk["metadata"].get("filename")
    
    def test_chunk_overlap(self, sample_documents_dir):
        """Test that chunks have proper overlap"""
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        documents = processor.ingest_documents(str(sample_documents_dir))
        chunks = processor.chunk_documents(documents)
        
        if len(chunks) >= 2:
            # Get all text
            full_texts = {doc["metadata"]["filename"]: doc["content"] for doc in documents}
            
            # Check overlap by looking at character positions
            for i in range(len(chunks) - 1):
                curr = chunks[i]["metadata"]
                next_chunk = chunks[i + 1]["metadata"]
                
                # If same file, check overlap
                if curr.get("filename") == next_chunk.get("filename"):
                    gap = next_chunk["char_start"] - curr["char_end"]
                    expected_overlap = processor.chunk_size - processor.chunk_overlap
                    # Gap should be negative (overlap) or small
                    assert gap <= expected_overlap


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_folder(self, tmp_path):
        """Test processing empty folder"""
        processor = DocumentProcessor()
        documents = processor.ingest_documents(str(tmp_path))
        assert len(documents) == 0
    
    def test_small_chunk_size(self, sample_documents_dir):
        """Test with very small chunk size"""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        chunks = processor.process_pipeline(str(sample_documents_dir), verbose=False)
        
        # Should create many chunks
        assert len(chunks) > 5
    
    def test_large_overlap(self, sample_documents_dir):
        """Test with large overlap"""
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=400)
        chunks = processor.process_pipeline(str(sample_documents_dir), verbose=False)
        
        assert len(chunks) > 0
    
    def test_non_text_files_ignored(self, tmp_path):
        """Test that non-text files are ignored"""
        # Create test files
        (tmp_path / "test.txt").write_text("Test content")
        (tmp_path / "image.jpg").write_bytes(b"fake image")
        (tmp_path / "archive.zip").write_bytes(b"fake archive")
        
        processor = DocumentProcessor()
        documents = processor.ingest_documents(str(tmp_path))
        
        # Should only process TXT
        assert len(documents) == 1
        assert documents[0]["metadata"]["filename"] == "test.txt"
