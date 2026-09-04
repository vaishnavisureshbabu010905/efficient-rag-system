"""
Document ingestion, extraction, and chunking for Milestone 1.
Supports PDF and TXT files with character-based chunking.
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


class DocumentProcessor:
    """
    Process documents: ingest → extract → chunk → metadata.
    M1 supports character-based chunking only.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize DocumentProcessor.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks in characters
            
        Raises:
            ValueError: If chunk_overlap >= chunk_size
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def ingest_documents(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Ingest all PDF and TXT files from a folder.
        
        Args:
            folder_path: Path to folder containing documents
            
        Returns:
            List of documents with content and metadata
        """
        documents = []
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        for file_path in sorted(folder.glob("*")):
            if file_path.is_file():
                if file_path.suffix.lower() == ".pdf":
                    documents.extend(self._process_pdf(file_path))
                elif file_path.suffix.lower() == ".txt":
                    documents.extend(self._process_txt(file_path))
        
        return documents
    
    def _process_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extract text from a PDF file.
        Pages are joined with double newlines to preserve boundaries.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List containing one document dict (or empty if error)
        """
        if not HAS_PDFPLUMBER:
            raise ImportError(
                "pdfplumber not installed. Install with: pip install pdfplumber"
            )
        
        try:
            with pdfplumber.open(file_path) as pdf:
                pages = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
                
                # Join pages with double newlines to preserve page boundaries
                text = "\n\n".join(pages)
                
                return [{
                    "content": text,
                    "metadata": self._extract_metadata(file_path, len(pdf.pages))
                }]
        
        except Exception as e:
            print(f"⚠️  Error processing {file_path.name}: {e}")
            return []
    
    def _process_txt(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extract text from a TXT file.
        
        Args:
            file_path: Path to TXT file
            
        Returns:
            List containing one document dict (or empty if error)
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                return [{
                    "content": text,
                    "metadata": self._extract_metadata(file_path)
                }]
        
        except Exception as e:
            print(f"⚠️  Error processing {file_path.name}: {e}")
            return []
    
    def _extract_metadata(self, file_path: Path, pages: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract metadata from file.
        
        Args:
            file_path: Path to file
            pages: Number of pages (for PDFs)
            
        Returns:
            Metadata dictionary
        """
        stat = file_path.stat()
        metadata = {
            "filename": file_path.name,
            "file_type": file_path.suffix,
            "file_size_bytes": stat.st_size,
            "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }
        
        if pages is not None:
            metadata["pages"] = pages
        
        return metadata
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk documents using character-based chunking with overlap.
        Generates stable chunk IDs: {filename}_{chunk_number}
        
        Args:
            documents: List of documents from ingest_documents()
            
        Returns:
            List of chunks with text and metadata
        """
        chunked = []
        
        for doc in documents:
            text = doc["content"]
            metadata = doc["metadata"]
            filename = metadata["filename"]
            chunk_num = 0
            
            # Character-based chunking with overlap
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk_text = text[i:i + self.chunk_size]
                
                # Skip empty chunks
                if len(chunk_text.strip()) == 0:
                    continue
                
                chunked.append({
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_id": f"{filename}_{chunk_num}",
                        "char_start": i,
                        "char_end": min(i + self.chunk_size, len(text)),
                    }
                })
                chunk_num += 1
        
        return chunked
    
    def process_pipeline(
        self,
        folder_path: str,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Complete M1 pipeline: ingest → extract → chunk → metadata.
        
        Args:
            folder_path: Path to folder with documents
            verbose: Print progress messages
            
        Returns:
            List of chunks with text and metadata
        """
        if verbose:
            print(f"\n📂 Ingesting documents from '{folder_path}'...")
        
        documents = self.ingest_documents(folder_path)
        
        if verbose:
            print(f"✅ Extracted {len(documents)} document(s)")
            print(f"🔗 Chunking (size={self.chunk_size}, overlap={self.chunk_overlap})...")
        
        chunks = self.chunk_documents(documents)
        
        if verbose:
            print(f"✅ Created {len(chunks)} chunk(s)\n")
        
        return chunks
