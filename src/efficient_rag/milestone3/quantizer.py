"""
Binary quantization for M3: Convert dense embeddings to 1-bit representations.
"""

import numpy as np
from typing import Tuple


class BinaryQuantizer:
    """Quantize float embeddings to binary (1-bit per dimension)."""
    
    def __init__(self, dim: int = 384):
        """
        Initialize quantizer.
        
        Args:
            dim: Embedding dimension (must be divisible by 8 for byte packing)
        """
        if dim % 8 != 0:
            raise ValueError(f"Dimension {dim} must be divisible by 8 for efficient packing")
        self.dim = dim
        self.dtype = np.uint8
    
    def quantize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Quantize embeddings to binary using threshold at 0.
        
        Args:
            embeddings: Float array (N, dim) with values normalized to [-1, 1]
            
        Returns:
            Packed binary array (N, dim//8) as uint8
        """
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {embeddings.shape[1]}")
        
        # Threshold at 0: positive -> 1, negative -> 0
        binary = (embeddings > 0).astype(np.uint8)
        
        # Pack 8 bits into each byte
        n_samples = binary.shape[0]
        n_bytes = self.dim // 8
        packed = np.zeros((n_samples, n_bytes), dtype=self.dtype)
        
        for i in range(n_bytes):
            # Take 8 bits and pack into one byte (bit 0 = leftmost bit)
            byte_idx = i
            bits = binary[:, i*8:(i+1)*8]
            packed[:, byte_idx] = (bits * (1 << np.arange(7, -1, -1))).sum(axis=1)
        
        return packed
    
    def dequantize(self, packed: np.ndarray) -> np.ndarray:
        """
        Dequantize binary back to float (0.0 or 1.0).
        
        Args:
            packed: Packed binary array (N, dim//8) as uint8
            
        Returns:
            Float array (N, dim) with values in {0, 1}
        """
        if packed.shape[1] != self.dim // 8:
            raise ValueError(f"Expected {self.dim // 8} bytes, got {packed.shape[1]}")
        
        n_samples = packed.shape[0]
        binary = np.zeros((n_samples, self.dim), dtype=np.float32)
        
        for i in range(self.dim // 8):
            # Unpack byte i into 8 bits
            byte_val = packed[:, i:i+1]  # (N, 1)
            for bit_idx in range(8):
                shift = 7 - bit_idx
                binary[:, i*8 + bit_idx] = ((byte_val >> shift) & 1).flatten()
        
        return binary


def hamming_distance(binary1: np.ndarray, binary2: np.ndarray) -> int:
    """
    Compute Hamming distance between two binary vectors (packed).
    
    Args:
        binary1: Packed binary vector (n_bytes,)
        binary2: Packed binary vector (n_bytes,)
        
    Returns:
        Number of differing bits
    """
    xor = binary1 ^ binary2
    # Count set bits
    distance = 0
    for byte_val in xor:
        distance += bin(byte_val).count('1')
    return distance
