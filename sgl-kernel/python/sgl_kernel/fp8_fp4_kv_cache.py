"""
FP8 to FP4 KV Cache Compression Module

This module provides Python interfaces for compressing FP8 KV cache to FP4 format
and decompressing FP4 cache back to FP8 for efficient memory usage.
"""

import torch
from typing import Optional, Tuple


def compute_and_compress_fp8_kv_cache(
    input_k: torch.Tensor,
    input_v: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_v_cache: torch.Tensor,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    cache_offset: int,
    cuda_stream: Optional[int] = None
) -> None:
    """
    Compute FP8 KV cache and compress to FP4 storage format.
    
    Args:
        input_k: Input key tensor (FP32) [batch_size, num_heads, head_dim]
        input_v: Input value tensor (FP32) [batch_size, num_heads, head_dim]
        compressed_k_cache: Compressed key cache (UInt8) - output tensor
        compressed_v_cache: Compressed value cache (UInt8) - output tensor
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        cache_offset: Offset in cache for this token
        cuda_stream: CUDA stream (optional)
    
    Storage format: 9 int8 per group
    - 8 int8: 16 FP4 values (2 FP4 per int8)
    - 1 int8: 1 FP8 scale factor
    - Total: 16 FP4 + 1 FP8 = 9 int8 per group
    """
    if cuda_stream is None:
        cuda_stream = torch.cuda.current_stream().cuda_stream
    
    torch.ops.sgl_kernel.compute_and_compress_fp8_kv_cache_forward(
        input_k, input_v, compressed_k_cache, compressed_v_cache,
        batch_size, num_heads, seq_len, head_dim, cache_offset, cuda_stream
    )


def decompress_fp4_cache_to_fp8(
    compressed_k_cache: torch.Tensor,
    compressed_v_cache: torch.Tensor,
    k_fp8_output: torch.Tensor,
    v_fp8_output: torch.Tensor,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    cache_start: int,
    cache_length: int,
    cuda_stream: Optional[int] = None
) -> None:
    """
    Decompress FP4 cache back to FP8 for computation.
    
    Args:
        compressed_k_cache: Compressed key cache (UInt8)
        compressed_v_cache: Compressed value cache (UInt8)
        k_fp8_output: Decompressed key (FP8) - output tensor
        v_fp8_output: Decompressed value (FP8) - output tensor
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        cache_start: Start position in cache
        cache_length: Number of tokens to decompress
        cuda_stream: CUDA stream (optional)
    """
    if cuda_stream is None:
        cuda_stream = torch.cuda.current_stream().cuda_stream
    
    torch.ops.sgl_kernel.decompress_fp4_cache_to_fp8_forward(
        compressed_k_cache, compressed_v_cache, k_fp8_output, v_fp8_output,
        batch_size, num_heads, seq_len, head_dim, cache_start, cache_length, cuda_stream
    )


def create_compressed_cache_tensors(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create compressed cache tensors with the correct size.
    
    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        device: Device to create tensors on
    
    Returns:
        Tuple of (compressed_k_cache, compressed_v_cache) tensors
    """
    # Calculate compressed size
    # Each group of 16 FP4 values + 1 FP8 scale = 9 int8
    group_size = 16
    groups_per_head_dim = (head_dim + group_size - 1) // group_size
    int8_per_group = 9
    
    compressed_size = batch_size * num_heads * seq_len * groups_per_head_dim * int8_per_group
    
    compressed_k_cache = torch.zeros(compressed_size, dtype=torch.uint8, device=device)
    compressed_v_cache = torch.zeros(compressed_size, dtype=torch.uint8, device=device)
    
    return compressed_k_cache, compressed_v_cache


def create_fp8_output_tensors(
    batch_size: int,
    num_heads: int,
    cache_length: int,
    head_dim: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create FP8 output tensors for decompression.
    
    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        cache_length: Number of tokens to decompress
        head_dim: Head dimension
        device: Device to create tensors on
    
    Returns:
        Tuple of (k_fp8_output, v_fp8_output) tensors
    """
    output_size = batch_size * num_heads * cache_length * head_dim
    
    k_fp8_output = torch.zeros(output_size, dtype=torch.float8_e4m3fn, device=device)
    v_fp8_output = torch.zeros(output_size, dtype=torch.float8_e4m3fn, device=device)
    
    return k_fp8_output, v_fp8_output


def calculate_memory_savings(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    num_layers: int
) -> float:
    """
    Calculate memory savings percentage from FP8 to FP4 compression.
    
    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        num_layers: Number of layers
    
    Returns:
        Memory savings percentage
    """
    # Standard FP8 storage (1 byte per value)
    standard_memory = batch_size * seq_len * num_heads * head_dim * 2 * num_layers
    
    # FP8-FP4 compressed storage
    group_size = 16
    int8_per_group = 9
    groups_per_head_dim = (head_dim + group_size - 1) // group_size
    compressed_memory = batch_size * seq_len * num_heads * groups_per_head_dim * int8_per_group * num_layers
    
    savings = (standard_memory - compressed_memory) / standard_memory * 100
    return savings


class FP8FP4KVCache:
    """
    High-level interface for FP8 to FP4 KV cache compression.
    """
    
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        device: torch.device
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.device = device
        
        # Create compressed cache tensors
        self.compressed_k_cache, self.compressed_v_cache = create_compressed_cache_tensors(
            batch_size, num_heads, seq_len, head_dim, device
        )
        
        # Track current position
        self.current_offset = 0
    
    def compress_and_store(
        self,
        input_k: torch.Tensor,
        input_v: torch.Tensor,
        cuda_stream: Optional[int] = None
    ) -> int:
        """
        Compress and store KV cache for current token.
        
        Args:
            input_k: Input key tensor (FP32)
            input_v: Input value tensor (FP32)
            cuda_stream: CUDA stream (optional)
        
        Returns:
            Cache offset for this token
        """
        if self.current_offset >= self.seq_len:
            raise ValueError("Cache is full")
        
        compute_and_compress_fp8_kv_cache(
            input_k, input_v, self.compressed_k_cache, self.compressed_v_cache,
            self.batch_size, self.num_heads, self.seq_len, self.head_dim,
            self.current_offset, cuda_stream
        )
        
        offset = self.current_offset
        self.current_offset += 1
        return offset
    
    def decompress_range(
        self,
        cache_start: int,
        cache_length: int,
        cuda_stream: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress a range of cached tokens.
        
        Args:
            cache_start: Start position in cache
            cache_length: Number of tokens to decompress
            cuda_stream: CUDA stream (optional)
        
        Returns:
            Tuple of (k_fp8, v_fp8) tensors
        """
        k_fp8_output, v_fp8_output = create_fp8_output_tensors(
            self.batch_size, self.num_heads, cache_length, self.head_dim, self.device
        )
        
        decompress_fp4_cache_to_fp8(
            self.compressed_k_cache, self.compressed_v_cache,
            k_fp8_output, v_fp8_output,
            self.batch_size, self.num_heads, self.seq_len, self.head_dim,
            cache_start, cache_length, cuda_stream
        )
        
        # Reshape to standard format
        k_fp8 = k_fp8_output.view(self.batch_size, self.num_heads, cache_length, self.head_dim)
        v_fp8 = v_fp8_output.view(self.batch_size, self.num_heads, cache_length, self.head_dim)
        
        return k_fp8, v_fp8
    
    def get_memory_usage(self) -> float:
        """Get memory usage in GB."""
        total_elements = self.compressed_k_cache.numel() + self.compressed_v_cache.numel()
        memory_gb = total_elements * 1 / (1024**3)  # 1 byte per uint8
        return memory_gb
    
    def reset(self):
        """Reset cache to empty state."""
        self.current_offset = 0
        self.compressed_k_cache.zero_()
        self.compressed_v_cache.zero_() 