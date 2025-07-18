# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Radix attention."""

from enum import Enum
from typing import Optional

import torch
from torch import nn

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# Import FP8-FP4 KV cache functionality
try:
    from sgl_kernel.fp8_fp4_kv_cache import (
        FP8FP4KVCache,
        calculate_memory_savings,
    )
    FP8_FP4_AVAILABLE = True
except ImportError:
    FP8_FP4_AVAILABLE = False


class AttentionType(Enum):
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """

    # Decoder attention between previous layer Q/K/V
    DECODER = "decoder"
    # Encoder attention between previous layer Q/K/V
    ENCODER_ONLY = "encoder_only"


class RadixAttention(nn.Module):
    """
    The attention layer implementation.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        logit_cap: float = 0.0,
        v_head_dim: int = -1,
        sliding_window_size: int = -1,
        is_cross_attention: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        use_irope: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.qk_head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim != -1 else head_dim
        self.scaling = scaling
        self.layer_id = layer_id
        self.logit_cap = logit_cap
        self.sliding_window_size = sliding_window_size or -1
        self.is_cross_attention = is_cross_attention
        self.use_irope = use_irope
        self.k_scale = None
        self.v_scale = None
        self.k_scale_float = None
        self.v_scale_float = None
        self.quant_method = None
        if quant_config is not None:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if self.quant_method is not None:
            self.quant_method.create_weights(self)
        self.attn_type = attn_type

    def forward(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if k is not None:
            # For cross-layer sharing, kv can be None
            assert v is not None
            if "k_rope" not in kwargs:
                k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
            else:
                k = k.view(-1, self.tp_k_head_num, self.v_head_dim)

        return forward_batch.attn_backend.forward(
            q,
            k,
            v,
            self,
            forward_batch,
            save_kv_cache,
            **kwargs,
        )


class FP8FP4RadixAttention(RadixAttention):
    """
    FP8-FP4 Radix Attention with KV cache compression.
    
    This class extends RadixAttention to use FP8-FP4 KV cache compression.
    It compresses FP8 KV cache to FP4 format for storage and decompresses
    back to FP8 for computation.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        logit_cap: float = 0.0,
        v_head_dim: int = -1,
        sliding_window_size: int = -1,
        is_cross_attention: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        use_irope: bool = False,
        prefix: str = "",
        enable_fp8_fp4_cache: bool = True,
    ):
        # Call parent constructor
        super().__init__(
            num_heads=num_heads,
            head_dim=head_dim,
            scaling=scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
            logit_cap=logit_cap,
            v_head_dim=v_head_dim,
            sliding_window_size=sliding_window_size,
            is_cross_attention=is_cross_attention,
            quant_config=quant_config,
            attn_type=attn_type,
            use_irope=use_irope,
            prefix=prefix,
        )
        
        # FP8-FP4 cache parameters
        self.enable_fp8_fp4_cache = enable_fp8_fp4_cache and FP8_FP4_AVAILABLE
        
        # Initialize FP8-FP4 cache if enabled
        self.fp8_fp4_cache = None
        if self.enable_fp8_fp4_cache:
            self._initialize_fp8_fp4_cache()
    
    def _initialize_fp8_fp4_cache(self):
        """Initialize FP8-FP4 KV cache."""
        if not FP8_FP4_AVAILABLE:
            raise ImportError("FP8-FP4 KV cache compression not available")
        
        # Cache will be initialized dynamically when first used
        self.fp8_fp4_cache = None
    
    def _ensure_cache_initialized(self, batch_size: int, seq_len: int):
        """Ensure FP8-FP4 cache is initialized with correct dimensions."""
        if self.fp8_fp4_cache is None:
            device = next(self.parameters()).device if list(self.parameters()) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.fp8_fp4_cache = FP8FP4KVCache(
                batch_size=batch_size,
                num_heads=self.tp_k_head_num,
                seq_len=seq_len,
                head_dim=self.head_dim,
                device=device
            )
    
    def forward(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """
        Forward pass with FP8-FP4 KV cache compression.
        
        This method:
        1. Compresses K/V to FP8-FP4 format and stores in cache
        2. Decompresses cached K/V for attention computation
        3. Performs attention with decompressed K/V
        """
        if k is not None:
            # For cross-layer sharing, kv can be None
            assert v is not None
            if "k_rope" not in kwargs:
                k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
            else:
                k = k.view(-1, self.tp_k_head_num, self.v_head_dim)
        
        # Use FP8-FP4 cache if enabled
        if self.enable_fp8_fp4_cache and save_kv_cache:
            return self._forward_with_fp8_fp4_cache(q, k, v, forward_batch, **kwargs)
        else:
            # Fall back to standard attention
            return super().forward(q, k, v, forward_batch, save_kv_cache, **kwargs)
    
    def _forward_with_fp8_fp4_cache(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        """
        Forward pass with FP8-FP4 KV cache compression.
        """
        batch_size = q.size(0)
        seq_len = q.size(1)
        
        # Initialize cache if needed
        if self.fp8_fp4_cache is None:
            self._ensure_cache_initialized(batch_size, seq_len)
        
        # Step 1: Compress and store current K/V in FP8-FP4 cache
        if k is not None and v is not None:
            # Convert to FP8 for compression
            k_fp8 = k.to(torch.float8_e4m3fn)
            v_fp8 = v.to(torch.float8_e4m3fn)
            
            # Compress and store in cache
            cache_offset = self.fp8_fp4_cache.compress_and_store(
                k_fp8, v_fp8
            )
        
        # Step 2: Decompress cached K/V for attention computation
        if self.fp8_fp4_cache.current_offset > 0:
            # Decompress all cached tokens
            k_decompressed, v_decompressed = self.fp8_fp4_cache.decompress_range(
                cache_start=0,
                cache_length=self.fp8_fp4_cache.current_offset
            )
            
            # Reshape for attention computation
            k_decompressed = k_decompressed.view(
                batch_size, self.fp8_fp4_cache.current_offset, 
                self.tp_k_head_num, self.head_dim
            )
            v_decompressed = v_decompressed.view(
                batch_size, self.fp8_fp4_cache.current_offset,
                self.tp_v_head_num, self.head_dim
            )
            
            # Use decompressed K/V for attention
            k = k_decompressed
            v = v_decompressed
        
        # Step 3: Perform attention with standard backend
        return super().forward(q, k, v, forward_batch, save_kv_cache=False, **kwargs)
    
    def get_cache_memory_usage(self) -> float:
        """
        Get current cache memory usage in GB.
        
        Returns:
            Memory usage in GB
        """
        if self.fp8_fp4_cache is None:
            return 0.0
        
        # Calculate memory usage
        total_elements = (
            self.fp8_fp4_cache.compressed_k_cache.numel() +
            self.fp8_fp4_cache.compressed_v_cache.numel()
        )
        
        # Convert to GB (uint8 = 1 byte per element)
        memory_gb = total_elements / (1024**3)
        return memory_gb
    
    def get_memory_savings(self) -> float:
        """
        Calculate memory savings compared to standard FP8 storage.
        
        Returns:
            Memory savings percentage
        """
        if not self.enable_fp8_fp4_cache or self.fp8_fp4_cache is None:
            return 0.0
        
        # Get current cache dimensions
        batch_size = self.fp8_fp4_cache.batch_size
        seq_len = self.fp8_fp4_cache.seq_len
        
        return calculate_memory_savings(
            batch_size=batch_size,
            num_heads=self.tp_k_head_num,
            seq_len=seq_len,
            head_dim=self.head_dim,
            num_layers=1  # Per layer
        )
    
    def reset_cache(self):
        """Reset the FP8-FP4 cache."""
        if self.fp8_fp4_cache is not None:
            self.fp8_fp4_cache.reset()
    
    def get_cache_info(self) -> dict:
        """
        Get information about the current cache state.
        
        Returns:
            Dictionary with cache information
        """
        if self.fp8_fp4_cache is None:
            return {
                "enabled": self.enable_fp8_fp4_cache,
                "initialized": False,
                "current_offset": 0,
                "memory_usage_gb": 0.0,
                "memory_savings_percent": 0.0
            }
        
        return {
            "enabled": True,
            "initialized": True,
            "current_offset": self.fp8_fp4_cache.current_offset,
            "memory_usage_gb": self.get_cache_memory_usage(),
            "memory_savings_percent": self.get_memory_savings(),
            "batch_size": self.fp8_fp4_cache.batch_size,
            "seq_len": self.fp8_fp4_cache.seq_len
        }
