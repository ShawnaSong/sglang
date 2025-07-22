 /* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cuda_runtime.h>

#include <cub/block/block_reduce.cuh>
#include <cuda/std/cmath>
#include <cuda/std/algorithm>
#include <cub/block/block_reduce.cuh>

// FP8 to FP4 KV Cache Compression System
// 
// Storage format: 9 int8 per group
// - 8 int8: 16 FP4 values (2 FP4 per int8)
// - 1 int8: 1 FP8 scale factor
// 
// Total: 16 FP4 + 1 FP8 = 9 int8 per group

// Constants
constexpr int GROUP_SIZE = 16;     // 16 tensors per group
constexpr int INT8_PER_GROUP = 9;  // 8 for FP4 + 1 for FP8 scale

// Convert FP8 to FP4 with scale calculation
__device__ __forceinline__ uint8_t fp8_to_fp4(__nv_fp8_e4m3 fp8_val, float scale) {
  float val = (float)fp8_val / scale;
  // Clamp to FP4 range and convert
  val = fmaxf(fminf(val, 7.0f), -8.0f);  // FP4 range: -8 to 7
  return (uint8_t)(val + 8);             // Convert to 0-15 range
}

// Convert FP4 back to FP8
__device__ __forceinline__ __nv_fp8_e4m3 fp4_to_fp8(uint8_t fp4_val, float scale) {
  float val = (float)(fp4_val - 8) * scale;  // Convert back from 0-15 range
  return __nv_fp8_e4m3(val);
}

// Pack 16 FP8 values into 16 FP4 (8 INT8) values + 1 FP8 scale
__device__ __forceinline__ void
pack_fp8_group_to_fp4(const __nv_fp8_e4m3* fp8_values, uint8_t* packed_data, int group_offset) {
  // Find max absolute value for scale calculation
  float max_abs_val = 0.0f;
  for (int i = 0; i < GROUP_SIZE; i++) {
    float val = (float)fp8_values[i];
    max_abs_val = fmaxf(max_abs_val, fabsf(val));
  }

  // Calculate scale factor (FP8)
  float scale = max_abs_val / 7.0f;  // Normalize to FP4 range
  __nv_fp8_e4m3 fp8_scale = __nv_fp8_e4m3(scale);

  // Store scale as int8 (last byte in group)
  uint8_t* scale_ptr = packed_data + group_offset * INT8_PER_GROUP + 8;
  *scale_ptr = (uint8_t)fp8_scale;

  // Pack 16 FP8 values into 8 FP4 values
  uint8_t* data_ptr = packed_data + group_offset * INT8_PER_GROUP;
  for (int i = 0; i < GROUP_SIZE; i += 2) {
    uint8_t fp4_1 = fp8_to_fp4(fp8_values[i], scale);
    uint8_t fp4_2 = fp8_to_fp4(fp8_values[i + 1], scale);

    // Pack 2 FP4 values into 1 int8
    uint8_t packed = (fp4_2 << 4) | fp4_1;
    data_ptr[i / 2] = packed;
  }
}

// Unpack 8 INT8 (16 FP4) values + 1 FP8 scale back to 16 FP8 values
__device__ __forceinline__ void
unpack_fp4_group_to_fp8(const uint8_t* packed_data, __nv_fp8_e4m3* fp8_values, int group_offset) {
  const uint8_t* data_ptr = packed_data + group_offset * INT8_PER_GROUP;
  const uint8_t* scale_ptr = packed_data + group_offset * INT8_PER_GROUP + 8;

  // Extract scale factor
  __nv_fp8_e4m3 fp8_scale = (__nv_fp8_e4m3)*scale_ptr;
  float scale = (float)fp8_scale;

  // Unpack 8 FP4 values into 16 FP8 values
  for (int i = 0; i < GROUP_SIZE; i += 2) {
    uint8_t packed = data_ptr[i / 2];
    uint8_t fp4_1 = packed & 0x0F;
    uint8_t fp4_2 = (packed >> 4) & 0x0F;

    fp8_values[i] = fp4_to_fp8(fp4_1, scale);
    fp8_values[i + 1] = fp4_to_fp8(fp4_2, scale);
  }
}

// Kernel to compute FP8 KV cache and compress to FP4 storage
__global__ void compute_and_compress_fp8_kv_cache(
    const float* input_k,           // Input key tensor (FP32)
    const float* input_v,           // Input value tensor (FP32)
    uint8_t* compressed_k_cache,    // Compressed key cache (FP4 + FP8 scale)
    uint8_t* compressed_v_cache,    // Compressed value cache (FP4 + FP8 scale)
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const int cache_offset          // Offset in cache for this token
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * num_heads * head_dim;

  if (tid >= total_elements) return;

  // Calculate indices
  int batch_idx = tid / (num_heads * head_dim);
  int head_idx = (tid / head_dim) % num_heads;
  int dim_idx = tid % head_dim;

  // Convert input to FP8
  __nv_fp8_e4m3 k_fp8 = __nv_fp8_e4m3(input_k[tid]);
  __nv_fp8_e4m3 v_fp8 = __nv_fp8_e4m3(input_v[tid]);

  // Calculate group index
  int group_idx = tid / GROUP_SIZE;
  int group_offset = tid % GROUP_SIZE;

  // Use shared memory for group processing
  extern __shared__ __nv_fp8_e4m3 shared_fp8[];
  __nv_fp8_e4m3* k_group = shared_fp8;
  __nv_fp8_e4m3* v_group = shared_fp8 + GROUP_SIZE;

  // Load values into shared memory
  k_group[group_offset] = k_fp8;
  v_group[group_offset] = v_fp8;

  __syncthreads();

  // Only first thread in group packs the data
  if (group_offset == 0) {
    // Calculate cache indices
    int cache_k_offset = (batch_idx * num_heads * seq_len + cache_offset) * head_dim + group_idx * GROUP_SIZE;
    int cache_v_offset = (batch_idx * num_heads * seq_len + cache_offset) * head_dim + group_idx * GROUP_SIZE;

    // Pack key group
    pack_fp8_group_to_fp4(k_group, compressed_k_cache, cache_k_offset / GROUP_SIZE);

    // Pack value group
    pack_fp8_group_to_fp4(v_group, compressed_v_cache, cache_v_offset / GROUP_SIZE);
  }
}

// Kernel to decompress FP4 cache back to FP8 for computation
__global__ void decompress_fp4_cache_to_fp8(
    const uint8_t* compressed_k_cache,  // Compressed key cache
    const uint8_t* compressed_v_cache,  // Compressed value cache
    __nv_fp8_e4m3* k_fp8_output,       // Decompressed key (FP8)
    __nv_fp8_e4m3* v_fp8_output,       // Decompressed value (FP8)
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const int cache_start,              // Start position in cache
    const int cache_length              // Number of tokens to decompress
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_heads * cache_length * head_dim;
    
    if (tid >= total_elements) return;
    
    // Calculate indices
    int batch_idx = tid / (num_heads * cache_length * head_dim);
    int head_idx = (tid / (cache_length * head_dim)) % num_heads;
    int seq_idx = (tid / head_dim) % cache_length;
    int dim_idx = tid % head_dim;
    
    // Calculate cache positions
    int cache_pos = cache_start + seq_idx;
    int cache_k_offset = (batch_idx * num_heads * seq_len + cache_pos) * head_dim + dim_idx;
    int cache_v_offset = (batch_idx * num_heads * seq_len + cache_pos) * head_dim + dim_idx;
    
    // Calculate group indices
    int group_idx = dim_idx / GROUP_SIZE;
    int group_offset = dim_idx % GROUP_SIZE;
    
    // Use shared memory for group processing
    extern __shared__ __nv_fp8_e4m3 shared_fp8[];
    __nv_fp8_e4m3* k_group = shared_fp8;
    __nv_fp8_e4m3* v_group = shared_fp8 + GROUP_SIZE;
    
    // Only first thread in group unpacks the data
    if (group_offset == 0) {
        // Unpack key group
        unpack_fp4_group_to_fp8(compressed_k_cache, k_group, cache_k_offset / GROUP_SIZE);
        
        // Unpack value group
        unpack_fp4_group_to_fp8(compressed_v_cache, v_group, cache_v_offset / GROUP_SIZE);
    }
    
    __syncthreads();
    
    // Copy from shared memory to output
    k_fp8_output[tid] = k_group[group_offset];
    v_fp8_output[tid] = v_group[group_offset];
}

// Entry function for computing and compressing KV cache
void compute_and_compress_fp8_kv_cache_forward(
    const float* input_k,
    const float* input_v,
    uint8_t* compressed_k_cache,
    uint8_t* compressed_v_cache,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const int cache_offset,
    cudaStream_t stream
) {
    int total_elements = batch_size * num_heads * head_dim;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    int shared_memory_size = 2 * GROUP_SIZE * sizeof(__nv_fp8_e4m3);
    
    compute_and_compress_fp8_kv_cache<<<blocks, threads_per_block, shared_memory_size, stream>>>(
        input_k, input_v, compressed_k_cache, compressed_v_cache,
        batch_size, num_heads, seq_len, head_dim, cache_offset
    );
}

// Entry function for decompressing KV cache
void decompress_fp4_cache_to_fp8_forward(
    const uint8_t* compressed_k_cache,
    const uint8_t* compressed_v_cache,
    __nv_fp8_e4m3* k_fp8_output,
    __nv_fp8_e4m3* v_fp8_output,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const int cache_start,
    const int cache_length,
    cudaStream_t stream
) {
    int total_elements = batch_size * num_heads * cache_length * head_dim;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    int shared_memory_size = 2 * GROUP_SIZE * sizeof(__nv_fp8_e4m3);
    
    decompress_fp4_cache_to_fp8<<<blocks, threads_per_block, shared_memory_size, stream>>>(
        compressed_k_cache, compressed_v_cache, k_fp8_output, v_fp8_output,
        batch_size, num_heads, seq_len, head_dim, cache_start, cache_length
    );
}

// PyTorch wrapper functions
#include <torch/all.h>

// PyTorch wrapper for compute and compress FP8 KV cache
void compute_and_compress_fp8_kv_cache_forward_wrapper(
    const torch::Tensor& input_k,
    const torch::Tensor& input_v,
    torch::Tensor& compressed_k_cache,
    torch::Tensor& compressed_v_cache,
    int64_t batch_size,
    int64_t num_heads,
    int64_t seq_len,
    int64_t head_dim,
    int64_t cache_offset,
    int64_t cuda_stream) {
  // Validate input tensors
  TORCH_CHECK(input_k.is_cuda(), "input_k must be a CUDA tensor");
  TORCH_CHECK(input_v.is_cuda(), "input_v must be a CUDA tensor");
  TORCH_CHECK(compressed_k_cache.is_cuda(), "compressed_k_cache must be a CUDA tensor");
  TORCH_CHECK(compressed_v_cache.is_cuda(), "compressed_v_cache must be a CUDA tensor");

  TORCH_CHECK(input_k.dtype() == torch::kFloat32, "input_k must be float32");
  TORCH_CHECK(input_v.dtype() == torch::kFloat32, "input_v must be float32");
  TORCH_CHECK(compressed_k_cache.dtype() == torch::kUInt8, "compressed_k_cache must be uint8");
  TORCH_CHECK(compressed_v_cache.dtype() == torch::kUInt8, "compressed_v_cache must be uint8");

  // Get tensor data pointers
  const float* input_k_ptr = input_k.data_ptr<float>();
  const float* input_v_ptr = input_v.data_ptr<float>();
  uint8_t* compressed_k_cache_ptr = compressed_k_cache.data_ptr<uint8_t>();
  uint8_t* compressed_v_cache_ptr = compressed_v_cache.data_ptr<uint8_t>();

  // Get CUDA stream
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

  // Call CUDA kernel
  compute_and_compress_fp8_kv_cache_forward(
      input_k_ptr,
      input_v_ptr,
      compressed_k_cache_ptr,
      compressed_v_cache_ptr,
      static_cast<int>(batch_size),
      static_cast<int>(num_heads),
      static_cast<int>(seq_len),
      static_cast<int>(head_dim),
      static_cast<int>(cache_offset),
      stream);
}

// PyTorch wrapper for decompress FP4 cache to FP8
void decompress_fp4_cache_to_fp8_forward_wrapper(
    const torch::Tensor& compressed_k_cache,
    const torch::Tensor& compressed_v_cache,
    torch::Tensor& k_fp8_output,
    torch::Tensor& v_fp8_output,
    int64_t batch_size,
    int64_t num_heads,
    int64_t seq_len,
    int64_t head_dim,
    int64_t cache_start,
    int64_t cache_length,
    int64_t cuda_stream) {
  // Validate input tensors
  TORCH_CHECK(compressed_k_cache.is_cuda(), "compressed_k_cache must be a CUDA tensor");
  TORCH_CHECK(compressed_v_cache.is_cuda(), "compressed_v_cache must be a CUDA tensor");
  TORCH_CHECK(k_fp8_output.is_cuda(), "k_fp8_output must be a CUDA tensor");
  TORCH_CHECK(v_fp8_output.is_cuda(), "v_fp8_output must be a CUDA tensor");

  TORCH_CHECK(compressed_k_cache.dtype() == torch::kUInt8, "compressed_k_cache must be uint8");
  TORCH_CHECK(compressed_v_cache.dtype() == torch::kUInt8, "compressed_v_cache must be uint8");
  TORCH_CHECK(k_fp8_output.dtype() == torch::kFloat8_e4m3fn, "k_fp8_output must be fp8");
  TORCH_CHECK(v_fp8_output.dtype() == torch::kFloat8_e4m3fn, "v_fp8_output must be fp8");

  // Get tensor data pointers
  const uint8_t* compressed_k_cache_ptr = compressed_k_cache.data_ptr<uint8_t>();
  const uint8_t* compressed_v_cache_ptr = compressed_v_cache.data_ptr<uint8_t>();
  __nv_fp8_e4m3* k_fp8_output_ptr = reinterpret_cast<__nv_fp8_e4m3*>(k_fp8_output.data_ptr());
  __nv_fp8_e4m3* v_fp8_output_ptr = reinterpret_cast<__nv_fp8_e4m3*>(v_fp8_output.data_ptr());

  // Get CUDA stream
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

  // Call CUDA kernel
  decompress_fp4_cache_to_fp8_forward(
      compressed_k_cache_ptr,
      compressed_v_cache_ptr,
      k_fp8_output_ptr,
      v_fp8_output_ptr,
      static_cast<int>(batch_size),
      static_cast<int>(num_heads),
      static_cast<int>(seq_len),
      static_cast<int>(head_dim),
      static_cast<int>(cache_start),
      static_cast<int>(cache_length),
      stream);
}
