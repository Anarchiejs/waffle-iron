// Estimates the memory required before loading (weights, KV cache, buffers).
#pragma once

#include "core/types.hpp"

#include <string_view>

namespace waffle {

// Bytes used by n_elements of a named ggml type ("f16", "q8_0", "q4_0", "q4_K"…).
// Throws waffle::Error if the type is unknown.
uint64_t ggml_type_bytes(std::string_view type_name, uint64_t n_elements);

// Types accepted for the KV cache, in display order: f32, f16, bf16, q8_0, q5_1, q5_0, q4_1, q4_0, iq4_nl.
const std::vector<std::string>& kv_cache_types();

// Estimate, following llama.cpp:
// - weights: tensors "blk.<i>." go to the GPU when i >= n_layer - n_gpu_layers; token_embd stays on the CPU;
//   output.* goes to the GPU when every layer is there;
// - KV cache: per layer n_ctx_eff * n_head_kv * (head_dim_k * size(type_k) + head_dim_v * size(type_v)),
//   n_ctx_eff = min(n_ctx, sliding_window) for SWA layers when the architecture has them;
// - compute buffers: a function of n_ubatch, n_embd, n_vocab, n_ctx (formula documented in the .cpp,
//   calibrated against real measurements);
// - GPU overhead: a constant (CUDA context + cuBLAS), 0 when n_gpu_layers == 0.
VramEstimate estimate_vram(const ModelInfo& model, const VramPlanInput& input);

// Largest n_gpu_layers whose total_gpu() fits in budget_bytes (0 if no layer fits).
int max_gpu_layers_for(const ModelInfo& model, VramPlanInput input, uint64_t budget_bytes);

} // namespace waffle
