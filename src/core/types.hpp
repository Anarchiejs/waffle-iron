// Shared types for every core module.
// Other modules record the fields they need in their own notes instead of editing this file.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace waffle {

// Expected error (unreadable file, insufficient memory, corrupt model, …):
// the message is shown to the user as-is, so keep it precise.
struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Thrown when a job is cancelled (Engine::cancel_all or stop_token).
struct Cancelled : Error {
    Cancelled() : Error("Operation cancelled") {}
};

// Progress: fraction 0..1 and a short label. Return false to request cancellation.
using ProgressFn = std::function<bool(float fraction, const std::string& label)>;

constexpr uint64_t KiB = 1024;
constexpr uint64_t MiB = 1024 * KiB;
constexpr uint64_t GiB = 1024 * MiB;

// ---------------------------------------------------------------------------
// GGUF metadata
// ---------------------------------------------------------------------------
struct TensorInfo {
    std::string name;
    std::string type;           // ggml name: "q4_K", "f16", "f32"…
    std::vector<int64_t> shape; // ne[0..n_dims)
    uint64_t n_elements = 0;
    uint64_t n_bytes = 0;
    int layer = -1; // index parsed from "blk.<n>." ; -1 = not a layer
};

struct ModelInfo {
    std::filesystem::path path;
    std::string file_name;
    uint64_t file_size = 0;

    std::string architecture;    // general.architecture
    std::string name;            // general.name, otherwise the file name without extension
    std::string file_type;       // "Q4_K_M", "F16"… (translated general.file_type), "?" if missing
    uint64_t n_params = 0;       // sum of tensor elements
    uint32_t context_length = 0; // <arch>.context_length
    uint32_t n_layer = 0;        // <arch>.block_count
    uint32_t n_embd = 0;         // <arch>.embedding_length
    uint32_t n_head = 0;         // <arch>.attention.head_count
    uint32_t n_head_kv = 0;      // <arch>.attention.head_count_kv (default n_head)
    uint32_t head_dim_k = 0;     // <arch>.attention.key_length (default n_embd / n_head)
    uint32_t head_dim_v = 0;     // <arch>.attention.value_length (default n_embd / n_head)
    uint32_t sliding_window = 0; // <arch>.attention.sliding_window (0 = none)
    // One value per layer, 1 = sliding-window attention (<arch>.attention.sliding_window_pattern,
    // array or period); empty if the model has no SWA.
    std::vector<uint8_t> swa_layers;
    uint32_t head_dim_k_swa = 0;              // <arch>.attention.key_length_swa (default head_dim_k)
    uint32_t head_dim_v_swa = 0;              // <arch>.attention.value_length_swa (default head_dim_v)
    uint32_t shared_kv_layers = 0;            // <arch>.attention.shared_kv_layers: the last layers reuse the cache
    uint32_t n_expert = 0;                    // <arch>.expert_count (0 = dense)
    uint32_t n_expert_used = 0;               // <arch>.expert_used_count
    std::optional<std::string> chat_template; // tokenizer.chat_template

    std::map<std::string, uint64_t> bytes_by_type;             // ggml type -> total bytes
    std::vector<TensorInfo> tensors;                           // empty if read without tensors
    std::vector<std::pair<std::string, std::string>> metadata; // key -> readable value (arrays summarized)
};

// ---------------------------------------------------------------------------
// Model load by the engine
// ---------------------------------------------------------------------------
struct LoadParams {
    std::filesystem::path model_path;
    int n_gpu_layers = -1; // -1 = every layer on the GPU
    uint32_t n_ctx = 4096;
    uint32_t n_batch = 2048;
    uint32_t n_ubatch = 512;
    std::string type_k = "f16"; // ggml type of the K cache
    std::string type_v = "f16"; // ggml type of the V cache (quantized => flash attention required)
    int flash_attn = -1;        // -1 auto, 0 off, 1 on
    int n_threads = 0;          // 0 = automatic
    uint32_t n_seq_max = 1;
    // SWA cache sized to the context (llama.cpp default). false = only the window plus one micro-batch:
    // much less VRAM on long contexts, but rewinding the sequence is impossible.
    bool swa_full = true;

    bool operator==(const LoadParams&) const = default;
};

// ---------------------------------------------------------------------------
// VRAM planner
// ---------------------------------------------------------------------------
struct VramPlanInput {
    int n_gpu_layers = -1;
    uint32_t n_ctx = 4096;
    uint32_t n_batch = 2048;
    uint32_t n_ubatch = 512;
    std::string type_k = "f16";
    std::string type_v = "f16";
    bool flash_attn = true;
    uint32_t n_seq = 1;
    bool swa_full = true;
};

struct VramEstimate {
    uint64_t weights_gpu = 0;
    uint64_t weights_cpu = 0;
    uint64_t kv_cache_gpu = 0;
    uint64_t kv_cache_cpu = 0;
    uint64_t compute_gpu = 0; // compute buffers
    uint64_t compute_cpu = 0;
    uint64_t overhead_gpu = 0;      // CUDA context, cuBLAS… (calibrated constant)
    std::vector<std::string> notes; // assumptions and warnings (SWA, MoE…)

    uint64_t total_gpu() const { return weights_gpu + kv_cache_gpu + compute_gpu + overhead_gpu; }
    uint64_t total_cpu() const { return weights_cpu + kv_cache_cpu + compute_cpu; }
};

// ---------------------------------------------------------------------------
// GPU telemetry
// ---------------------------------------------------------------------------
struct GpuSample {
    double t_seconds = 0; // seconds since Telemetry was created
    uint32_t gpu_util_pct = 0;
    uint32_t mem_util_pct = 0;
    uint64_t vram_used = 0;
    uint64_t vram_total = 0;
    uint32_t temperature_c = 0;
    double power_w = 0;
    uint32_t sm_clock_mhz = 0;
};

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------
struct BenchConfig {
    LoadParams load;
    uint32_t n_prompt = 512; // prompt tokens measured (pp)
    uint32_t n_gen = 128;    // generated tokens measured (tg)
    int repetitions = 5;
    int warmup = 1; // discarded runs before measurement
};

struct BenchSample {
    double pp_tps = 0;      // prompt processing, tokens/s
    double tg_tps = 0;      // generation, tokens/s
    double ttft_ms = 0;     // time to first token (prompt + 1 token)
    uint64_t vram_peak = 0; // bytes, 0 if unknown
};

struct Stats {
    int n = 0;
    double mean = 0, stddev = 0, median = 0, min = 0, max = 0;
    std::vector<int> outliers; // indices of outliers (modified z-score > 3.5)
};

struct BenchResult {
    BenchConfig config;
    std::string model_name;
    std::string model_file_type;
    double load_ms = 0;
    std::vector<BenchSample> samples;
    Stats pp, tg, ttft;
    uint64_t vram_peak = 0;
    std::string gpu_name;
    std::string llama_version; // WAFFLE_LLAMA_VERSION
    std::string timestamp;     // ISO 8601 UTC
};

} // namespace waffle
