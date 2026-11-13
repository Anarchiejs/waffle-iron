// Benchmark: speed measurements (pp, tg, ttft), statistics, grid campaigns, and export.
#pragma once

#include "core/types.hpp"

#include <filesystem>
#include <span>
#include <string_view>
#include <string_view>
#include <stop_token>
#include <string>
#include <vector>

namespace waffle {

class Engine;
class EngineSession;
class Telemetry;

// Mean, sample standard deviation (n-1), median, min, max; outliers by modified z-score
// (0.6745 * |x - median| / MAD > 3.5; none if MAD == 0). n == 0 -> all zeros.
Stats compute_stats(std::span<const double> values);

// One measurement, on the engine thread, model already loaded with cfg.load:
// clears the cache, decodes n_prompt random tokens (fixed seed, excluding special tokens) -> pp_tps,
// then generates n_gen tokens one by one greedily -> tg_tps; ttft = prompt + first token.
// vram_peak stays 0 (filled in by run_bench).
BenchSample run_bench_once(EngineSession& session, const BenchConfig& cfg);

// Loads the model, runs warmup + repetitions measurements, and computes the stats.
// Blocks the calling thread (not the engine thread); submits work to the Engine.
// telemetry may be nullptr (vram_peak = 0). progress may cancel.
BenchResult run_bench(Engine& engine, Telemetry* telemetry, const BenchConfig& cfg, const ProgressFn& progress = {});

// Settings grid: Cartesian product of the non-empty lists (an empty list keeps `base`).
struct CampaignGrid {
    BenchConfig base;
    std::vector<int> n_gpu_layers;
    std::vector<uint32_t> n_ctx;
    std::vector<std::string> kv_type; // applied to type_k AND type_v
    std::vector<int> flash_attn;
    std::vector<uint32_t> n_ubatch;
    // Extra model files. Empty keeps base.load.model_path.
    std::vector<std::filesystem::path> models;
};
std::vector<BenchConfig> expand_grid(const CampaignGrid& grid);

// Full campaign; one BenchResult per configuration (a configuration that fails — insufficient
// memory… — is skipped and its error is appended to `errors` when non-null).
std::vector<BenchResult> run_campaign(Engine& engine, Telemetry* telemetry, const CampaignGrid& grid,
                                      const ProgressFn& progress = {}, std::vector<std::string>* errors = nullptr);

// Exports: one line per BenchResult (mean ± standard deviation), stable documented columns.
std::string to_csv(const std::vector<BenchResult>& results);
std::string to_markdown(const std::vector<BenchResult>& results);

// Short configuration label: "ngl=99 ctx=4096 kv=q8_0 fa=on ub=512 swa=full".
std::string config_label(const BenchConfig& cfg);

std::string results_to_json(const std::vector<BenchResult>& results);
std::vector<BenchResult> results_from_json(std::string_view text);

} // namespace waffle
