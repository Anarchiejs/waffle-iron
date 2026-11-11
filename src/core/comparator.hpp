// A/B comparator: the same prompt on two models (or two quantizations) loaded one after the other.
#pragma once

#include "core/types.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace waffle {

class Engine;
class EngineSession;
class Telemetry;

struct CompareSide {
    std::string label; // "A", "B", or a free-form name
    LoadParams load;
};

struct CompareRequest {
    CompareSide a, b;
    std::string prompt; // user prompt (passed through apply_chat_template)
    int max_tokens = 256;
    float temperature = 0.0f; // 0 = greedy (reproducible comparison)
    uint32_t seed = 42;
    std::string perplexity_text; // empty = no perplexity
};

struct CompareOutput {
    std::string label;
    std::string model_name;
    std::string text;
    int n_generated = 0;
    double pp_tps = 0;
    double tg_tps = 0;
    double load_ms = 0;
    uint64_t vram_peak = 0;
    std::optional<double> perplexity;
    std::string error; // non-empty if this side failed (the other side is still produced)
};

struct CompareResult {
    CompareOutput a, b;
};

// Perplexity on the loaded model: text split into windows of n_ctx tokens (at most the loaded
// context), log-likelihood on the second half of each window (as in llama-perplexity).
// Throws Error if the text is shorter than 2 windows of 16 tokens.
double compute_perplexity(EngineSession& session, std::string_view text, uint32_t n_ctx = 512);

// Generates on the loaded model. on_token receives each text piece (may be empty).
CompareOutput generate_side(EngineSession& session, const CompareRequest& req,
                            const std::function<void(const std::string&)>& on_token = {});

// Full orchestration: load A, generate (+ perplexity), unload, load B, same again.
// Blocks the calling thread; on_token(side 0|1, piece) for live display.
CompareResult run_compare(Engine& engine, Telemetry* telemetry, const CompareRequest& req,
                          const std::function<void(int side, const std::string&)>& on_token = {},
                          const ProgressFn& progress = {});

} // namespace waffle
