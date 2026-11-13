// Sampling lab: step-by-step generation, top-N candidates, branch tree.
// Every method that takes EngineSession& is called INSIDE an engine job.
// Const accessors (nodes, current…) are read under the caller's mutex: the UI
// copies state via snapshot() returned by a job and never reads the object directly.
#pragma once

#include "core/types.hpp"

#include <optional>
#include <string>
#include <vector>

namespace waffle {

class EngineSession;

struct SamplerSettings {
    float temperature = 0.8f;
    int top_k = 40;              // 0 = disabled
    float top_p = 0.95f;         // 1 = disabled
    float min_p = 0.05f;         // 0 = disabled
    float repeat_penalty = 1.0f; // 1 = disabled
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    int penalty_last_n = 64;
    uint32_t seed = 42;
    bool mirostat_v2 = false;
    float mirostat_tau = 5.0f;
    float mirostat_eta = 0.1f;
    bool dry = false;
    float dry_multiplier = 0.8f;
    float dry_base = 1.75f;
    int dry_allowed_length = 2;
    int dry_penalty_last_n = 256;

    bool operator==(const SamplerSettings&) const = default;
};

struct Candidate {
    int32_t token = 0;
    std::string piece;
    float logit = 0;
    float p_raw = 0;   // softmax of the raw logits (temperature 1)
    float p_final = 0; // probability after the full chain; 0 if filtered
};

struct LabNode {
    int id = 0;
    int parent = -1;    // -1 for the root (the prompt)
    int32_t token = -1; // -1 for the root
    std::string piece;  // token text (the whole prompt for the root)
    float p_final = 0;  // probability of the chosen token at the time of the choice
    int depth = 0;      // number of tokens generated since the root
    std::vector<int> children;
};

struct LabSnapshot {
    std::vector<LabNode> nodes;
    int current = 0;
    std::vector<Candidate> candidates; // top-N at the current node with the given settings
    std::string text;                  // text generated from the root to the current node
    bool at_eog = false;               // the current node is an end of generation
};

class SamplerLab {
public:
    // Tokenizes the prompt (with the chat template when requested), clears the cache, decodes,
    // and creates the root. Clears the previous tree.
    void start(EngineSession& session, const std::string& prompt, bool use_chat_template);
    bool started() const { return !nodes_.empty(); }

    // Top-N candidates at the current node, recomputed from logits kept in memory:
    // no decode, so it is instant when a slider moves.
    std::vector<Candidate> candidates(EngineSession& session, const SamplerSettings& settings, int top_n = 20);

    // Advances one token: samples with settings, or takes `forced` (manual candidate choice).
    // If that token already exists as a child of the current node, reuses that node.
    // Decodes the token and returns the id of the new current node.
    int step(EngineSession& session, const SamplerSettings& settings, std::optional<int32_t> forced = {});

    // Moves to any node in the tree: clears the KV cache and redecodes prompt + path
    // (partial cache removal is not reliable for every architecture).
    void go_to(EngineSession& session, int node_id);

    LabSnapshot snapshot(EngineSession& session, const SamplerSettings& settings, int top_n = 20);

    const std::vector<LabNode>& nodes() const { return nodes_; }
    int current() const { return current_; }
    std::string text_to(int node_id) const;              // generated text root -> node (without the prompt)
    std::vector<int32_t> path_tokens(int node_id) const; // generated tokens root -> node

private:
    std::vector<LabNode> nodes_;
    int current_ = 0;
    std::vector<int32_t> prompt_tokens_;
    std::vector<float> logits_;      // logits of the current node (copy)
    std::vector<int32_t> kv_tokens_; // generated tokens currently in the cache after the prompt
};

} // namespace waffle
