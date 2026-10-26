#include "core/sampler_lab.hpp"
#include "core/engine.hpp"

#include <llama.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <span>

namespace banc {
namespace {
using Sampler = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
void validate(const SamplerSettings& s) {
    if (!std::isfinite(s.temperature) || s.temperature < 0 || s.top_k < 0 || !std::isfinite(s.top_p) || s.top_p <= 0 ||
        s.top_p > 1 || !std::isfinite(s.min_p) || s.min_p < 0 || s.min_p > 1 || !std::isfinite(s.repeat_penalty) ||
        s.repeat_penalty <= 0 || !std::isfinite(s.frequency_penalty) || !std::isfinite(s.presence_penalty) ||
        s.penalty_last_n < -1)
        throw Error("Réglages d'échantillonnage invalides");
}
Sampler chain_for(const SamplerSettings& s, int n_vocab, const std::vector<int32_t>& prompt,
                  const std::vector<int32_t>& path, bool draw) {
    validate(s);
    Sampler chain(llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    const int last_n = s.penalty_last_n < 0 ? static_cast<int>(prompt.size() + path.size()) : s.penalty_last_n;
    llama_sampler_chain_add(chain.get(), llama_sampler_init_penalties(n_vocab, last_n, s.repeat_penalty,
                                                                      s.frequency_penalty, s.presence_penalty));
    if (s.top_k > 0) llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(s.top_k));
    if (s.top_p < 1) llama_sampler_chain_add(chain.get(), llama_sampler_init_top_p(s.top_p, 1));
    if (s.min_p > 0) llama_sampler_chain_add(chain.get(), llama_sampler_init_min_p(s.min_p, 1));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(s.temperature));
    if (draw)
        llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(s.seed + static_cast<uint32_t>(path.size())));
    for (int32_t token : prompt) llama_sampler_accept(chain.get(), token);
    for (int32_t token : path) llama_sampler_accept(chain.get(), token);
    return chain;
}
std::vector<llama_token_data> token_data(const std::vector<float>& logits) {
    std::vector<llama_token_data> data;
    data.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) data.push_back({static_cast<llama_token>(i), logits[i], 0});
    return data;
}
// Recalculer explicitement : certains filtres conservent des probabilités antérieures à la température.
void normalize(llama_token_data_array& data) {
    double maximum = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < data.size; ++i) maximum = std::max(maximum, double(data.data[i].logit));
    if (!std::isfinite(maximum)) throw Error("Aucun logit fini pour l'échantillonnage");
    double sum = 0;
    for (size_t i = 0; i < data.size; ++i) sum += std::exp(double(data.data[i].logit) - maximum);
    for (size_t i = 0; i < data.size; ++i)
        data.data[i].p = static_cast<float>(std::exp(double(data.data[i].logit) - maximum) / sum);
}
std::vector<float> copy_logits(EngineSession& s) {
    const float* logits = s.last_logits();
    return {logits, logits + s.n_vocab()};
}
void require_node(const std::vector<LabNode>& nodes, int id) {
    if (id < 0 || static_cast<size_t>(id) >= nodes.size()) throw Error("Nœud du laboratoire introuvable");
}
} // namespace

void SamplerLab::start(EngineSession& session, const std::string& prompt, bool use_chat_template) {
    auto tokens = session.tokenize(use_chat_template ? session.apply_chat_template(prompt) : prompt, true);
    if (tokens.empty() || tokens.size() > session.info().n_ctx)
        throw Error("La consigne est vide ou dépasse le contexte");
    nodes_.clear();
    logits_.clear();
    kv_tokens_.clear();
    prompt_tokens_.clear();
    current_ = 0;
    session.clear_memory();
    session.decode(tokens, 0);
    auto logits = copy_logits(session);
    prompt_tokens_ = std::move(tokens);
    logits_ = std::move(logits);
    nodes_.push_back({0, -1, -1, prompt, 0, 0, {}});
}

std::vector<Candidate> SamplerLab::candidates(EngineSession& session, const SamplerSettings& settings, int top_n) {
    require_node(nodes_, current_);
    session.throw_if_stopped();
    if (logits_.size() != static_cast<size_t>(session.n_vocab())) throw Error("Le modèle du laboratoire a changé");
    auto raw = token_data(logits_);
    llama_token_data_array raw_array{raw.data(), raw.size(), -1, false};
    normalize(raw_array);
    auto filtered = token_data(logits_);
    llama_token_data_array final_array{filtered.data(), filtered.size(), -1, false};
    auto chain = chain_for(settings, session.n_vocab(), prompt_tokens_, path_tokens(current_), false);
    llama_sampler_apply(chain.get(), &final_array);
    normalize(final_array);
    std::vector<float> final(logits_.size(), 0);
    for (size_t i = 0; i < final_array.size; ++i) final[final_array.data[i].id] = final_array.data[i].p;
    const size_t count = static_cast<size_t>(std::clamp(top_n, 0, session.n_vocab()));
    std::partial_sort(raw.begin(), raw.begin() + count, raw.end(),
                      [](const auto& a, const auto& b) { return a.p != b.p ? a.p > b.p : a.id < b.id; });
    std::vector<Candidate> out;
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const auto& t = raw[i];
        out.push_back({t.id, session.token_to_piece(t.id), t.logit, t.p, final[t.id]});
    }
    return out;
}

int SamplerLab::step(EngineSession& session, const SamplerSettings& settings, std::optional<int32_t> forced) {
    require_node(nodes_, current_);
    session.throw_if_stopped();
    if (current_ != 0 && session.is_eog(nodes_[current_].token))
        throw Error("Impossible d'avancer après une fin de génération");
    if (prompt_tokens_.size() + kv_tokens_.size() >= session.info().n_ctx) throw Error("Contexte du laboratoire plein");
    if (logits_.size() != static_cast<size_t>(session.n_vocab())) throw Error("Le modèle du laboratoire a changé");
    if (forced && (*forced < 0 || *forced >= session.n_vocab())) throw Error("Token forcé hors du vocabulaire");
    auto data = token_data(logits_);
    llama_token_data_array distribution{data.data(), data.size(), -1, false};
    auto chain = chain_for(settings, session.n_vocab(), prompt_tokens_, path_tokens(current_), !forced);
    llama_sampler_apply(chain.get(), &distribution);
    normalize(distribution);
    const int32_t token = forced ? *forced : distribution.data[distribution.selected].id;
    float probability = 0;
    for (size_t i = 0; i < distribution.size; ++i)
        if (distribution.data[i].id == token) probability = distribution.data[i].p;
    int child = -1;
    for (int id : nodes_[current_].children)
        if (nodes_[id].token == token) {
            child = id;
            break;
        }
    const auto piece = session.token_to_piece(token);
    session.decode(std::span(&token, 1), static_cast<int32_t>(prompt_tokens_.size() + kv_tokens_.size()));
    auto logits = copy_logits(session);
    if (child < 0) {
        child = static_cast<int>(nodes_.size());
        const int parent = current_, depth = nodes_[parent].depth + 1;
        nodes_.push_back({child, parent, token, piece, probability, depth, {}});
        nodes_[parent].children.push_back(child);
    }
    kv_tokens_.push_back(token);
    logits_ = std::move(logits);
    current_ = child;
    return current_;
}

void SamplerLab::go_to(EngineSession& session, int node_id) {
    require_node(nodes_, node_id);
    session.throw_if_stopped();
    if (node_id == current_) return;
    auto target = path_tokens(node_id);
    try {
        // Toujours repartir d'un cache vide : llama_memory_seq_rm rend « succès » sur les caches SWA
        // (gemma 4) mais laisse des cellules incohérentes, et les logits revisités divergent.
        // Redécoder consigne + chemin coûte peu et reste exact pour toutes les architectures.
        session.clear_memory();
        session.decode(prompt_tokens_, 0, 0, target.empty());
        if (!target.empty()) session.decode(target, static_cast<int32_t>(prompt_tokens_.size()));
        logits_ = copy_logits(session);
        kv_tokens_ = std::move(target);
        current_ = node_id;
    } catch (...) {
        // Une interruption peut laisser un cache partiel ; imposer start avant de le réutiliser.
        nodes_.clear();
        logits_.clear();
        kv_tokens_.clear();
        prompt_tokens_.clear();
        current_ = 0;
        throw;
    }
}

LabSnapshot SamplerLab::snapshot(EngineSession& session, const SamplerSettings& settings, int top_n) {
    require_node(nodes_, current_);
    const bool eog = current_ != 0 && session.is_eog(nodes_[current_].token);
    return {nodes_, current_, eog ? std::vector<Candidate>{} : candidates(session, settings, top_n), text_to(current_),
            eog};
}

std::string SamplerLab::text_to(int node_id) const {
    require_node(nodes_, node_id);
    std::vector<int> path;
    for (int id = node_id; nodes_[id].parent >= 0; id = nodes_[id].parent) path.push_back(id);
    std::string text;
    for (auto it = path.rbegin(); it != path.rend(); ++it) text += nodes_[*it].piece;
    return text;
}

std::vector<int32_t> SamplerLab::path_tokens(int node_id) const {
    require_node(nodes_, node_id);
    std::vector<int32_t> path;
    for (int id = node_id; nodes_[id].parent >= 0; id = nodes_[id].parent) path.push_back(nodes_[id].token);
    std::reverse(path.begin(), path.end());
    return path;
}
} // namespace banc
