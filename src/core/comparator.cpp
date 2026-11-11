#include "core/comparator.hpp"
#include "core/engine.hpp"
#include "core/telemetry.hpp"

#include <llama.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <span>

namespace waffle {
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
using Sampler = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
struct Batch {
    llama_batch value;
    explicit Batch(int n) : value(llama_batch_init(n, 0, 1)) {}
    ~Batch() { llama_batch_free(value); }
};
void progress_or_cancel(Engine& engine, const ProgressFn& progress, float f, const std::string& label) {
    if (progress && !progress(f, label)) {
        engine.cancel_all();
        throw Cancelled();
    }
}
} // namespace

double compute_perplexity(EngineSession& session, std::string_view text, uint32_t n_ctx) {
    const auto& info = session.info();
    auto tokens = session.tokenize(text, true);
    // At least two windows of 16 tokens; shrink the window for small corpora.
    const auto window = std::min<size_t>({n_ctx, info.n_ctx, tokens.size() / 2});
    if (window < 16) throw Error("Perplexity needs at least two windows of 16 tokens and a context of 16 tokens");
    if (!info.params.n_batch) throw Error("Batch size must be positive");
    Batch batch(static_cast<int>(std::min<size_t>(window, info.params.n_batch)));
    double nll = 0;
    size_t count = 0;
    const size_t first = window / 2;
    for (size_t start = 0; start + window <= tokens.size(); start += window) {
        session.throw_if_stopped();
        session.clear_memory();
        // As in llama-perplexity, each window starts with BOS when the model requests it.
        if (llama_vocab_get_add_bos(session.vocab())) tokens[start] = llama_vocab_bos(session.vocab());
        for (size_t pos = 0; pos < window; pos += info.params.n_batch) {
            session.throw_if_stopped();
            auto& b = batch.value;
            b.n_tokens = static_cast<int32_t>(std::min<size_t>(info.params.n_batch, window - pos));
            for (int i = 0; i < b.n_tokens; ++i) {
                const auto p = pos + static_cast<size_t>(i);
                b.token[i] = tokens[start + p];
                b.pos[i] = static_cast<llama_pos>(p);
                b.n_seq_id[i] = 1;
                b.seq_id[i][0] = 0;
                b.logits[i] = p >= first && p + 1 < window;
            }
            const auto rc = llama_decode(session.ctx(), b);
            session.throw_if_stopped();
            if (rc == 2) throw Cancelled();
            if (rc != 0) throw Error("Perplexity decode failed (code " + std::to_string(rc) + ")");
            for (int i = 0; i < b.n_tokens; ++i) {
                if (!b.logits[i]) continue;
                const float* logits = llama_get_logits_ith(session.ctx(), i);
                if (!logits) throw Error("Perplexity logits unavailable");
                const double maximum = *std::max_element(logits, logits + session.n_vocab());
                double sum = 0;
                for (int t = 0; t < session.n_vocab(); ++t) sum += std::exp(double(logits[t]) - maximum);
                const auto next = tokens[start + pos + static_cast<size_t>(i) + 1];
                nll += maximum + std::log(sum) - logits[next];
                ++count;
            }
        }
    }
    const double result = std::exp(nll / static_cast<double>(count));
    if (!std::isfinite(result)) throw Error("Non-finite perplexity: invalid logits or corpus");
    return result;
}

CompareOutput generate_side(EngineSession& session, const CompareRequest& req,
                            const std::function<void(const std::string&)>& on_token) {
    const auto& info = session.info();
    if (req.max_tokens < 0 || !std::isfinite(req.temperature) || req.temperature < 0)
        throw Error("Invalid token count or temperature");
    auto prompt = session.tokenize(session.apply_chat_template(req.prompt), true);
    if (prompt.empty() || prompt.size() > info.n_ctx) throw Error("Prompt is empty or longer than the context");
    Sampler sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    if (req.temperature == 0)
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
    else {
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(req.temperature));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(req.seed));
    }
    CompareOutput output;
    output.model_name = info.description;
    output.load_ms = info.load_ms;
    session.clear_memory();
    const auto pp_start = Clock::now();
    session.decode(prompt, 0);
    llama_synchronize(session.ctx());
    output.pp_tps = static_cast<double>(prompt.size()) / seconds(pp_start);
    const auto tg_start = Clock::now();
    const auto budget = std::min<size_t>(static_cast<size_t>(req.max_tokens), info.n_ctx - prompt.size());
    for (size_t i = 0; i < budget; ++i) {
        session.throw_if_stopped();
        const llama_token token = llama_sampler_sample(sampler.get(), session.ctx(), -1);
        llama_sampler_accept(sampler.get(), token);
        if (session.is_eog(token)) break;
        const auto piece = session.token_to_piece(token);
        session.decode(std::span(&token, 1), static_cast<int32_t>(prompt.size() + i));
        llama_synchronize(session.ctx());
        output.text += piece;
        ++output.n_generated;
        // Callback runs on the engine thread: the caller synchronizes its display data.
        if (on_token) on_token(piece);
    }
    output.tg_tps = output.n_generated ? output.n_generated / seconds(tg_start) : 0;
    if (!req.perplexity_text.empty()) output.perplexity = compute_perplexity(session, req.perplexity_text);
    return output;
}

CompareResult run_compare(Engine& engine, Telemetry* telemetry, const CompareRequest& req,
                          const std::function<void(int, const std::string&)>& on_token, const ProgressFn& progress) {
    CompareResult result;
    const CompareSide* sides[] = {&req.a, &req.b};
    CompareOutput* outputs[] = {&result.a, &result.b};
    for (int i = 0; i < 2; ++i) {
        progress_or_cancel(engine, progress, static_cast<float>(i) / 2, "Comparing " + sides[i]->label);
        *outputs[i] = engine
                          .submit([&, i](EngineSession& s) {
                              CompareOutput out;
                              out.label = sides[i]->label;
                              // Even on error, release the weights before trying the other side.
                              struct Unload {
                                  EngineSession& session;
                                  ~Unload() { session.unload(); }
                              } unload{s};
                              std::unique_ptr<VramPeakWatcher> watcher;
                              try {
                                  engine.set_status("Comparing " + sides[i]->label);
                                  if (telemetry) watcher = std::make_unique<VramPeakWatcher>(*telemetry);
                                  s.unload();
                                  const auto& info = s.load(sides[i]->load);
                                  out.model_name = info.description;
                                  out.load_ms = info.load_ms;
                                  out = generate_side(s, req, [&](const std::string& piece) {
                                      if (on_token) on_token(i, piece);
                                  });
                                  out.label = sides[i]->label;
                              } catch (const Cancelled&) {
                                  throw;
                              } catch (const std::exception& e) {
                                  out.error = e.what();
                              }
                              if (watcher) out.vram_peak = watcher->stop();
                              return out;
                          })
                          .get();
    }
    progress_or_cancel(engine, progress, 1, "Comparison finished");
    return result;
}
} // namespace waffle
