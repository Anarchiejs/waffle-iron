// The engine: the ONLY thread that talks to llama.cpp.
// The UI and the CLI submit jobs (functions that receive EngineSession&)
// and get the result through std::future. A loaded model stays loaded between jobs.
#pragma once

#include "core/types.hpp"

#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

struct llama_model;
struct llama_context;
struct llama_vocab;
typedef int32_t llama_token;

namespace waffle {

struct LoadedModel {
    LoadParams params;
    std::string description; // llama_model_desc
    uint64_t size_bytes = 0;
    uint64_t n_params = 0;
    int n_layer = 0;
    int n_ctx_train = 0;
    int n_vocab = 0;
    uint32_t n_ctx = 0; // context actually allocated
    double load_ms = 0;
    bool has_chat_template = false;
};

struct DeviceMemory {
    std::string name; // "" if no GPU
    uint64_t free = 0;
    uint64_t total = 0;
};

// Access to llama.cpp. Exists only on the engine thread: never keep a reference
// or a llama_* pointer outside a job.
class EngineSession {
public:
    ~EngineSession();

    bool loaded() const { return model_ != nullptr; }
    const LoadedModel& info() const; // throws Error if no model is loaded

    // Loads (unloading any previous model). Does not reload when params are identical.
    // Throws Error with a clear message (missing file, insufficient memory…) or Cancelled.
    const LoadedModel& load(const LoadParams& params, const ProgressFn& progress = {});
    void unload();

    llama_model* model() const { return model_; }
    llama_context* ctx() const { return ctx_; }
    const llama_vocab* vocab() const { return vocab_; }
    int n_vocab() const;

    std::vector<llama_token> tokenize(std::string_view text, bool add_special, bool parse_special = true) const;
    std::string token_to_piece(llama_token token, bool special = true) const;
    bool is_eog(llama_token token) const;
    // Formats a user prompt with the model's chat template; raw text if it has no template.
    std::string apply_chat_template(std::string_view user_message) const;

    // Clears the KV cache of every sequence.
    void clear_memory();
    // Decodes tokens starting at position `pos` in sequence `seq`, in batches of n_batch.
    // Only the last token produces logits when want_last_logits. Throws Error / Cancelled.
    void decode(std::span<const llama_token> tokens, int32_t pos, int32_t seq = 0, bool want_last_logits = true);
    // Logits of the last decoded token (n_vocab values).
    const float* last_logits() const;

    // Cancellation of the current job (also checked during llama_decode via abort_callback).
    bool stop_requested() const { return stop_.stop_requested(); }
    void throw_if_stopped() const;

    DeviceMemory device_memory() const; // GPU 0 via ggml_backend_dev_memory

private:
    friend class Engine;
    EngineSession() = default;
    EngineSession(const EngineSession&) = delete;
    EngineSession& operator=(const EngineSession&) = delete;

    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
    std::unique_ptr<LoadedModel> info_;
    std::stop_token stop_;
};

enum class EngineState { Idle, Busy };

class Engine {
public:
    Engine();  // initializes the llama.cpp backend and starts the thread
    ~Engine(); // cancels everything and waits for the thread to finish
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Submits a job; the future carries its value or its exception (Error, Cancelled…).
    template <class F> auto submit(F&& fn) -> std::future<std::invoke_result_t<F&, EngineSession&>> {
        using R = std::invoke_result_t<F&, EngineSession&>;
        // A job cancelled before it starts still runs, but throws Cancelled immediately:
        // its future therefore always receives a value or an exception.
        auto task = std::make_shared<std::packaged_task<R(EngineSession&)>>(
            [f = std::forward<F>(fn)](EngineSession& s) mutable -> R {
                s.throw_if_stopped();
                return f(s);
            });
        auto fut = task->get_future();
        enqueue([task](EngineSession& s) { (*task)(s); });
        return fut;
    }

    // Cancels the current job and clears the queue (waiting futures receive Cancelled).
    void cancel_all();
    EngineState state() const;
    size_t pending() const;
    // Label of the current job, set by set_status from a job ("" when idle).
    std::string status() const;
    // Call from a job to show how far it has got.
    void set_status(std::string text);

    // llama.cpp and engine logs (ggml level: 2 info, 3 warn, 4 error). Thread-safe.
    void set_log_sink(std::function<void(int level, std::string_view text)> sink);

private:
    using Job = std::function<void(EngineSession&)>;
    struct Queued {
        Job job;
        std::stop_source stop;
    };
    void enqueue(Job job);
    void run(std::stop_token thread_stop);

    mutable std::mutex mutex_;
    std::condition_variable_any cv_;
    std::deque<std::shared_ptr<Queued>> queue_;
    std::shared_ptr<Queued> current_;
    std::string status_;
    std::unique_ptr<EngineSession> session_;
    std::jthread thread_; // declared last: starts after the rest
};

// ggml name -> ggml_type enum (for type_k / type_v). Throws Error if unknown.
int ggml_type_from_name(std::string_view name);

} // namespace waffle
