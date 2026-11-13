#include "core/engine.hpp"
#include "core/util.hpp"

#include <llama.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <deque>

namespace waffle {

namespace {

// --- llama.cpp logs ---------------------------------------------------------
// llama_log_set is global: a single shared sink, protected by a mutex.
std::mutex g_log_mutex;
std::function<void(int, std::string_view)> g_log_sink;
std::deque<std::string> g_recent_errors; // recent errors, used to enrich messages

void log_callback(ggml_log_level level, const char* text, void*) {
    if (!text) return;
    std::function<void(int, std::string_view)> sink;
    {
        std::lock_guard lock(g_log_mutex);
        if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
            std::string line(text);
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
            if (!line.empty()) {
                g_recent_errors.push_back(std::move(line));
                if (g_recent_errors.size() > 4) g_recent_errors.pop_front();
            }
        }
        sink = g_log_sink;
    }
    if (sink) sink(static_cast<int>(level), text);
}

std::string take_recent_errors() {
    std::lock_guard lock(g_log_mutex);
    std::string out;
    for (auto& line : g_recent_errors) {
        if (!out.empty()) out += " | ";
        out += line;
    }
    g_recent_errors.clear();
    return out;
}

void clear_recent_errors() {
    std::lock_guard lock(g_log_mutex);
    g_recent_errors.clear();
}

std::once_flag g_backend_once;

int default_threads() {
    unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 4 : static_cast<int>(std::max(1u, hc / 2)); // ~ physical cores
}

struct ProgressBridge {
    const ProgressFn* fn;
    const EngineSession* session;
};

bool progress_trampoline(float progress, void* user) {
    auto* bridge = static_cast<ProgressBridge*>(user);
    if (bridge->session->stop_requested()) return false;
    if (bridge->fn && *bridge->fn) return (*bridge->fn)(progress, "Loading weights");
    return true;
}

bool abort_trampoline(void* data) {
    return static_cast<const EngineSession*>(data)->stop_requested();
}

} // namespace

int ggml_type_from_name(std::string_view name) {
    auto lower = [](std::string_view s) {
        std::string r(s);
        std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) { return std::tolower(c); });
        return r;
    };
    const std::string wanted = lower(name);
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const char* n = ggml_type_name(static_cast<ggml_type>(i));
        if (n && lower(n) == wanted) return i;
    }
    throw Error("Unknown cache type: " + std::string(name));
}

// ---------------------------------------------------------------------------
// EngineSession
// ---------------------------------------------------------------------------

EngineSession::~EngineSession() {
    unload();
}

const LoadedModel& EngineSession::info() const {
    if (!info_) throw Error("No model loaded");
    return *info_;
}

void EngineSession::unload() {
    if (ctx_) llama_free(ctx_);
    if (model_) llama_model_free(model_);
    ctx_ = nullptr;
    model_ = nullptr;
    vocab_ = nullptr;
    info_.reset();
}

const LoadedModel& EngineSession::load(const LoadParams& params, const ProgressFn& progress) {
    if (info_ && info_->params == params) return *info_;
    unload();
    throw_if_stopped();

    std::error_code ec;
    if (!std::filesystem::is_regular_file(params.model_path, ec))
        throw Error("Model file not found: " + narrow_utf8(params.model_path));

    const int type_k = ggml_type_from_name(params.type_k);
    const int type_v = ggml_type_from_name(params.type_v);

    clear_recent_errors();
    const auto t0 = std::chrono::steady_clock::now();

    ProgressBridge bridge{&progress, this};
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = params.n_gpu_layers;
    mp.progress_callback = progress_trampoline;
    mp.progress_callback_user_data = &bridge;

    const std::string model_utf8 = narrow_utf8(params.model_path);
    model_ = llama_model_load_from_file(model_utf8.c_str(), mp);
    if (!model_) {
        throw_if_stopped();
        const std::string details = take_recent_errors();
        throw Error("Cannot load model \"" + narrow_utf8(params.model_path.filename()) +
                    "\": corrupted file, unsupported architecture, or insufficient memory." +
                    (details.empty() ? "" : "\nDetail: " + details));
    }
    vocab_ = llama_model_get_vocab(model_);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = params.n_ctx;
    cp.n_batch = params.n_batch;
    cp.n_ubatch = std::min(params.n_ubatch, params.n_batch);
    cp.n_seq_max = std::max(1u, params.n_seq_max);
    cp.type_k = static_cast<ggml_type>(type_k);
    cp.type_v = static_cast<ggml_type>(type_v);
    cp.flash_attn_type = params.flash_attn < 0    ? LLAMA_FLASH_ATTN_TYPE_AUTO
                         : params.flash_attn == 0 ? LLAMA_FLASH_ATTN_TYPE_DISABLED
                                                  : LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads = params.n_threads > 0 ? params.n_threads : default_threads();
    cp.n_threads_batch = cp.n_threads;
    cp.abort_callback = abort_trampoline;
    cp.abort_callback_data = this;
    cp.no_perf = false;
    cp.swa_full = params.swa_full;

    ctx_ = llama_init_from_model(model_, cp);
    if (!ctx_) {
        const std::string details = take_recent_errors();
        unload();
        throw Error("Cannot create the context (" + std::to_string(params.n_ctx) + " tokens, cache " + params.type_k +
                    "/" + params.type_v +
                    "): insufficient memory or incompatible setting (a quantized V cache requires flash attention)." +
                    (details.empty() ? "" : "\nDetail: " + details));
    }

    auto info = std::make_unique<LoadedModel>();
    info->params = params;
    char desc[256] = {};
    llama_model_desc(model_, desc, sizeof desc);
    info->description = desc;
    info->size_bytes = llama_model_size(model_);
    info->n_params = llama_model_n_params(model_);
    info->n_layer = llama_model_n_layer(model_);
    info->n_ctx_train = llama_model_n_ctx_train(model_);
    info->n_vocab = llama_vocab_n_tokens(vocab_);
    info->n_ctx = llama_n_ctx(ctx_);
    info->has_chat_template = llama_model_chat_template(model_, nullptr) != nullptr;
    info->load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    info_ = std::move(info);
    return *info_;
}

int EngineSession::n_vocab() const {
    if (!vocab_) throw Error("No model loaded");
    return llama_vocab_n_tokens(vocab_);
}

std::vector<llama_token> EngineSession::tokenize(std::string_view text, bool add_special, bool parse_special) const {
    if (!vocab_) throw Error("No model loaded");
    std::vector<llama_token> tokens(text.size() + 16);
    int32_t n = llama_tokenize(vocab_, text.data(), static_cast<int32_t>(text.size()), tokens.data(),
                               static_cast<int32_t>(tokens.size()), add_special, parse_special);
    if (n < 0) {
        tokens.resize(static_cast<size_t>(-n));
        n = llama_tokenize(vocab_, text.data(), static_cast<int32_t>(text.size()), tokens.data(),
                           static_cast<int32_t>(tokens.size()), add_special, parse_special);
    }
    if (n < 0) throw Error("Tokenization failed");
    tokens.resize(static_cast<size_t>(n));
    return tokens;
}

std::string EngineSession::token_to_piece(llama_token token, bool special) const {
    if (!vocab_) throw Error("No model loaded");
    std::string piece(64, '\0');
    int32_t n = llama_token_to_piece(vocab_, token, piece.data(), static_cast<int32_t>(piece.size()), 0, special);
    if (n < 0) {
        piece.resize(static_cast<size_t>(-n));
        n = llama_token_to_piece(vocab_, token, piece.data(), static_cast<int32_t>(piece.size()), 0, special);
    }
    piece.resize(static_cast<size_t>(std::max(n, 0)));
    return piece;
}

bool EngineSession::is_eog(llama_token token) const {
    return vocab_ && llama_vocab_is_eog(vocab_, token);
}

std::string EngineSession::apply_chat_template(std::string_view user_message) const {
    if (!model_) throw Error("No model loaded");
    const char* tmpl = llama_model_chat_template(model_, nullptr);
    if (!tmpl) return {user_message.begin(), user_message.end()};
    std::string content(user_message);
    llama_chat_message msg{"user", content.c_str()};
    std::vector<char> buf(content.size() * 2 + 256);
    int32_t n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), static_cast<int32_t>(buf.size()));
    if (n > static_cast<int32_t>(buf.size())) {
        buf.resize(static_cast<size_t>(n));
        n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), static_cast<int32_t>(buf.size()));
    }
    // Jinja template not recognized by the non-Jinja build: fall back to the raw text.
    if (n < 0) return content;
    return {buf.data(), static_cast<size_t>(n)};
}

void EngineSession::clear_memory() {
    if (!ctx_) throw Error("No model loaded");
    llama_memory_clear(llama_get_memory(ctx_), true);
}

void EngineSession::decode(std::span<const llama_token> tokens, int32_t pos, int32_t seq, bool want_last_logits) {
    if (!ctx_) throw Error("No model loaded");
    if (tokens.empty()) return;
    const auto n_batch = static_cast<size_t>(info_->params.n_batch);
    llama_batch batch = llama_batch_init(static_cast<int32_t>(std::min(n_batch, tokens.size())), 0, 1);
    struct BatchGuard {
        llama_batch& b;
        ~BatchGuard() { llama_batch_free(b); }
    } guard{batch};

    for (size_t start = 0; start < tokens.size(); start += n_batch) {
        throw_if_stopped();
        const size_t count = std::min(n_batch, tokens.size() - start);
        batch.n_tokens = static_cast<int32_t>(count);
        for (size_t i = 0; i < count; ++i) {
            batch.token[i] = tokens[start + i];
            batch.pos[i] = pos + static_cast<int32_t>(start + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = seq;
            batch.logits[i] = (want_last_logits && start + i == tokens.size() - 1) ? 1 : 0;
        }
        const int32_t rc = llama_decode(ctx_, batch);
        if (rc == 2) throw Cancelled();
        if (rc == 1)
            throw Error("Context is full: no room left in the KV cache (" + std::to_string(info_->n_ctx) +
                        " tokens). Increase the context size.");
        if (rc != 0) {
            const std::string details = take_recent_errors();
            throw Error("llama_decode failed (code " + std::to_string(rc) + ")" +
                        (details.empty() ? "" : ": " + details));
        }
    }
}

const float* EngineSession::last_logits() const {
    if (!ctx_) throw Error("No model loaded");
    const float* logits = llama_get_logits_ith(ctx_, -1);
    if (!logits) throw Error("No logits available (nothing has been decoded)");
    return logits;
}

void EngineSession::throw_if_stopped() const {
    if (stop_.stop_requested()) throw Cancelled();
}

DeviceMemory EngineSession::device_memory() const {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        DeviceMemory m;
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        m.name = ggml_backend_dev_description(dev);
        m.free = free;
        m.total = total;
        return m;
    }
    return {};
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

Engine::Engine() : session_(new EngineSession()) {
    std::call_once(g_backend_once, [] {
        llama_log_set(log_callback, nullptr);
        llama_backend_init();
    });
    thread_ = std::jthread([this](const std::stop_token& st) { run(st); });
}

Engine::~Engine() {
    cancel_all();
    thread_.request_stop();
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void Engine::enqueue(Job job) {
    auto q = std::make_shared<Queued>();
    q->job = std::move(job);
    {
        std::lock_guard lock(mutex_);
        queue_.push_back(std::move(q));
    }
    cv_.notify_one();
}

void Engine::cancel_all() {
    std::lock_guard lock(mutex_);
    if (current_) current_->stop.request_stop();
    for (auto& q : queue_) q->stop.request_stop();
}

EngineState Engine::state() const {
    std::lock_guard lock(mutex_);
    return (current_ || !queue_.empty()) ? EngineState::Busy : EngineState::Idle;
}

size_t Engine::pending() const {
    std::lock_guard lock(mutex_);
    return queue_.size();
}

std::string Engine::status() const {
    std::lock_guard lock(mutex_);
    return status_;
}

void Engine::set_status(std::string text) {
    std::lock_guard lock(mutex_);
    status_ = std::move(text);
}

void Engine::set_log_sink(std::function<void(int, std::string_view)> sink) {
    std::lock_guard lock(g_log_mutex);
    g_log_sink = std::move(sink);
}

void Engine::run(const std::stop_token& thread_stop) {
    for (;;) {
        std::shared_ptr<Queued> item;
        {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, thread_stop, [this] { return !queue_.empty(); });
            // On shutdown, still drain the queue: each future receives Cancelled.
            if (queue_.empty()) break;
            item = std::move(queue_.front());
            queue_.pop_front();
            if (thread_stop.stop_requested()) item->stop.request_stop();
            current_ = item;
        }
        session_->stop_ = item->stop.get_token();
        item->job(*session_); // packaged_task: exceptions go into the future
        session_->stop_ = {};
        {
            std::lock_guard lock(mutex_);
            current_.reset();
            status_.clear();
        }
    }
    // llama.cpp objects are freed on the thread that created them.
    session_->unload();
}

} // namespace waffle
