// Le moteur : le SEUL thread qui parle à llama.cpp.
// L'interface et la CLI lui soumettent des travaux (fonctions qui reçoivent EngineSession&)
// et récupèrent le résultat par std::future. Un modèle chargé reste chargé entre deux travaux.
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

namespace banc {

struct LoadedModel {
    LoadParams params;
    std::string description; // llama_model_desc
    uint64_t size_bytes = 0;
    uint64_t n_params = 0;
    int n_layer = 0;
    int n_ctx_train = 0;
    int n_vocab = 0;
    uint32_t n_ctx = 0; // contexte réellement alloué
    double load_ms = 0;
    bool has_chat_template = false;
};

struct DeviceMemory {
    std::string name; // "" si aucun GPU
    uint64_t free = 0;
    uint64_t total = 0;
};

// Accès à llama.cpp. N'existe que sur le thread du moteur : ne jamais garder de référence
// ni de pointeur llama_* hors d'un travail.
class EngineSession {
public:
    ~EngineSession();

    bool loaded() const { return model_ != nullptr; }
    const LoadedModel& info() const; // lève Error si aucun modèle

    // Charge (en déchargeant l'éventuel modèle précédent). Ne recharge pas si params identiques.
    // Lève Error avec un message clair (fichier absent, mémoire insuffisante…) ou Cancelled.
    const LoadedModel& load(const LoadParams& params, const ProgressFn& progress = {});
    void unload();

    llama_model* model() const { return model_; }
    llama_context* ctx() const { return ctx_; }
    const llama_vocab* vocab() const { return vocab_; }
    int n_vocab() const;

    std::vector<llama_token> tokenize(std::string_view text, bool add_special, bool parse_special = true) const;
    std::string token_to_piece(llama_token token, bool special = true) const;
    bool is_eog(llama_token token) const;
    // Met une consigne utilisateur au format de conversation du modèle ; texte brut s'il n'a pas de modèle.
    std::string apply_chat_template(std::string_view user_message) const;

    // Vide le cache KV de toutes les séquences.
    void clear_memory();
    // Décode des tokens à partir de la position `pos` dans la séquence `seq`, par paquets de n_batch.
    // Seul le dernier token produit des logits si want_last_logits. Lève Error / Cancelled.
    void decode(std::span<const llama_token> tokens, int32_t pos, int32_t seq = 0, bool want_last_logits = true);
    // Logits du dernier token décodé (n_vocab valeurs).
    const float* last_logits() const;

    // Annulation du travail courant (vérifiée aussi pendant llama_decode via abort_callback).
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
    Engine();  // initialise le backend llama.cpp et démarre le thread
    ~Engine(); // annule tout et attend la fin du thread
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Soumet un travail ; le futur porte sa valeur ou son exception (Error, Cancelled…).
    template <class F> auto submit(F&& fn) -> std::future<std::invoke_result_t<F&, EngineSession&>> {
        using R = std::invoke_result_t<F&, EngineSession&>;
        // Un travail annulé avant de démarrer s'exécute quand même, mais lève Cancelled d'emblée :
        // son futur reçoit donc toujours une valeur ou une exception.
        auto task = std::make_shared<std::packaged_task<R(EngineSession&)>>(
            [f = std::forward<F>(fn)](EngineSession& s) mutable -> R {
                s.throw_if_stopped();
                return f(s);
            });
        auto fut = task->get_future();
        enqueue([task](EngineSession& s) { (*task)(s); });
        return fut;
    }

    // Annule le travail en cours et vide la file (les futurs en attente reçoivent Cancelled).
    void cancel_all();
    EngineState state() const;
    size_t pending() const;
    // Libellé du travail en cours fourni par set_status depuis un travail ("" au repos).
    std::string status() const;
    // À appeler depuis un travail pour afficher où il en est.
    void set_status(std::string text);

    // Journaux de llama.cpp et du moteur (niveau ggml : 2 info, 3 warn, 4 error). Thread-safe.
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
    std::jthread thread_; // déclaré en dernier : démarre après le reste
};

// Nom ggml -> enum ggml_type (pour type_k / type_v). Lève Error si inconnu.
int ggml_type_from_name(std::string_view name);

} // namespace banc
