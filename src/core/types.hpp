// Types partagés par tous les modules du cœur. Fichier possédé par Claude :
// les lots notent leurs besoins de champs dans leur rapport au lieu de le modifier.
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

namespace banc {

// Erreur « attendue » (fichier illisible, mémoire insuffisante, modèle corrompu…) :
// son message est montré tel quel à l'utilisateur, donc en français et précis.
struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Levée quand un travail est annulé (Engine::cancel_all ou stop_token).
struct Cancelled : Error {
    Cancelled() : Error("Opération annulée") {}
};

// Avancement : fraction 0..1 et libellé court. Retourne false pour demander l'annulation.
using ProgressFn = std::function<bool(float fraction, const std::string& label)>;

constexpr uint64_t KiB = 1024;
constexpr uint64_t MiB = 1024 * KiB;
constexpr uint64_t GiB = 1024 * MiB;

// ---------------------------------------------------------------------------
// Métadonnées GGUF (lot L1)
// ---------------------------------------------------------------------------
struct TensorInfo {
    std::string name;
    std::string type;           // nom ggml : "q4_K", "f16", "f32"…
    std::vector<int64_t> shape; // ne[0..n_dims)
    uint64_t n_elements = 0;
    uint64_t n_bytes = 0;
    int layer = -1; // indice extrait de "blk.<n>." ; -1 = hors couche
};

struct ModelInfo {
    std::filesystem::path path;
    std::string file_name;
    uint64_t file_size = 0;

    std::string architecture;                 // general.architecture
    std::string name;                         // general.name, sinon nom de fichier sans extension
    std::string file_type;                    // "Q4_K_M", "F16"… (general.file_type traduit), "?" si absent
    uint64_t n_params = 0;                    // somme des éléments des tenseurs
    uint32_t context_length = 0;              // <arch>.context_length
    uint32_t n_layer = 0;                     // <arch>.block_count
    uint32_t n_embd = 0;                      // <arch>.embedding_length
    uint32_t n_head = 0;                      // <arch>.attention.head_count
    uint32_t n_head_kv = 0;                   // <arch>.attention.head_count_kv (défaut n_head)
    uint32_t head_dim_k = 0;                  // <arch>.attention.key_length (défaut n_embd / n_head)
    uint32_t head_dim_v = 0;                  // <arch>.attention.value_length (défaut n_embd / n_head)
    uint32_t sliding_window = 0;              // <arch>.attention.sliding_window (0 = aucune)
    // Une valeur par couche, 1 = attention à fenêtre glissante (<arch>.attention.sliding_window_pattern,
    // tableau ou période) ; vide si le modèle n'a pas de SWA.
    std::vector<uint8_t> swa_layers;
    uint32_t head_dim_k_swa = 0;   // <arch>.attention.key_length_swa (défaut head_dim_k)
    uint32_t head_dim_v_swa = 0;   // <arch>.attention.value_length_swa (défaut head_dim_v)
    uint32_t shared_kv_layers = 0; // <arch>.attention.shared_kv_layers : les dernières couches réutilisent le cache
    uint32_t n_expert = 0;                    // <arch>.expert_count (0 = dense)
    uint32_t n_expert_used = 0;               // <arch>.expert_used_count
    std::optional<std::string> chat_template; // tokenizer.chat_template

    std::map<std::string, uint64_t> bytes_by_type;             // type ggml -> octets cumulés
    std::vector<TensorInfo> tensors;                           // vide si lu sans tenseurs
    std::vector<std::pair<std::string, std::string>> metadata; // clé -> valeur lisible (tableaux résumés)
};

// ---------------------------------------------------------------------------
// Chargement d'un modèle par le moteur
// ---------------------------------------------------------------------------
struct LoadParams {
    std::filesystem::path model_path;
    int n_gpu_layers = -1; // -1 = toutes les couches sur GPU
    uint32_t n_ctx = 4096;
    uint32_t n_batch = 2048;
    uint32_t n_ubatch = 512;
    std::string type_k = "f16"; // type ggml du cache K
    std::string type_v = "f16"; // type ggml du cache V (quantifié => flash attention requise)
    int flash_attn = -1;        // -1 auto, 0 non, 1 oui
    int n_threads = 0;          // 0 = automatique
    uint32_t n_seq_max = 1;
    // Cache SWA de la taille du contexte (défaut de llama.cpp). false = seulement la fenêtre + un micro-lot :
    // beaucoup moins de VRAM sur les longs contextes, mais revenir en arrière dans la séquence est impossible.
    bool swa_full = true;

    bool operator==(const LoadParams&) const = default;
};

// ---------------------------------------------------------------------------
// Planificateur de VRAM (lot L1)
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
    uint64_t compute_gpu = 0; // tampons de calcul
    uint64_t compute_cpu = 0;
    uint64_t overhead_gpu = 0;      // contexte CUDA, cuBLAS… (constante calibrée)
    std::vector<std::string> notes; // hypothèses, avertissements (SWA, MoE…)

    uint64_t total_gpu() const { return weights_gpu + kv_cache_gpu + compute_gpu + overhead_gpu; }
    uint64_t total_cpu() const { return weights_cpu + kv_cache_cpu + compute_cpu; }
};

// ---------------------------------------------------------------------------
// Télémétrie GPU (lot L3)
// ---------------------------------------------------------------------------
struct GpuSample {
    double t_seconds = 0; // secondes depuis la création de Telemetry
    uint32_t gpu_util_pct = 0;
    uint32_t mem_util_pct = 0;
    uint64_t vram_used = 0;
    uint64_t vram_total = 0;
    uint32_t temperature_c = 0;
    double power_w = 0;
    uint32_t sm_clock_mhz = 0;
};

// ---------------------------------------------------------------------------
// Banc d'essai (lot L2)
// ---------------------------------------------------------------------------
struct BenchConfig {
    LoadParams load;
    uint32_t n_prompt = 512; // tokens de consigne mesurés (pp)
    uint32_t n_gen = 128;    // tokens générés mesurés (tg)
    int repetitions = 5;
    int warmup = 1; // exécutions jetées avant les mesures
};

struct BenchSample {
    double pp_tps = 0;      // lecture de la consigne, tokens/s
    double tg_tps = 0;      // génération, tokens/s
    double ttft_ms = 0;     // temps jusqu'au premier token (consigne + 1 token)
    uint64_t vram_peak = 0; // octets, 0 si inconnu
};

struct Stats {
    int n = 0;
    double mean = 0, stddev = 0, median = 0, min = 0, max = 0;
    std::vector<int> outliers; // indices des valeurs aberrantes (score z modifié > 3.5)
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
    std::string llama_version; // BANC_LLAMA_VERSION
    std::string timestamp;     // ISO 8601 UTC
};

} // namespace banc
