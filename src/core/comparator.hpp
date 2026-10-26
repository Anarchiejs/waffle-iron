// Comparateur A/B : même consigne à deux modèles (ou deux quantifications) chargés l'un après l'autre.
#pragma once

#include "core/types.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace banc {

class Engine;
class EngineSession;
class Telemetry;

struct CompareSide {
    std::string label; // "A", "B" ou nom libre
    LoadParams load;
};

struct CompareRequest {
    CompareSide a, b;
    std::string prompt; // consigne utilisateur (passée par apply_chat_template)
    int max_tokens = 256;
    float temperature = 0.0f; // 0 = glouton (comparaison reproductible)
    uint32_t seed = 42;
    std::string perplexity_text; // vide = pas de perplexité
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
    std::string error; // non vide si ce côté a échoué (l'autre côté est quand même rendu)
};

struct CompareResult {
    CompareOutput a, b;
};

// Perplexité sur le modèle chargé : texte découpé en fenêtres de n_ctx tokens (au plus le contexte
// chargé), log-vraisemblance sur la seconde moitié de chaque fenêtre (comme llama-perplexity).
// Lève Error si le texte fait moins de 2 fenêtres de 16 tokens.
double compute_perplexity(EngineSession& session, std::string_view text, uint32_t n_ctx = 512);

// Génère sur le modèle chargé. on_token reçoit chaque morceau de texte (peut être vide).
CompareOutput generate_side(EngineSession& session, const CompareRequest& req,
                            const std::function<void(const std::string&)>& on_token = {});

// Orchestration complète : charge A, génère (+ perplexité), décharge, charge B, idem.
// Bloque le thread appelant ; on_token(side 0|1, morceau) pour l'affichage en direct.
CompareResult run_compare(Engine& engine, Telemetry* telemetry, const CompareRequest& req,
                          const std::function<void(int side, const std::string&)>& on_token = {},
                          const ProgressFn& progress = {});

} // namespace banc
