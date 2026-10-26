// Banc d'essai : mesures de vitesse (pp, tg, ttft), statistiques, campagnes en grille, export.
#pragma once

#include "core/types.hpp"

#include <span>
#include <stop_token>
#include <string>
#include <vector>

namespace banc {

class Engine;
class EngineSession;
class Telemetry;

// Moyenne, écart-type (échantillon, n-1), médiane, min, max ; valeurs aberrantes par score z
// modifié (0.6745 * |x - médiane| / MAD > 3.5 ; aucune si MAD == 0). n == 0 -> tout à 0.
Stats compute_stats(std::span<const double> values);

// Une mesure, sur le thread du moteur, modèle déjà chargé avec cfg.load :
// vide le cache, décode n_prompt tokens aléatoires (graine fixe, hors tokens spéciaux) -> pp_tps,
// puis génère n_gen tokens un par un en glouton -> tg_tps ; ttft = consigne + premier token.
// vram_peak reste à 0 (rempli par run_bench).
BenchSample run_bench_once(EngineSession& session, const BenchConfig& cfg);

// Charge le modèle, fait warmup + repetitions mesures, calcule les stats.
// Bloque le thread appelant (pas le thread du moteur) ; soumet les travaux à l'Engine.
// telemetry peut être nullptr (vram_peak = 0). progress peut annuler.
BenchResult run_bench(Engine& engine, Telemetry* telemetry, const BenchConfig& cfg, const ProgressFn& progress = {});

// Grille de réglages : produit cartésien des listes non vides (une liste vide = valeur de `base`).
struct CampaignGrid {
    BenchConfig base;
    std::vector<int> n_gpu_layers;
    std::vector<uint32_t> n_ctx;
    std::vector<std::string> kv_type; // appliqué à type_k ET type_v
    std::vector<int> flash_attn;
    std::vector<uint32_t> n_ubatch;
};
std::vector<BenchConfig> expand_grid(const CampaignGrid& grid);

// Campagne complète ; un BenchResult par configuration (une configuration qui échoue — mémoire
// insuffisante… — est sautée et son erreur ajoutée à `errors` si non nul).
std::vector<BenchResult> run_campaign(Engine& engine, Telemetry* telemetry, const CampaignGrid& grid,
                                      const ProgressFn& progress = {}, std::vector<std::string>* errors = nullptr);

// Exports : une ligne par BenchResult (moyennes ± écart-type), colonnes stables et documentées.
std::string to_csv(const std::vector<BenchResult>& results);
std::string to_markdown(const std::vector<BenchResult>& results);

// Libellé court d'une configuration : "ngl=99 ctx=4096 kv=q8_0 fa=on ub=512".
std::string config_label(const BenchConfig& cfg);

} // namespace banc
