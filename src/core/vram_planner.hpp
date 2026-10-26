// Estimation de la mémoire nécessaire avant chargement (poids, cache KV, tampons).
#pragma once

#include "core/types.hpp"

#include <string_view>

namespace banc {

// Octets occupés par n_elements d'un type ggml nommé ("f16", "q8_0", "q4_0", "q4_K"…).
// Lève banc::Error si le type est inconnu.
uint64_t ggml_type_bytes(std::string_view type_name, uint64_t n_elements);

// Types acceptés pour le cache KV, dans l'ordre d'affichage : f32, f16, bf16, q8_0, q5_1, q5_0, q4_1, q4_0, iq4_nl.
const std::vector<std::string>& kv_cache_types();

// Estimation, modèle de llama.cpp :
// - poids : tenseurs "blk.<i>." sur GPU si i >= n_layer - n_gpu_layers ; token_embd reste sur CPU ;
//   output.* va sur GPU si toutes les couches y sont ;
// - cache KV : par couche n_ctx_eff * n_head_kv * (head_dim_k * taille(type_k) + head_dim_v * taille(type_v)),
//   n_ctx_eff = min(n_ctx, sliding_window) pour les couches SWA quand l'architecture en a ;
// - tampons de calcul : fonction de n_ubatch, n_embd, n_vocab, n_ctx (formule documentée dans le .cpp,
//   calibrée sur des mesures réelles) ;
// - overhead GPU : constante (contexte CUDA + cuBLAS), 0 si n_gpu_layers == 0.
VramEstimate estimate_vram(const ModelInfo& model, const VramPlanInput& input);

// Plus grand n_gpu_layers dont total_gpu() tient dans budget_bytes (0 si aucune couche ne tient).
int max_gpu_layers_for(const ModelInfo& model, VramPlanInput input, uint64_t budget_bytes);

} // namespace banc
