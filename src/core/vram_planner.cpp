#include "core/vram_planner.hpp"

#include <ggml.h>
#include <algorithm>
#include <charconv>
#include <limits>
#include <sstream>

namespace banc {
namespace {
// Hypothèses initiales à calibrer sur les mesures CUDA réelles.
constexpr uint64_t cuda_overhead = 256 * MiB;
constexpr uint64_t activation_vectors = 4;
constexpr uint64_t float_bytes = 4;
uint64_t add(uint64_t a, uint64_t b) {
    if (b > UINT64_MAX - a) throw Error("Estimation mémoire trop grande");
    return a + b;
}
uint64_t mul(uint64_t a, uint64_t b) {
    if (a && b > UINT64_MAX / a) throw Error("Estimation mémoire trop grande");
    return a * b;
}
ggml_type find_type(std::string_view name) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const auto type = static_cast<ggml_type>(i);
        const auto* text = ggml_type_name(type);
        if (text && name == text && ggml_blck_size(type) > 0 && ggml_type_size(type) > 0) return type;
    }
    throw Error("Type ggml inconnu : " + std::string(name));
}
} // namespace

uint64_t ggml_type_bytes(std::string_view name, uint64_t n) {
    const auto type = find_type(name);
    const auto block = static_cast<uint64_t>(ggml_blck_size(type));
    // Un bloc incomplet occupe un bloc entier ; évite les assertions de ggml_row_size.
    return mul(n / block + (n % block != 0), ggml_type_size(type));
}

const std::vector<std::string>& kv_cache_types() {
    static const std::vector<std::string> types{"f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"};
    return types;
}

VramEstimate estimate_vram(const ModelInfo& model, const VramPlanInput& input) {
    for (const auto* type : {&input.type_k, &input.type_v}) {
        if (std::find(kv_cache_types().begin(), kv_cache_types().end(), *type) == kv_cache_types().end())
            throw Error("Type de cache KV non pris en charge : " + *type);
    }
    if (input.n_gpu_layers < -1) throw Error("Nombre de couches GPU invalide");
    if (!input.n_ctx || !input.n_ubatch || !input.n_batch || !input.n_seq)
        throw Error("Les tailles du contexte, des lots et des séquences doivent être positives");
    if (model.tensors.empty() && model.n_params) throw Error("Relire le modèle avec ses tenseurs pour estimer la VRAM");
    const uint32_t gpu_layers =
        input.n_gpu_layers == -1 ? model.n_layer : std::min(model.n_layer, static_cast<uint32_t>(input.n_gpu_layers));
    const bool output_gpu = input.n_gpu_layers == -1 || static_cast<uint64_t>(input.n_gpu_layers) > model.n_layer;
    const auto on_gpu = [&](uint32_t layer) { return layer >= model.n_layer - gpu_layers; };
    VramEstimate result;
    uint64_t vocab = 0;
    for (const auto& tensor : model.tensors) {
        const bool gpu =
            tensor.layer >= 0 && static_cast<uint32_t>(tensor.layer) < model.n_layer
                ? on_gpu(static_cast<uint32_t>(tensor.layer))
                : output_gpu && (tensor.name.starts_with("output.") || tensor.name.starts_with("output_norm."));
        auto& bytes = gpu ? result.weights_gpu : result.weights_cpu;
        bytes = add(bytes, tensor.n_bytes);
        if (tensor.name == "token_embd.weight" && tensor.shape.size() >= 2)
            vocab = static_cast<uint64_t>(tensor.shape[1]);
    }
    // Sortie liée (pas d'output.weight) : llama.cpp duplique token_embd sur le GPU pour la projection finale.
    const auto has_output = std::any_of(model.tensors.begin(), model.tensors.end(),
                                        [](const TensorInfo& t) { return t.name == "output.weight"; });
    if (output_gpu && !has_output) {
        const auto embd = std::find_if(model.tensors.begin(), model.tensors.end(),
                                       [](const TensorInfo& t) { return t.name == "token_embd.weight"; });
        if (embd != model.tensors.end()) {
            result.weights_gpu = add(result.weights_gpu, embd->n_bytes);
            result.notes.push_back("Sortie liée aux embeddings : token_embd est aussi copié sur le GPU");
        }
    }
    // Le résumé des tableaux conserve leur taille même sans leurs valeurs.
    if (!vocab)
        for (const auto& [key, value] : model.metadata) {
            if (key == "tokenizer.ggml.tokens" && value.starts_with("["))
                std::from_chars(value.data() + 1, value.data() + value.size(), vocab);
        }
    // Cache KV, comme llama.cpp : les couches partagées (shared_kv_layers) n'ont pas de cache ; les couches
    // SWA gardent pad256(min(n_ctx, fenêtre × séquences + ubatch)) cellules (toutes si swa_full), avec leurs propres dimensions de tête.
    const auto pad256 = [](uint64_t n) { return (n + 255) / 256 * 256; };
    const uint32_t n_layer_kv =
        model.shared_kv_layers < model.n_layer ? model.n_layer - model.shared_kv_layers : model.n_layer;
    const uint64_t swa_cells =
        input.swa_full ? mul(input.n_ctx, input.n_seq)
                       : std::min<uint64_t>(input.n_ctx, pad256(add(mul(model.sliding_window, input.n_seq), input.n_ubatch)));
    for (uint32_t layer = 0; layer < n_layer_kv; ++layer) {
        const bool swa = model.sliding_window && layer < model.swa_layers.size() && model.swa_layers[layer];
        const uint32_t dim_k = swa && model.head_dim_k_swa ? model.head_dim_k_swa : model.head_dim_k;
        const uint32_t dim_v = swa && model.head_dim_v_swa ? model.head_dim_v_swa : model.head_dim_v;
        // Les cellules SWA servent toutes les séquences ; le cache plein est multiplié par n_seq.
        const uint64_t cells = swa ? swa_cells : mul(input.n_ctx, input.n_seq);
        const auto row = add(ggml_type_bytes(input.type_k, dim_k), ggml_type_bytes(input.type_v, dim_v));
        const auto bytes = mul(mul(row, model.n_head_kv), cells);
        auto& target = on_gpu(layer) ? result.kv_cache_gpu : result.kv_cache_cpu;
        target = add(target, bytes);
    }
    if (model.shared_kv_layers)
        result.notes.push_back(std::to_string(model.shared_kv_layers) +
                               " couches réutilisent le cache KV des précédentes (aucun cache pour elles)");
    // Activations et logits en f32 par micro-lot. Sans flash attention, matérialisation
    // des scores d'attention : micro-lot × contexte × têtes × sizeof(float).
    const uint64_t ubatch = std::min({input.n_ubatch, input.n_batch, input.n_ctx});
    uint64_t compute = mul(mul(ubatch, add(mul(model.n_embd, activation_vectors), vocab)), float_bytes);
    if (!input.flash_attn) compute = add(compute, mul(mul(mul(ubatch, input.n_ctx), model.n_head), float_bytes));
    const bool gpu_active = gpu_layers > 0 || (output_gpu && result.weights_gpu > 0);
    if (gpu_active) {
        result.compute_gpu = compute;
        // Une partie CPU subsiste pour les couches et les entrées non déportées.
        if (gpu_layers < model.n_layer) result.compute_cpu = compute;
        result.overhead_gpu = cuda_overhead;
    } else
        result.compute_cpu = compute;
    result.notes.push_back("Tampons et surcoût CUDA approximatifs, à calibrer sur des mesures réelles");
    if (model.n_expert)
        result.notes.push_back("MoE : tous les poids des experts sont comptés, même les experts inactifs");
    if (!input.flash_attn && (input.type_v != "f16" && input.type_v != "f32" && input.type_v != "bf16"))
        result.notes.push_back("Le cache V quantifié nécessite flash attention lors du chargement");
    (void)add(add(result.weights_gpu, result.kv_cache_gpu), add(result.compute_gpu, result.overhead_gpu));
    (void)add(add(result.weights_cpu, result.kv_cache_cpu), result.compute_cpu);
    return result;
}

int max_gpu_layers_for(const ModelInfo& model, VramPlanInput input, uint64_t budget) {
    if (model.n_layer >= static_cast<uint32_t>(std::numeric_limits<int>::max() - 1))
        throw Error("Nombre de couches trop grand");
    int lo = 0, hi = static_cast<int>(model.n_layer) + 1;
    while (lo < hi) {
        const int mid = lo + (hi - lo + 1) / 2;
        input.n_gpu_layers = mid;
        if (estimate_vram(model, input).total_gpu() <= budget)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}
} // namespace banc
