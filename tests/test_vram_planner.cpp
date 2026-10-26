#include "core/gguf_reader.hpp"
#include "core/vram_planner.hpp"
#include "test_helpers.hpp"
#include <catch2/catch_approx.hpp>

namespace banc::test {
void write_l1_model(const std::filesystem::path& path);
}

TEST_CASE("VRAM : poids, KV et budget sur GGUF généré", "[vram_planner]") {
    banc::test::TempDir temp("l1_vram");
    const auto path = temp.path / "model.gguf";
    banc::test::write_l1_model(path);
    const auto model = banc::read_gguf(path);
    banc::VramPlanInput input;
    input.n_ctx = 512;
    input.n_gpu_layers = 0;
    const auto cpu = banc::estimate_vram(model, input);
    CHECK(cpu.total_gpu() == 0);
    CHECK(cpu.weights_cpu == 20480);
    CHECK(cpu.kv_cache_cpu == 262144);
    input.n_gpu_layers = 1;
    const auto partial = banc::estimate_vram(model, input);
    CHECK(partial.weights_gpu == 4096);
    CHECK(partial.weights_cpu == 16384);
    CHECK(partial.kv_cache_gpu == cpu.kv_cache_cpu / 2);
    input.n_gpu_layers = 2;
    CHECK(banc::estimate_vram(model, input).weights_gpu == 8192);
    input.n_gpu_layers = -1;
    const auto full = banc::estimate_vram(model, input);
    CHECK(full.weights_gpu == 16384);
    CHECK(full.weights_cpu == 4096);
    CHECK(full.kv_cache_cpu == 0);
    CHECK(full.overhead_gpu > 0);
    input.n_ctx *= 2;
    CHECK(banc::estimate_vram(model, input).kv_cache_gpu == 2 * full.kv_cache_gpu);
    input.n_ctx /= 2;
    input.type_k = input.type_v = "q8_0";
    CHECK(double(banc::estimate_vram(model, input).kv_cache_gpu) / full.kv_cache_gpu == Catch::Approx(34.0 / 64.0));
    input.type_k = input.type_v = "f16";
    input.n_seq = 2;
    CHECK(banc::estimate_vram(model, input).kv_cache_gpu == 2 * full.kv_cache_gpu);
    input.n_seq = 1;
    CHECK(banc::max_gpu_layers_for(model, input, 0) == 0);
    CHECK(banc::max_gpu_layers_for(model, input, partial.total_gpu()) == 1);
    CHECK(banc::max_gpu_layers_for(model, input, full.total_gpu()) == 3);
    int previous = 0;
    for (int layers = 0; layers <= 3; ++layers) {
        input.n_gpu_layers = layers;
        const auto budget = banc::estimate_vram(model, input).total_gpu();
        const auto fitted = banc::max_gpu_layers_for(model, input, budget);
        CHECK(fitted >= previous);
        CHECK(fitted >= layers);
        previous = fitted;
    }
    input.n_gpu_layers = -1;
    input.flash_attn = false;
    CHECK(banc::estimate_vram(model, input).compute_gpu > full.compute_gpu);
    CHECK_THROWS_AS(banc::estimate_vram(banc::read_gguf(path, false), input), banc::Error);
}

TEST_CASE("VRAM : SWA, MoE, types et entrées invalides", "[vram_planner]") {
    banc::ModelInfo model;
    model.architecture = "llama";
    model.n_layer = 4;
    model.n_head_kv = 2;
    model.head_dim_k = model.head_dim_v = 32;
    banc::VramPlanInput input;
    input.n_ctx = 512;
    const auto baseline = banc::estimate_vram(model, input);
    model.sliding_window = 128;
    model.swa_layers = {1, 0, 1, 0};
    // Cache SWA complet (défaut de llama.cpp) : aucune économie.
    CHECK(banc::estimate_vram(model, input).kv_cache_gpu == baseline.kv_cache_gpu);
    // Fenêtre seule : pad256(128 + 64) = 256 cellules au lieu de 512 sur 2 couches sur 4.
    input.swa_full = false;
    input.n_ubatch = 64;
    const auto swa = banc::estimate_vram(model, input);
    CHECK(swa.kv_cache_gpu == baseline.kv_cache_gpu * 3 / 4);
    // Dimensions de tête propres aux couches SWA.
    model.head_dim_k_swa = model.head_dim_v_swa = 16;
    CHECK(banc::estimate_vram(model, input).kv_cache_gpu == baseline.kv_cache_gpu * 5 / 8);
    model.head_dim_k_swa = model.head_dim_v_swa = 0;
    // Couches finales partagées : pas de cache pour elles.
    model.shared_kv_layers = 2;
    CHECK(banc::estimate_vram(model, input).kv_cache_gpu == baseline.kv_cache_gpu * 3 / 8);
    model.shared_kv_layers = 0;
    model.n_expert = 8;
    CHECK(banc::estimate_vram(model, input).notes.size() > swa.notes.size());
    CHECK(banc::ggml_type_bytes("q8_0", 32) == 34);
    CHECK(banc::ggml_type_bytes("q8_0", 33) == 68);
    CHECK(banc::ggml_type_bytes("f16", 32) == 64);
    CHECK(banc::ggml_type_bytes("bf16", 32) == 64);
    CHECK(banc::ggml_type_bytes("q4_K", 256) == 144);
    CHECK_THROWS_AS(banc::ggml_type_bytes("inconnu", 32), banc::Error);
    CHECK_THROWS_AS(banc::ggml_type_bytes("f32", UINT64_MAX), banc::Error);
    input.type_k = "inconnu";
    CHECK_THROWS_AS(banc::estimate_vram(model, input), banc::Error);
    input.type_k = "f16";
    input.n_ctx = 0;
    CHECK_THROWS_AS(banc::estimate_vram(model, input), banc::Error);
}
