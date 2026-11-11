#include "core/gguf_reader.hpp"
#include "core/vram_planner.hpp"
#include "test_helpers.hpp"
#include <catch2/catch_approx.hpp>

namespace waffle::test {
void write_l1_model(const std::filesystem::path& path);
}

TEST_CASE("VRAM: weights, KV, and budget on a generated GGUF", "[vram_planner]") {
    waffle::test::TempDir temp("l1_vram");
    const auto path = temp.path / "model.gguf";
    waffle::test::write_l1_model(path);
    const auto model = waffle::read_gguf(path);
    waffle::VramPlanInput input;
    input.n_ctx = 512;
    input.n_gpu_layers = 0;
    const auto cpu = waffle::estimate_vram(model, input);
    CHECK(cpu.total_gpu() == 0);
    CHECK(cpu.weights_cpu == 20480);
    CHECK(cpu.kv_cache_cpu == 262144);
    input.n_gpu_layers = 1;
    const auto partial = waffle::estimate_vram(model, input);
    CHECK(partial.weights_gpu == 4096);
    CHECK(partial.weights_cpu == 16384);
    CHECK(partial.kv_cache_gpu == cpu.kv_cache_cpu / 2);
    input.n_gpu_layers = 2;
    CHECK(waffle::estimate_vram(model, input).weights_gpu == 8192);
    input.n_gpu_layers = -1;
    const auto full = waffle::estimate_vram(model, input);
    CHECK(full.weights_gpu == 16384);
    CHECK(full.weights_cpu == 4096);
    CHECK(full.kv_cache_cpu == 0);
    CHECK(full.overhead_gpu > 0);
    input.n_ctx *= 2;
    CHECK(waffle::estimate_vram(model, input).kv_cache_gpu == 2 * full.kv_cache_gpu);
    input.n_ctx /= 2;
    input.type_k = input.type_v = "q8_0";
    CHECK(double(waffle::estimate_vram(model, input).kv_cache_gpu) / full.kv_cache_gpu == Catch::Approx(34.0 / 64.0));
    input.type_k = input.type_v = "f16";
    input.n_seq = 2;
    CHECK(waffle::estimate_vram(model, input).kv_cache_gpu == 2 * full.kv_cache_gpu);
    input.n_seq = 1;
    CHECK(waffle::max_gpu_layers_for(model, input, 0) == 0);
    CHECK(waffle::max_gpu_layers_for(model, input, partial.total_gpu()) == 1);
    CHECK(waffle::max_gpu_layers_for(model, input, full.total_gpu()) == 3);
    int previous = 0;
    for (int layers = 0; layers <= 3; ++layers) {
        input.n_gpu_layers = layers;
        const auto budget = waffle::estimate_vram(model, input).total_gpu();
        const auto fitted = waffle::max_gpu_layers_for(model, input, budget);
        CHECK(fitted >= previous);
        CHECK(fitted >= layers);
        previous = fitted;
    }
    input.n_gpu_layers = -1;
    input.flash_attn = false;
    CHECK(waffle::estimate_vram(model, input).compute_gpu > full.compute_gpu);
    CHECK_THROWS_AS(waffle::estimate_vram(waffle::read_gguf(path, false), input), waffle::Error);
}

TEST_CASE("VRAM: SWA, MoE, types, and invalid inputs", "[vram_planner]") {
    waffle::ModelInfo model;
    model.architecture = "llama";
    model.n_layer = 4;
    model.n_head_kv = 2;
    model.head_dim_k = model.head_dim_v = 32;
    waffle::VramPlanInput input;
    input.n_ctx = 512;
    const auto baseline = waffle::estimate_vram(model, input);
    model.sliding_window = 128;
    model.swa_layers = {1, 0, 1, 0};
    // Full SWA cache (llama.cpp default): no savings.
    CHECK(waffle::estimate_vram(model, input).kv_cache_gpu == baseline.kv_cache_gpu);
    // Window only: pad256(128 + 64) = 256 cells instead of 512 on 2 of 4 layers.
    input.swa_full = false;
    input.n_ubatch = 64;
    const auto swa = waffle::estimate_vram(model, input);
    CHECK(swa.kv_cache_gpu == baseline.kv_cache_gpu * 3 / 4);
    // Head dimensions specific to SWA layers.
    model.head_dim_k_swa = model.head_dim_v_swa = 16;
    CHECK(waffle::estimate_vram(model, input).kv_cache_gpu == baseline.kv_cache_gpu * 5 / 8);
    model.head_dim_k_swa = model.head_dim_v_swa = 0;
    // Shared final layers: no cache for them.
    model.shared_kv_layers = 2;
    CHECK(waffle::estimate_vram(model, input).kv_cache_gpu == baseline.kv_cache_gpu * 3 / 8);
    model.shared_kv_layers = 0;
    model.n_expert = 8;
    CHECK(waffle::estimate_vram(model, input).notes.size() > swa.notes.size());
    CHECK(waffle::ggml_type_bytes("q8_0", 32) == 34);
    CHECK(waffle::ggml_type_bytes("q8_0", 33) == 68);
    CHECK(waffle::ggml_type_bytes("f16", 32) == 64);
    CHECK(waffle::ggml_type_bytes("bf16", 32) == 64);
    CHECK(waffle::ggml_type_bytes("q4_K", 256) == 144);
    CHECK_THROWS_AS(waffle::ggml_type_bytes("unknown", 32), waffle::Error);
    CHECK_THROWS_AS(waffle::ggml_type_bytes("f32", UINT64_MAX), waffle::Error);
    input.type_k = "unknown";
    CHECK_THROWS_AS(waffle::estimate_vram(model, input), waffle::Error);
    input.type_k = "f16";
    input.n_ctx = 0;
    CHECK_THROWS_AS(waffle::estimate_vram(model, input), waffle::Error);
}
