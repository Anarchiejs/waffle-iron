#include "core/gguf_reader.hpp"
#include "test_helpers.hpp"
#include <gguf.h>
#include <fstream>
#include <memory>

namespace banc::test {
void write_l1_model(const std::filesystem::path& path) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(ggml_init({1024 * 1024, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(gguf_init_empty(), gguf_free);
    REQUIRE(tensors);
    REQUIRE(ctx);
    gguf_set_val_str(ctx.get(), "general.architecture", "llama");
    gguf_set_val_str(ctx.get(), "general.name", "Test L1");
    gguf_set_val_u32(ctx.get(), "general.file_type", 1);
    gguf_set_val_u32(ctx.get(), "llama.block_count", 2);
    gguf_set_val_u32(ctx.get(), "llama.context_length", 2048);
    gguf_set_val_u32(ctx.get(), "llama.embedding_length", 64);
    gguf_set_val_u32(ctx.get(), "llama.attention.head_count", 2);
    gguf_set_val_str(ctx.get(), "tokenizer.chat_template", "{{ messages }}");
    const char* tokens[] = {"a", "b", "c"};
    gguf_set_arr_str(ctx.get(), "tokenizer.ggml.tokens", tokens, 3);
    const uint32_t values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    gguf_set_arr_data(ctx.get(), "test.array", GGUF_TYPE_UINT32, values, 9);
    gguf_set_val_bool(ctx.get(), "test.bool", true);
    gguf_set_val_f64(ctx.get(), "test.double", 1.25);
    for (const auto* name : {"blk.0.attn_q.weight", "blk.1.attn_q.weight", "token_embd.weight", "output.weight"}) {
        auto* tensor = ggml_new_tensor_2d(tensors.get(),
                                          std::string(name) == "output.weight" ? GGML_TYPE_F32 : GGML_TYPE_F16, 64, 32);
        ggml_set_name(tensor, name);
        ggml_set_zero(tensor);
        gguf_add_tensor(ctx.get(), tensor);
    }
    REQUIRE(gguf_write_to_file(ctx.get(), path.string().c_str(), false));
}
} // namespace banc::test

TEST_CASE("GGUF : métadonnées, tenseurs et lecture légère", "[gguf_reader]") {
    banc::test::TempDir temp("l1_reader");
    const auto path = temp.path / "petit.gguf";
    banc::test::write_l1_model(path);
    REQUIRE(banc::is_gguf_file(path));
    const auto model = banc::read_gguf(path);
    CHECK(model.path == path);
    CHECK(model.file_name == "petit.gguf");
    CHECK(model.file_size == std::filesystem::file_size(path));
    CHECK(model.name == "Test L1");
    CHECK(model.architecture == "llama");
    CHECK(model.file_type == "F16");
    CHECK(model.n_layer == 2);
    CHECK(model.n_embd == 64);
    CHECK(model.context_length == 2048);
    CHECK(model.n_head_kv == 2);
    CHECK(model.head_dim_k == 32);
    CHECK(model.head_dim_v == 32);
    CHECK(model.sliding_window == 0);
    CHECK(model.n_expert == 0);
    REQUIRE(model.chat_template);
    CHECK(*model.chat_template == "{{ messages }}");
    CHECK(model.n_params == 8192);
    CHECK(model.bytes_by_type.at("f16") == 12288);
    CHECK(model.bytes_by_type.at("f32") == 8192);
    REQUIRE(model.tensors.size() == 4);
    CHECK(model.tensors[0].layer == 0);
    CHECK(model.tensors[1].layer == 1);
    CHECK(model.tensors[2].layer == -1);
    CHECK(model.tensors[0].shape == std::vector<int64_t>{64, 32});
    const std::map<std::string, std::string> meta(model.metadata.begin(), model.metadata.end());
    CHECK(meta.at("tokenizer.ggml.tokens") == "[3 × str]");
    CHECK(meta.at("test.array").find("1, 2, 3, 4, 5, 6, 7, 8, …") != std::string::npos);
    CHECK(meta.at("test.bool") == "true");
    CHECK(meta.at("test.double") == "1.25");
    const auto light = banc::read_gguf(path, false);
    CHECK(light.tensors.empty());
    CHECK(light.n_params == model.n_params);
    CHECK(light.bytes_by_type == model.bytes_by_type);
    CHECK(banc::ftype_name(15) == "Q4_K_M");
    CHECK(banc::ftype_name(999) == "?");
}

TEST_CASE("GGUF : erreurs et scan récursif filtré", "[gguf_reader]") {
    banc::test::TempDir temp("l1_scan");
    std::filesystem::create_directories(temp.path / "sub");
    for (const auto* name : {"blob", "model-00001-of-00002.gguf", "model-00002-of-00002.gguf", "mmproj-test.gguf"})
        banc::test::write_l1_model(temp.path / "sub" / name);
    const auto invalid = temp.path / "invalide.gguf";
    {
        std::ofstream file(invalid);
        file << "incorrect";
    }
    CHECK_FALSE(banc::is_gguf_file(invalid));
    CHECK_FALSE(banc::is_gguf_file(temp.path / "absent"));
    try {
        banc::read_gguf(invalid);
        FAIL("Erreur attendue");
    } catch (const banc::Error& e) {
        CHECK(std::string(e.what()).find(invalid.string()) != std::string::npos);
    }
    const auto paths = banc::scan_model_dirs({temp.path, temp.path, temp.path / "absent"});
    REQUIRE(paths.size() == 2);
    CHECK(paths[0].filename() == "blob");
    CHECK(paths[1].filename() == "model-00001-of-00002.gguf");
    {
        std::ofstream file(invalid, std::ios::binary);
        file << "GGUF";
    }
    CHECK_THROWS_AS(banc::read_gguf(invalid), banc::Error);
}

TEST_CASE("GGUF : modèle réel optionnel", "[gguf_reader]") {
    BANC_REQUIRE_MODEL(path);
    const auto model = banc::read_gguf(path);
    CHECK(model.n_params > 0);
    CHECK(model.n_layer > 0);
    CHECK(model.n_embd > 0);
    CHECK_FALSE(model.architecture.empty());
    CHECK_FALSE(model.tensors.empty());
    uint64_t bytes = 0;
    for (const auto& [type, size] : model.bytes_by_type) bytes += size;
    CHECK(bytes > 0);
    CHECK(bytes <= model.file_size);
}
