#include "core/gguf_reader.hpp"
#include "core/util.hpp"
#include "test_helpers.hpp"
#include <gguf.h>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <memory>

namespace {
struct EnvOverride {
    std::string name;
    bool had = false;
    std::string previous;
    EnvOverride(const char* name, const std::string& value) : name(name) {
        if (const char* old = std::getenv(name)) {
            had = true;
            previous = old;
        }
#ifdef _WIN32
        _putenv_s(name, value.c_str());
#else
        setenv(name, value.c_str(), 1);
#endif
    }
    ~EnvOverride() {
#ifdef _WIN32
        _putenv_s(name.c_str(), had ? previous.c_str() : "");
#else
        if (had)
            setenv(name.c_str(), previous.c_str(), 1);
        else
            unsetenv(name.c_str());
#endif
    }
    EnvOverride(const EnvOverride&) = delete;
    EnvOverride& operator=(const EnvOverride&) = delete;
};

bool has_root(const std::vector<std::filesystem::path>& roots, const std::filesystem::path& expected) {
    const auto wanted = expected.lexically_normal();
    for (const auto& root : roots)
        if (root.lexically_normal() == wanted) return true;
    return false;
}

int roots_under(const std::vector<std::filesystem::path>& roots, const std::filesystem::path& base) {
    const auto prefix = base.lexically_normal().generic_string();
    int count = 0;
    for (const auto& root : roots) {
        const auto text = root.lexically_normal().generic_string();
        if (text == prefix || text.starts_with(prefix + "/")) ++count;
    }
    return count;
}

// Windows reads USERPROFILE. Linux and macOS read HOME.
struct ProfileOverride {
#ifdef _WIN32
    EnvOverride env;
    explicit ProfileOverride(const std::filesystem::path& path) : env("USERPROFILE", path.string()) {}
#else
    EnvOverride env;
    explicit ProfileOverride(const std::filesystem::path& path) : env("HOME", path.string()) {}
#endif
};
} // namespace

namespace waffle::test {
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
} // namespace waffle::test

TEST_CASE("GGUF: metadata, tensors, and light read", "[gguf_reader]") {
    waffle::test::TempDir temp("l1_reader");
    const auto path = temp.path / "small.gguf";
    waffle::test::write_l1_model(path);
    REQUIRE(waffle::is_gguf_file(path));
    const auto model = waffle::read_gguf(path);
    CHECK(model.path == path);
    CHECK(model.file_name == "small.gguf");
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
    const auto light = waffle::read_gguf(path, false);
    CHECK(light.tensors.empty());
    CHECK(light.n_params == model.n_params);
    CHECK(light.bytes_by_type == model.bytes_by_type);
    CHECK(waffle::ftype_name(15) == "Q4_K_M");
    CHECK(waffle::ftype_name(999) == "?");
}

TEST_CASE("GGUF: errors and filtered recursive scan", "[gguf_reader]") {
    waffle::test::TempDir temp("l1_scan");
    std::filesystem::create_directories(temp.path / "sub");
    for (const auto* name : {"blob", "model-00001-of-00002.gguf", "model-00002-of-00002.gguf",
                             "model-0001-of-0003.gguf", "model-0002-of-0003.gguf", "mmproj-test.gguf"})
        waffle::test::write_l1_model(temp.path / "sub" / name);
    const auto invalid = temp.path / "invalid.gguf";
    {
        std::ofstream file(invalid);
        file << "incorrect";
    }
    CHECK_FALSE(waffle::is_gguf_file(invalid));
    CHECK_FALSE(waffle::is_gguf_file(temp.path / "absent"));
    try {
        waffle::read_gguf(invalid);
        FAIL("Expected an error");
    } catch (const waffle::Error& e) {
        CHECK(std::string(e.what()).find(invalid.string()) != std::string::npos);
    }
    const auto scanned = waffle::scan_model_dirs({temp.path, temp.path, temp.path / "absent"});
    CHECK_FALSE(scanned.truncated);
    const auto& paths = scanned.files;
    REQUIRE(paths.size() == 3);
    CHECK(paths[0].filename() == "blob");
    CHECK(paths[1].filename() == "model-00001-of-00002.gguf");
    CHECK(paths[2].filename() == "model-0001-of-0003.gguf");
    {
        std::ofstream file(invalid, std::ios::binary);
        file << "GGUF";
    }
    CHECK_THROWS_AS(waffle::read_gguf(invalid), waffle::Error);
}

TEST_CASE("GGUF: Ollama blobs named from manifests, drafts hidden", "[gguf_reader]") {
    waffle::test::TempDir temp("ollama");
    const auto blobs = temp.path / "blobs";
    const auto manifest_dir = temp.path / "manifests" / "registry.ollama.ai" / "library" / "gemma4";
    std::filesystem::create_directories(blobs);
    std::filesystem::create_directories(manifest_dir);
    for (const auto* name : {"sha256-aaa1", "sha256-bbb2", "sha256-ccc3"}) waffle::test::write_l1_model(blobs / name);
    {
        std::ofstream file(manifest_dir / "e4b", std::ios::binary);
        file << R"({"schemaVersion":2,"layers":[)"
             << R"({"mediaType":"application/vnd.ollama.image.model","digest":"sha256:aaa1","size":1},)"
             << R"({"digest":"sha256:bbb2","mediaType":"application/vnd.ollama.image.draft","size":1}]})";
    }
    const auto model = waffle::find_ollama_blob(blobs / "sha256-aaa1");
    REQUIRE(model.has_value());
    CHECK(model->tags == "gemma4:e4b");
    CHECK(model->model);
    const auto draft = waffle::find_ollama_blob(blobs / "sha256-bbb2");
    REQUIRE(draft.has_value());
    CHECK_FALSE(draft->model);
    CHECK_FALSE(waffle::find_ollama_blob(blobs / "sha256-ccc3").has_value()); // orphan: left as-is

    const auto scanned = waffle::scan_model_dirs({blobs});
    CHECK_FALSE(scanned.truncated);
    const auto& paths = scanned.files;
    REQUIRE(paths.size() == 2);
    CHECK(paths[0].filename() == "sha256-aaa1");
    CHECK(paths[1].filename() == "sha256-ccc3");
    CHECK(waffle::read_gguf(paths[0], false).name == "gemma4:e4b (Ollama)");
}

TEST_CASE("GGUF: default model folders", "[gguf_reader]") {
    waffle::test::TempDir temp("roots");
    std::filesystem::create_directories(temp.path / "models");
    std::filesystem::create_directories(temp.path / ".ollama" / "models" / "blobs");
    std::filesystem::create_directories(temp.path / ".lmstudio" / "models");
    std::filesystem::create_directories(temp.path / "custom-ollama" / "blobs");
    ProfileOverride profile(temp.path);
    EnvOverride ollama("OLLAMA_MODELS", (temp.path / "custom-ollama").string());

    const auto candidates = waffle::candidate_model_roots();
    CHECK(has_root(candidates, temp.path / "models"));
    CHECK(has_root(candidates, temp.path / ".ollama" / "models" / "blobs"));
    CHECK(has_root(candidates, temp.path / ".lmstudio" / "models"));
    CHECK(has_root(candidates, temp.path / ".cache" / "lm-studio" / "models"));
    CHECK(has_root(candidates, temp.path / "custom-ollama" / "blobs"));

    const auto roots = waffle::default_model_roots();
    CHECK(has_root(roots, temp.path / "models"));
    CHECK(has_root(roots, temp.path / ".ollama" / "models" / "blobs"));
    CHECK(has_root(roots, temp.path / ".lmstudio" / "models"));
    CHECK(has_root(roots, temp.path / "custom-ollama" / "blobs"));
    CHECK_FALSE(has_root(roots, temp.path / ".cache" / "lm-studio" / "models"));
    CHECK(roots_under(roots, temp.path) == 4);
#ifdef _WIN32
    CHECK(roots.size() == 4);
#endif
}

TEST_CASE("GGUF: OLLAMA_MODELS does not duplicate the default folder", "[gguf_reader]") {
    waffle::test::TempDir temp("roots_dedup");
    std::filesystem::create_directories(temp.path / ".ollama" / "models" / "blobs");
    ProfileOverride profile(temp.path);
    EnvOverride ollama("OLLAMA_MODELS", (temp.path / ".ollama" / "models").string());
    const auto roots = waffle::default_model_roots();
    int blobs = 0;
    for (const auto& root : roots)
        if (root.lexically_normal() == (temp.path / ".ollama" / "models" / "blobs").lexically_normal()) ++blobs;
    CHECK(blobs == 1);
}

TEST_CASE("GGUF: OLLAMA_MODELS without a blobs subfolder", "[gguf_reader]") {
    waffle::test::TempDir temp("roots_flat");
    std::filesystem::create_directories(temp.path / "flat-models");
    ProfileOverride profile(temp.path);
    EnvOverride ollama("OLLAMA_MODELS", (temp.path / "flat-models").string());
    const auto roots = waffle::default_model_roots();
    CHECK(has_root(roots, temp.path / "flat-models"));
    CHECK_FALSE(has_root(roots, temp.path / "flat-models" / "blobs"));
}

#ifndef _WIN32
TEST_CASE("GGUF: HOME is the profile and USERPROFILE is ignored", "[gguf_reader]") {
    waffle::test::TempDir home("roots_home");
    waffle::test::TempDir other("roots_userprofile");
    std::filesystem::create_directories(home.path / "models");
    std::filesystem::create_directories(other.path / "models");
    ProfileOverride profile(home.path);
    EnvOverride windows_only("USERPROFILE", other.path.string());
    const auto roots = waffle::default_model_roots();
    CHECK(has_root(roots, home.path / "models"));
    CHECK_FALSE(has_root(roots, other.path / "models"));
}
#endif

#if !defined(_WIN32) && !defined(__APPLE__)
TEST_CASE("GGUF: Linux Ollama service library is a candidate", "[gguf_reader]") {
    const auto candidates = waffle::candidate_model_roots();
    CHECK(has_root(candidates, "/usr/share/ollama/.ollama/models/blobs"));
}
#endif

TEST_CASE("GGUF: systemd Environment lines yield OLLAMA_MODELS", "[gguf_reader]") {
    CHECK_FALSE(waffle::ollama_models_from_systemd_text("ExecStart=/usr/bin/ollama serve\n").has_value());
    const auto plain = waffle::ollama_models_from_systemd_text(
        "Environment=OLLAMA_HOST=127.0.0.1:11434\nEnvironment=OLLAMA_MODELS=/var/lib/ollama/models\n");
    REQUIRE(plain.has_value());
    CHECK(*plain == "/var/lib/ollama/models");
    const auto quoted =
        waffle::ollama_models_from_systemd_text("Environment=\"PATH=/usr/bin\" \"OLLAMA_MODELS=/opt/models\"\n");
    REQUIRE(quoted.has_value());
    CHECK(*quoted == "/opt/models");
    const auto cleared =
        waffle::ollama_models_from_systemd_text("Environment=OLLAMA_MODELS=/first\nEnvironment=OLLAMA_MODELS=\n");
    REQUIRE(cleared.has_value());
    CHECK(cleared->empty());
}

TEST_CASE("GGUF: optional real model", "[gguf_reader]") {
    WAFFLE_REQUIRE_MODEL(path);
    const auto model = waffle::read_gguf(path);
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

TEST_CASE("GGUF: block_count above the cap is rejected before the layer array", "[gguf_reader]") {
    waffle::test::TempDir temp("huge_layers");
    const auto path = temp.path / "huge.gguf";
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(ggml_init({1024 * 1024, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(gguf_init_empty(), gguf_free);
    REQUIRE(tensors);
    REQUIRE(ctx);
    gguf_set_val_str(ctx.get(), "general.architecture", "llama");
    gguf_set_val_str(ctx.get(), "general.name", "Huge");
    gguf_set_val_u32(ctx.get(), "llama.block_count", waffle::max_model_layers + 1);
    gguf_set_val_u32(ctx.get(), "llama.attention.sliding_window", 16);
    auto* tensor = ggml_new_tensor_1d(tensors.get(), GGML_TYPE_F32, 4);
    ggml_set_name(tensor, "token_embd.weight");
    ggml_set_zero(tensor);
    gguf_add_tensor(ctx.get(), tensor);
    REQUIRE(gguf_write_to_file(ctx.get(), path.string().c_str(), false));
    CHECK_THROWS_AS(waffle::read_gguf(path, false), waffle::Error);

    const auto small = temp.path / "swa.gguf";
    gguf_set_val_u32(ctx.get(), "llama.block_count", 2);
    REQUIRE(gguf_write_to_file(ctx.get(), small.string().c_str(), false));
    const auto model = waffle::read_gguf(small, false);
    CHECK(model.n_layer == 2);
    CHECK(model.swa_layers == std::vector<uint8_t>{1, 1});
}

TEST_CASE("GGUF: scan honors depth, entry cap, and cancel", "[gguf_reader]") {
    waffle::test::TempDir temp("scan_limits");
    std::filesystem::create_directories(temp.path / "a" / "b");
    waffle::test::write_l1_model(temp.path / "top.gguf");
    waffle::test::write_l1_model(temp.path / "a" / "b" / "deep.gguf");

    waffle::ScanLimits shallow;
    shallow.max_depth = 0;
    const auto top_only = waffle::scan_model_dirs({temp.path}, nullptr, shallow);
    CHECK(top_only.truncated);
    REQUIRE(top_only.files.size() == 1);
    CHECK(top_only.files[0].filename() == "top.gguf");

    waffle::ScanLimits few;
    few.max_entries = 0;
    const auto stopped = waffle::scan_model_dirs({temp.path}, nullptr, few);
    CHECK(stopped.truncated);
    CHECK(stopped.files.empty());

    std::atomic<bool> cancel{true};
    const auto none = waffle::scan_model_dirs({temp.path}, &cancel);
    CHECK(none.files.empty());
    CHECK_FALSE(none.truncated);
}

TEST_CASE("GGUF: a manifest larger than the read cap is not loaded whole", "[gguf_reader]") {
    waffle::test::TempDir temp("manifest_cap");
    const auto blobs = temp.path / "blobs";
    const auto manifest_dir = temp.path / "manifests" / "registry.ollama.ai" / "library" / "gemma4";
    std::filesystem::create_directories(blobs);
    std::filesystem::create_directories(manifest_dir);
    waffle::test::write_l1_model(blobs / "sha256-aaa1");
    waffle::test::write_l1_model(blobs / "sha256-bbb2");
    {
        std::ofstream file(manifest_dir / "e4b", std::ios::binary);
        file << R"({"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:aaa1","size":1}]})";
        file << std::string(300 * 1024, ' ');
        file << R"({"mediaType":"application/vnd.ollama.image.model","digest":"sha256:bbb2","size":1})";
    }
    const auto first = waffle::find_ollama_blob(blobs / "sha256-aaa1");
    REQUIRE(first.has_value());
    CHECK(first->model);
    CHECK_FALSE(waffle::find_ollama_blob(blobs / "sha256-bbb2").has_value());
}

TEST_CASE("paths: tilde and environment variables expand", "[util]") {
    const auto plain = waffle::expand_user_path("/tmp/plain");
    CHECK(plain == std::filesystem::path("/tmp/plain"));
#ifndef _WIN32
    waffle::test::TempDir home("expand_home");
    EnvOverride home_env("HOME", home.path.string());
    CHECK(waffle::expand_user_path("~/models") == home.path / "models");
    CHECK(waffle::expand_user_path("~") == home.path);
    CHECK(waffle::expand_user_path("  ~/models  ") == home.path / "models");
    EnvOverride models("WAFFLE_MODELS_TEST", (home.path / "opt").string());
    CHECK(waffle::expand_user_path("$WAFFLE_MODELS_TEST/sub") == home.path / "opt" / "sub");
    CHECK(waffle::expand_user_path("${WAFFLE_MODELS_TEST}/sub") == home.path / "opt" / "sub");
    CHECK(waffle::expand_user_path("/tmp/$WAFFLE_NOT_A_REAL_VAR").string().find("WAFFLE_NOT_A_REAL_VAR") !=
          std::string::npos);
#else
    waffle::test::TempDir home("expand_home");
    EnvOverride profile("USERPROFILE", home.path.string());
    CHECK(waffle::expand_user_path("%USERPROFILE%\\models") == home.path / "models");
    CHECK(waffle::expand_user_path("~\\models") == home.path / "models");
#endif
}
