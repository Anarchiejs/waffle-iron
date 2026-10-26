#include "store/store.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <fstream>

namespace {

std::string path_key(const std::filesystem::path& path) {
    const auto u8 = path.u8string();
    return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
}

banc::ModelInfo sample_model(const std::filesystem::path& path, uint64_t file_size = 100) {
    banc::ModelInfo info;
    info.path = path;
    info.file_name = path.filename().string();
    info.name = "tiny-llama";
    info.architecture = "llama";
    info.file_type = "F16";
    info.file_size = file_size;
    info.n_params = 1000;
    info.context_length = 256;
    info.n_layer = 2;
    return info;
}

void touch(const std::filesystem::path& path) {
    std::ofstream file(path, std::ios::binary);
    file << "gguf";
}

banc::BenchResult sample_result(const std::filesystem::path& model, const std::string& timestamp, double pp) {
    banc::BenchResult result;
    result.config.load.model_path = model;
    result.config.load.n_gpu_layers = 20;
    result.config.load.n_ctx = 512;
    result.config.load.n_batch = 128;
    result.config.load.n_ubatch = 64;
    result.config.load.type_k = "q8_0";
    result.config.load.type_v = "f16";
    result.config.load.flash_attn = 1;
    result.config.load.n_threads = 4;
    result.config.load.n_seq_max = 1;
    result.config.n_prompt = 32;
    result.config.n_gen = 8;
    result.config.repetitions = 2;
    result.config.warmup = 1;
    result.model_name = "nom'); DROP TABLE models;--";
    result.model_file_type = "F16";
    result.load_ms = 12.5;
    result.vram_peak = 4096;
    result.gpu_name = "cpu";
    result.llama_version = "b11907";
    result.timestamp = timestamp;
    result.pp = banc::Stats{2, pp, 0.5, pp, pp - 1, pp + 1, {1}};
    result.tg = banc::Stats{2, 20, 0.25, 20, 19, 21, {}};
    result.ttft = banc::Stats{2, 3.5, 0.1, 3.5, 3, 4, {0}};
    result.samples.push_back(banc::BenchSample{pp - 1, 19, 3, 4000});
    result.samples.push_back(banc::BenchSample{pp + 1, 21, 4, 4096});
    return result;
}

} // namespace

TEST_CASE("store : chemin par défaut", "[store]") {
    const auto path = banc::Store::default_path();
    CHECK(path.filename() == "banc.db");
    CHECK(path.parent_path().filename() == "BancLocal");
    CHECK(std::filesystem::is_directory(path.parent_path()));
}

TEST_CASE("store : modèles, étiquettes, notes et fichiers absents", "[store]") {
    banc::test::TempDir dir("l3_store_models");
    const auto db = dir.path / "banc.db";
    const auto model_path = dir.path / "tiny.gguf";
    touch(model_path);

    {
        banc::Store store(db);
        store.upsert_model(sample_model(model_path, 10));
        store.set_model_tags(path_key(model_path), {"été", "test", "test", ""});
        store.set_model_notes(path_key(model_path), "note, avec virgule");
        store.upsert_model(sample_model(model_path, 99));

        const auto found = store.find_model(path_key(model_path));
        REQUIRE(found.has_value());
        CHECK(found->file_size == 99);
        CHECK(found->notes == "note, avec virgule");
        CHECK(found->tags == std::vector<std::string>{"test", "été"});
        CHECK(found->name == "tiny-llama");
        REQUIRE_FALSE(found->last_seen.empty());

        REQUIRE(store.list_models().size() == 1);
        REQUIRE_THROWS_AS(store.set_model_notes("absent", "x"), banc::Error);
    }

    {
        banc::Store store(db);
        const auto found = store.find_model(path_key(model_path));
        REQUIRE(found.has_value());
        CHECK(found->notes == "note, avec virgule");
        CHECK(found->file_size == 99);
    }

    std::filesystem::remove(model_path);
    {
        banc::Store store(db);
        store.remove_missing_models();
        CHECK_FALSE(store.find_model(path_key(model_path)).has_value());
        touch(model_path);
        store.upsert_model(sample_model(model_path, 1));
        const auto again = store.find_model(path_key(model_path));
        REQUIRE(again.has_value());
        CHECK(again->tags.empty());
        CHECK(again->notes.empty());
    }

    REQUIRE_THROWS_AS(banc::Store(dir.path), banc::Error);
}

TEST_CASE("store : campagnes, historique et vérifications VRAM", "[store]") {
    banc::test::TempDir dir("l3_store_campaigns");
    banc::Store store(dir.path / "banc.db");
    const auto model = dir.path / "m.gguf";
    const auto other = dir.path / "autre.gguf";

    const auto early = sample_result(model, "2020-01-01T00:00:00Z", 10);
    const auto late = sample_result(model, "2020-01-02T00:00:00Z", 30);
    auto foreign = sample_result(other, "2020-01-03T00:00:00Z", 1);
    foreign.model_name = "autre";

    const int64_t id = store.save_campaign("grille", {late, early, foreign});
    REQUIRE(id > 0);

    const auto campaigns = store.list_campaigns();
    REQUIRE(campaigns.size() == 1);
    CHECK(campaigns[0].id == id);
    CHECK(campaigns[0].n_results == 3);
    CHECK(campaigns[0].name == "grille");
    CHECK(campaigns[0].model_names.find("autre") != std::string::npos);
    CHECK(campaigns[0].model_names.find("DROP TABLE") != std::string::npos);

    const auto loaded = store.load_campaign(id);
    REQUIRE(loaded.size() == 3);
    CHECK(loaded[0].model_name == late.model_name);
    CHECK(loaded[0].config.load == late.config.load);
    CHECK(loaded[0].config.n_prompt == 32);
    CHECK(loaded[0].config.n_gen == 8);
    CHECK(loaded[0].config.repetitions == 2);
    CHECK(loaded[0].config.warmup == 1);
    CHECK(loaded[0].pp.n == 2);
    CHECK(std::abs(loaded[0].pp.mean - 30.0) < 1e-9);
    CHECK(loaded[0].pp.outliers == std::vector<int>{1});
    CHECK(loaded[0].tg.outliers.empty());
    CHECK(loaded[0].ttft.outliers == std::vector<int>{0});
    REQUIRE(loaded[0].samples.size() == 2);
    CHECK(std::abs(loaded[0].samples[1].tg_tps - 21.0) < 1e-9);
    CHECK(loaded[0].samples[1].vram_peak == 4096);
    CHECK(loaded[0].llama_version == "b11907");
    CHECK(loaded[0].timestamp == "2020-01-02T00:00:00Z");

    const auto points = store.history(path_key(model));
    REQUIRE(points.size() == 2);
    CHECK(points[0].timestamp == "2020-01-01T00:00:00Z");
    CHECK(points[1].timestamp == "2020-01-02T00:00:00Z");
    CHECK(std::abs(points[1].pp_mean - 30.0) < 1e-9);
    CHECK(std::abs(points[1].tg_stddev - 0.25) < 1e-9);
    CHECK(points[0].config.find("ngl=20") != std::string::npos);
    CHECK(points[0].llama_version == "b11907");
    CHECK(store.history(path_key(other)).size() == 1);

    store.delete_campaign(id);
    CHECK(store.list_campaigns().empty());
    REQUIRE_THROWS_AS(store.load_campaign(id), banc::Error);
    REQUIRE_THROWS_AS(store.delete_campaign(id), banc::Error);

    banc::VramCheck check;
    check.model_path = path_key(model);
    check.timestamp = "2024-05-01T12:00:00Z";
    check.config = "ngl=20 ctx=512 kv=q8_0 fa=1";
    check.estimated_gpu = 1000;
    check.measured_gpu = 1100;
    store.save_vram_check(check);
    banc::VramCheck earlier = check;
    earlier.timestamp = "2024-04-01T12:00:00Z";
    earlier.estimated_gpu = 900;
    store.save_vram_check(earlier);

    const auto checks = store.vram_checks(path_key(model));
    REQUIRE(checks.size() == 2);
    CHECK(checks[0].timestamp == "2024-04-01T12:00:00Z");
    CHECK(checks[1].measured_gpu == 1100);
    CHECK(store.vram_checks("rien").empty());
}
