#include "cli/cli.hpp"
#include "core/bench.hpp"
#include "core/util.hpp"
#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <sstream>

TEST_CASE("CLI version and unknown command", "[cli]") {
    CHECK(waffle::cli::run({"--version"}) == 0);
    CHECK(waffle::cli::run({"-V"}) == 0);
    CHECK(waffle::cli::run({}) == 0);
    CHECK(waffle::cli::run({"--help"}) == 0);
    CHECK(waffle::cli::run({"bench", "--help"}) == 0);
    CHECK(waffle::cli::run({"not-a-command"}) == 1);
    CHECK(waffle::cli::version_text().find(WAFFLE_VERSION) != std::string::npos);
}

TEST_CASE("CLI scan returns non-zero when a file cannot be read", "[cli]") {
    waffle::test::TempDir temp("cli_scan");
    const auto db = temp.path / "waffle.db";
    const auto bad = temp.path / "broken.gguf";
    {
        std::ofstream file(bad, std::ios::binary);
        file << "GGUF";
    }
    const int code = waffle::cli::run({"--db", waffle::utf8_from_path(db), "scan", waffle::utf8_from_path(temp.path)});
    CHECK(code == 1);
}

TEST_CASE("CLI import and export round trip", "[cli]") {
    waffle::test::TempDir temp("cli_json");
    const auto db = temp.path / "waffle.db";
    waffle::BenchResult result;
    result.model_name = "tiny";
    result.config.load.model_path = temp.path / "tiny.gguf";
    result.config.load.swa_full = false;
    result.config.seed = 9;
    result.config.n_prompt = 4;
    result.config.n_gen = 2;
    result.pp.mean = 1.5;
    result.timestamp = "2020-01-01T00:00:00Z";
    const auto file = temp.path / "campaign.json";
    {
        std::ofstream out(file, std::ios::binary);
        out << waffle::results_to_json({result});
    }
    const auto db_key = waffle::utf8_from_path(db);
    CHECK(waffle::cli::run({"--db", db_key, "import", waffle::utf8_from_path(file), "--name", "round"}) == 0);
    const auto exported = temp.path / "out.json";
    CHECK(waffle::cli::run({"--db", db_key, "export", "1", "--json", "-o", waffle::utf8_from_path(exported)}) == 0);
    std::ifstream in(exported, std::ios::binary);
    std::stringstream buffer;
    buffer << in.rdbuf();
    const auto loaded = waffle::results_from_json(buffer.str());
    REQUIRE(loaded.size() == 1);
    CHECK(loaded[0].model_name == "tiny");
    CHECK_FALSE(loaded[0].config.load.swa_full);
    CHECK(loaded[0].config.seed == 9);
}

TEST_CASE("Config label records the SWA cache", "[cli]") {
    waffle::BenchConfig cfg;
    CHECK(waffle::config_label(cfg).find("swa=full") != std::string::npos);
    cfg.load.swa_full = false;
    cfg.seed = 3;
    const auto label = waffle::config_label(cfg);
    CHECK(label.find("swa=window") != std::string::npos);
    CHECK(label.find("seed=3") != std::string::npos);
}
