#include "core/bench.hpp"
#include "core/json.hpp"
#include "core/text_diff.hpp"
#include "core/util.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("JSON round trip keeps campaign fields", "[json]") {
    waffle::BenchResult result;
    result.model_name = "quoted \"name\"";
    result.model_file_type = "Q4_K_M";
    result.config.load.model_path = "models/one.gguf";
    result.config.load.n_ctx = 4096;
    result.config.load.swa_full = false;
    result.config.n_prompt = 32;
    result.config.n_gen = 8;
    result.config.seed = 7;
    result.pp.n = 2;
    result.pp.mean = 12.5;
    result.pp.stddev = 0.5;
    result.pp.outliers = {1};
    result.samples.push_back({10.0, 2.0, 3.0, 100});
    result.timestamp = "2020-01-02T00:00:00Z";
    result.llama_version = "abc";

    const auto loaded = waffle::results_from_json(waffle::results_to_json({result}));
    REQUIRE(loaded.size() == 1);
    CHECK(loaded[0].model_name == result.model_name);
    CHECK(loaded[0].config.load.swa_full == false);
    CHECK(loaded[0].config.seed == 7);
    CHECK(loaded[0].config.n_prompt == 32);
    CHECK(loaded[0].pp.outliers == std::vector<int>{1});
    CHECK(loaded[0].samples.size() == 1);
    CHECK(loaded[0].samples[0].vram_peak == 100);
    CHECK(loaded[0].timestamp == result.timestamp);
}

TEST_CASE("JSON rejects trailing data", "[json]") {
    CHECK_THROWS_AS(waffle::parse_json("{\"a\": 1} trailing"), waffle::Error);
}

TEST_CASE("Word diff marks an insertion", "[json]") {
    const auto spans = waffle::word_diff("alpha beta", "alpha gamma beta");
    REQUIRE(spans.size() == 3);
    CHECK(spans[0].op == waffle::DiffOp::Equal);
    CHECK(spans[1].op == waffle::DiffOp::Insert);
    CHECK(spans[1].text.find("gamma") != std::string::npos);
    CHECK(spans[2].op == waffle::DiffOp::Equal);
}

TEST_CASE("Byte formatting stays stable", "[json]") {
    CHECK(waffle::format_bytes(8) == "8 B");
    CHECK(waffle::format_bytes(512ull * waffle::MiB) == "512 MiB");
    CHECK(waffle::format_bytes(waffle::GiB + waffle::GiB / 2) == "1.50 GiB");
}
