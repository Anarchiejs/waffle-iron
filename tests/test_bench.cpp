#include "core/bench.hpp"
#include "core/engine.hpp"
#include "test_helpers.hpp"

#include <catch2/catch_approx.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <locale>
#include <set>

using namespace waffle;
using Catch::Approx;

TEST_CASE("bench: known statistics and edge cases", "[bench]") {
    const auto empty = compute_stats({});
    CHECK(empty.n == 0);
    CHECK(empty.mean == 0);
    CHECK(empty.stddev == 0);
    CHECK(empty.median == 0);
    CHECK(empty.min == 0);
    CHECK(empty.max == 0);
    CHECK(empty.outliers.empty());
    const std::array one{7.0};
    const auto single = compute_stats(one);
    CHECK(single.n == 1);
    CHECK(single.mean == 7);
    CHECK(single.stddev == 0);
    CHECK(single.min == 7);
    CHECK(single.max == 7);
    CHECK(single.median == 7);
    const std::array data{4.0, 1.0, 3.0, 2.0};
    const auto s = compute_stats(data);
    CHECK(s.n == 4);
    CHECK(s.mean == Approx(2.5));
    CHECK(s.stddev == Approx(std::sqrt(5.0 / 3)));
    CHECK(s.median == 2.5);
    CHECK(s.min == 1);
    CHECK(s.max == 4);
    CHECK(s.outliers.empty());
    const std::array zero_mad{1.0, 1.0, 1.0, 100.0};
    CHECK(compute_stats(zero_mad).outliers.empty());
    const std::array outlier{10.0, 11.0, 12.0, 13.0, 100.0};
    CHECK(compute_stats(outlier).outliers == std::vector<int>{4});
    const std::array bad{std::numeric_limits<double>::quiet_NaN()};
    CHECK_THROWS_AS(compute_stats(bad), Error);
}

TEST_CASE("bench: Cartesian product and base values", "[bench]") {
    CampaignGrid grid;
    grid.base.load.type_k = "q4_0";
    grid.base.load.type_v = "f16";
    grid.base.n_gen = 7;
    auto configs = expand_grid(grid);
    REQUIRE(configs.size() == 1);
    CHECK(configs[0].load == grid.base.load);
    CHECK(configs[0].n_gen == 7);
    grid.n_gpu_layers = {0, 99};
    grid.n_ctx = {512, 1024};
    grid.kv_type = {"f16", "q8_0"};
    grid.flash_attn = {0, 1};
    grid.n_ubatch = {32, 64};
    configs = expand_grid(grid);
    REQUIRE(configs.size() == 32);
    std::set<std::string> labels;
    for (const auto& c : configs) {
        CHECK(c.load.type_k == c.load.type_v);
        CHECK(c.n_gen == 7);
        labels.insert(config_label(c));
    }
    CHECK(labels.size() == 32);
    grid.kv_type.clear();
    configs = expand_grid(grid);
    REQUIRE(configs.size() == 16);
    CHECK(configs[0].load.type_k == "q4_0");
    CHECK(configs[0].load.type_v == "f16");
}

TEST_CASE("bench: stable exports and escaping", "[bench]") {
    BenchConfig cfg;
    cfg.load.n_gpu_layers = 99;
    cfg.load.type_k = cfg.load.type_v = "q8_0";
    cfg.load.flash_attn = 1;
    CHECK(config_label(cfg) == "ngl=99 ctx=4096 kv=q8_0 fa=on ub=512 swa=full");
    cfg.load.flash_attn = -1;
    CHECK(config_label(cfg).find("fa=auto") != std::string::npos);
    cfg.load.flash_attn = 0;
    CHECK(config_label(cfg).find("fa=off") != std::string::npos);
    BenchResult r;
    r.config = cfg;
    r.model_name = "model, \"trial\"|x";
    r.pp.mean = 12.5;
    r.pp.stddev = 1.25;
    r.samples.resize(2);
    const std::vector<BenchResult> results{r, r};
    const auto csv = to_csv(results), md = to_markdown(results);
    CHECK(csv.starts_with("model,model_file_type,n_gpu_layers,n_ctx,type_k,type_v,"));
    CHECK(csv.find("\"model, \"\"trial\"\"|x\"") != std::string::npos);
    CHECK(csv.find(",12.5,1.25,") != std::string::npos);
    CHECK(std::count(csv.begin(), csv.end(), '\n') == 3);
    CHECK(md.starts_with("| Model | Configuration |"));
    CHECK(md.find("\\|x") != std::string::npos);
    CHECK(md.find("12.50 ± 1.25") != std::string::npos);
    CHECK(std::count(md.begin(), md.end(), '\n') == 4);
    const auto empty_csv = to_csv({}), empty_md = to_markdown({});
    CHECK(std::count(empty_csv.begin(), empty_csv.end(), '\n') == 1);
    CHECK(std::count(empty_md.begin(), empty_md.end(), '\n') == 2);
    r.model_name = "next\nline";
    CHECK(to_csv({r}).find("\"next\nline\"") != std::string::npos);
    CHECK(to_markdown({r}).find("next<br>line") != std::string::npos);
    // A comma-decimal locale must not leak into the CSV.
    struct Comma : std::numpunct<char> {
        char do_decimal_point() const override { return ','; }
    };
    const auto old = std::locale();
    struct Restore {
        std::locale old;
        ~Restore() { std::locale::global(old); }
    } restore{old};
    std::locale::global(std::locale(old, new Comma));
    CHECK(to_csv(results).find(",12.5,1.25,") != std::string::npos);
}

TEST_CASE("bench: failed campaign and cancel without a model", "[bench]") {
    Engine engine;
    CampaignGrid grid;
    grid.base.load.model_path = "missing-model-L2.gguf";
    grid.n_ctx = {512, 1024};
    std::vector<std::string> errors;
    CHECK(run_campaign(engine, nullptr, grid, {}, &errors).empty());
    CHECK(errors.size() == 2);
    CHECK_THROWS_AS(run_campaign(engine, nullptr, grid, [](float, const auto&) { return false; }), Cancelled);
}

TEST_CASE("bench: real measurements and cancel between repetitions", "[bench][model]") {
    WAFFLE_REQUIRE_MODEL(model);
    Engine engine;
    BenchConfig cfg;
    cfg.load.model_path = model;
    cfg.load.n_ctx = 512;
    cfg.load.n_gpu_layers = 0;
    cfg.load.n_batch = 64;
    cfg.load.n_ubatch = 32;
    cfg.load.n_threads = 4;
    cfg.n_prompt = 32;
    cfg.n_gen = 8;
    cfg.repetitions = 2;
    cfg.warmup = 1;
    std::vector<float> fractions;
    auto r = run_bench(engine, nullptr, cfg, [&](float f, const auto&) {
        fractions.push_back(f);
        return true;
    });
    REQUIRE(r.samples.size() == 2);
    for (const auto& s : r.samples) {
        CHECK(s.pp_tps > 0);
        CHECK(std::isfinite(s.pp_tps));
        CHECK(s.tg_tps > 0);
        CHECK(std::isfinite(s.tg_tps));
        CHECK(s.ttft_ms > 0);
        CHECK(s.vram_peak == 0);
    }
    CHECK(r.pp.n == 2);
    CHECK(r.tg.n == 2);
    CHECK(r.ttft.n == 2);
    CHECK(r.load_ms > 0);
    CHECK_FALSE(r.model_name.empty());
    CHECK_FALSE(r.model_file_type.empty());
    CHECK_FALSE(r.llama_version.empty());
    CHECK(r.timestamp.size() == 20);
    CHECK(r.timestamp.back() == 'Z');
    REQUIRE_FALSE(fractions.empty());
    CHECK(fractions.back() == 1);
    CHECK(std::is_sorted(fractions.begin(), fractions.end()));
    cfg.warmup = 0;
    bool reached_second = false;
    CHECK_THROWS_AS(run_bench(engine, nullptr, cfg,
                              [&](float f, const auto&) {
                                  if (f >= 0.5f) {
                                      reached_second = true;
                                      return false;
                                  }
                                  return true;
                              }),
                    Cancelled);
    CHECK(reached_second);
    // Cancelling a campaign must not poison the next job.
    CHECK(engine
              .submit([](EngineSession& s) {
                  s.throw_if_stopped();
                  return 42;
              })
              .get() == 42);
    engine
        .submit([&](EngineSession& s) {
            auto invalid = cfg;
            invalid.n_prompt = 512;
            CHECK_THROWS_AS(run_bench_once(s, invalid), Error);
        })
        .get();
}
