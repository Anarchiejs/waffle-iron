#include "core/comparator.hpp"
#include "core/engine.hpp"
#include "test_helpers.hpp"

#include <catch2/catch_approx.hpp>
#include <cmath>
#include <thread>

using namespace waffle;

TEST_CASE("comparator: two independent errors and cancellation", "[comparator]") {
    Engine engine;
    CompareRequest req;
    req.a.label = "A";
    req.b.label = "B";
    req.a.load.model_path = "absent-A-L2.gguf";
    req.b.load.model_path = "absent-B-L2.gguf";
    auto result = run_compare(engine, nullptr, req);
    CHECK(result.a.label == "A");
    CHECK(result.b.label == "B");
    CHECK_FALSE(result.a.error.empty());
    CHECK_FALSE(result.b.error.empty());
    CHECK_FALSE(engine.submit([](EngineSession& s) { return s.loaded(); }).get());
    CHECK_THROWS_AS(run_compare(engine, nullptr, req, {}, [](float, const auto&) { return false; }), Cancelled);
}

TEST_CASE("comparator: reproducible generation, perplexity, and the failing side", "[comparator][model]") {
    WAFFLE_REQUIRE_MODEL(model);
    Engine engine;
    CompareRequest req;
    req.a.label = "A";
    req.b.label = "B";
    req.prompt = "The capital of France is";
    req.max_tokens = 8;
    req.a.load.model_path = model;
    req.a.load.n_gpu_layers = 0;
    req.a.load.n_ctx = 512;
    req.a.load.n_batch = 64;
    req.a.load.n_ubatch = 32;
    req.a.load.n_threads = 4;
    req.b.load = req.a.load;
    std::string corpus;
    for (int i = 0; i < 8; ++i) corpus += "The cat sat on the mat. A small bird flew over the garden. ";
    engine
        .submit([&](EngineSession& s) {
            s.load(req.a.load);
            CHECK_THROWS_AS(compute_perplexity(s, "short"), Error);
            CHECK_THROWS_AS(compute_perplexity(s, corpus, 15), Error);
            const double ppl = compute_perplexity(s, corpus, 32);
            CHECK(std::isfinite(ppl));
            CHECK(ppl > 1);
            std::string streamed;
            const auto first = generate_side(s, req, [&](const auto& piece) { streamed += piece; });
            const auto second = generate_side(s, req);
            CHECK(first.text == streamed);
            CHECK(first.text == second.text);
            CHECK(first.n_generated == second.n_generated);
            CHECK(first.pp_tps > 0);
            CHECK(first.n_generated <= 8);
            if (first.n_generated > 0) CHECK(first.tg_tps > 0);
            req.temperature = 0.7f;
            const auto random_a = generate_side(s, req), random_b = generate_side(s, req);
            CHECK(random_a.text == random_b.text);
            req.temperature = 0;
        })
        .get();
    // The first side fails and the second generates; also check the callback thread and unload.
    req.a.load.model_path = "absent-L2.gguf";
    const auto caller = std::this_thread::get_id();
    bool callbacks_on_engine = true;
    std::string streamed;
    auto result = run_compare(engine, nullptr, req, [&](int side, const auto& piece) {
        callbacks_on_engine = callbacks_on_engine && std::this_thread::get_id() != caller && side == 1;
        streamed += piece;
    });
    CHECK_FALSE(result.a.error.empty());
    CHECK(result.b.error.empty());
    CHECK(result.b.label == "B");
    CHECK(result.b.text == streamed);
    CHECK(callbacks_on_engine);
    CHECK_FALSE(result.b.model_name.empty());
    CHECK_FALSE(engine.submit([](EngineSession& s) { return s.loaded(); }).get());
}
