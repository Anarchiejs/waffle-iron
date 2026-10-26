#include "core/engine.hpp"
#include "test_helpers.hpp"

#include <catch2/matchers/catch_matchers_string.hpp>

#include <atomic>
#include <chrono>
#include <fstream>
#include <thread>

using namespace banc;

TEST_CASE("Engine : les travaux s'exécutent dans l'ordre sur un seul thread", "[engine]") {
    Engine engine;
    std::thread::id first_thread;
    auto f1 = engine.submit([&](EngineSession&) {
        first_thread = std::this_thread::get_id();
        return 1;
    });
    auto f2 = engine.submit([&](EngineSession&) { return std::this_thread::get_id() == first_thread ? 2 : -1; });
    REQUIRE(f1.get() == 1);
    REQUIRE(f2.get() == 2);
    REQUIRE(first_thread != std::this_thread::get_id());
}

TEST_CASE("Engine : une exception arrive dans le futur", "[engine]") {
    Engine engine;
    auto f = engine.submit([](EngineSession&) -> int { throw Error("boum"); });
    REQUIRE_THROWS_AS(f.get(), Error);
}

TEST_CASE("Engine : cancel_all annule le travail courant et la file", "[engine]") {
    Engine engine;
    std::atomic<bool> started{false};
    auto running = engine.submit([&](EngineSession& s) {
        started = true;
        for (;;) {
            s.throw_if_stopped();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });
    auto queued = engine.submit([](EngineSession&) { return 42; });
    while (!started) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(engine.state() == EngineState::Busy);
    engine.cancel_all();
    REQUIRE_THROWS_AS(running.get(), Cancelled);
    REQUIRE_THROWS_AS(queued.get(), Cancelled);
    // Le moteur reste utilisable après une annulation.
    REQUIRE(engine.submit([](EngineSession&) { return 7; }).get() == 7);
}

TEST_CASE("Engine : erreurs claires sans modèle", "[engine]") {
    Engine engine;
    auto f = engine.submit([](EngineSession& s) {
        LoadParams p;
        p.model_path = "C:/n_existe_pas/modele.gguf";
        s.load(p);
    });
    REQUIRE_THROWS_WITH(f.get(), Catch::Matchers::ContainsSubstring("introuvable"));
    auto g = engine.submit([](EngineSession& s) { return s.tokenize("x", true); });
    REQUIRE_THROWS_AS(g.get(), Error);
}

TEST_CASE("Engine : fichier non GGUF -> Error, pas de plantage", "[engine]") {
    test::TempDir dir("engine_bad");
    auto bad = dir.path / "faux.gguf";
    {
        std::ofstream(bad) << "ceci n'est pas un modèle";
    }
    Engine engine;
    auto f = engine.submit([&](EngineSession& s) {
        LoadParams p;
        p.model_path = bad;
        p.n_gpu_layers = 0;
        s.load(p);
    });
    REQUIRE_THROWS_AS(f.get(), Error);
}

TEST_CASE("Engine : chargement, tokenisation, décodage", "[engine][model]") {
    BANC_REQUIRE_MODEL(model);
    Engine engine;
    LoadParams p;
    p.model_path = model;
    p.n_ctx = 512;
    p.n_batch = 64; // force plusieurs paquets dans decode
    p.n_ubatch = 64;

    auto info = engine.submit([&](EngineSession& s) { return s.load(p); }).get();
    REQUIRE(info.n_vocab > 0);
    REQUIRE(info.n_ctx >= 512);
    REQUIRE(info.load_ms > 0);

    // Recharger avec les mêmes paramètres ne recharge pas.
    auto again = engine.submit([&](EngineSession& s) { return s.load(p).load_ms; }).get();
    REQUIRE(again == info.load_ms);

    auto argmax = engine
                      .submit([&](EngineSession& s) {
                          std::string text;
                          for (int i = 0; i < 20; ++i) text += "Le banc d'essai mesure la vitesse. ";
                          auto tokens = s.tokenize(text, true);
                          REQUIRE(tokens.size() > 64);
                          if (tokens.size() > 200) tokens.resize(200);
                          s.clear_memory();
                          s.decode(tokens, 0);
                          const float* logits = s.last_logits();
                          int best = 0;
                          for (int i = 1; i < s.n_vocab(); ++i)
                              if (logits[i] > logits[best]) best = i;
                          const bool empty_piece = s.token_to_piece(best).empty() && !s.is_eog(best);
                          REQUIRE_FALSE(empty_piece);
                          return best;
                      })
                      .get();
    REQUIRE(argmax >= 0);

    // Contexte plein -> Error explicite.
    auto overflow = engine.submit([&](EngineSession& s) {
        std::vector<llama_token> many(600, 1);
        s.clear_memory();
        s.decode(many, 0);
    });
    REQUIRE_THROWS_AS(overflow.get(), Error);

    engine.submit([](EngineSession& s) { s.unload(); }).get();
    REQUIRE_FALSE(engine.submit([](EngineSession& s) { return s.loaded(); }).get());
}
