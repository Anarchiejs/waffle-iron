#include "core/sampler_lab.hpp"
#include "core/engine.hpp"
#include "test_helpers.hpp"

#include <llama.h>
#include <catch2/catch_approx.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

using namespace banc;
using Catch::Approx;

TEST_CASE("sampler_lab : état initial et indices invalides", "[sampler_lab]") {
    SamplerLab lab;
    CHECK_FALSE(lab.started());
    CHECK(lab.nodes().empty());
    CHECK_THROWS_AS(lab.path_tokens(0), Error);
    CHECK_THROWS_AS(lab.text_to(-1), Error);
}

TEST_CASE("sampler_lab : probabilités, branches, restauration et EOG", "[sampler_lab][model]") {
    BANC_REQUIRE_MODEL(model);
    Engine engine;
    engine
        .submit([&](EngineSession& s) {
            LoadParams p;
            p.model_path = model;
            p.n_gpu_layers = 0;
            p.n_ctx = 512;
            p.n_batch = 64;
            p.n_ubatch = 32;
            p.n_threads = 4;
            s.load(p);
            SamplerLab lab;
            SamplerSettings settings;
            settings.top_k = 8;
            settings.top_p = 1;
            settings.min_p = 0;
            lab.start(s, "The capital of France is", false);
            REQUIRE(lab.started());
            CHECK(lab.current() == 0);
            CHECK(lab.text_to(0).empty());
            CHECK(lab.path_tokens(0).empty());
            const auto candidates = lab.candidates(s, settings, 20);
            REQUIRE(candidates.size() == 20);
            double final_sum = 0;
            for (size_t i = 0; i < candidates.size(); ++i) {
                if (i) CHECK(candidates[i - 1].p_raw >= candidates[i].p_raw);
                CHECK(std::isfinite(candidates[i].logit));
                CHECK(candidates[i].p_raw > 0);
                CHECK(candidates[i].p_final >= 0);
                final_sum += candidates[i].p_final;
            }
            CHECK(final_sum == Approx(1).margin(1e-5));
            auto filtered = settings;
            filtered.top_p = 0.4f;
            filtered.min_p = 0.2f;
            const auto narrow = lab.candidates(s, filtered, 20);
            double narrow_sum = 0;
            for (size_t i = 0; i < narrow.size(); ++i) {
                CHECK(narrow[i].p_raw == candidates[i].p_raw);
                narrow_sum += narrow[i].p_final;
            }
            CHECK(narrow_sum == Approx(1).margin(1e-5));
            CHECK(std::count_if(narrow.begin(), narrow.end(), [](auto& c) { return c.p_final > 0; }) <= 8);
            auto one = settings;
            one.top_k = 1;
            const auto greedy = lab.candidates(s, one, 20);
            CHECK(std::count_if(greedy.begin(), greedy.end(), [](auto& c) { return c.p_final > 0; }) == 1);
            CHECK(greedy.front().p_final == Approx(1));
            auto cold = settings;
            cold.temperature = 0;
            const auto cold_candidates = lab.candidates(s, cold, 20);
            CHECK(std::count_if(cold_candidates.begin(), cold_candidates.end(),
                                [](auto& c) { return c.p_final > 0; }) == 1);
            CHECK(lab.candidates(s, settings, 0).empty());
            auto invalid = settings;
            invalid.repeat_penalty = 0;
            CHECK_THROWS_AS(lab.candidates(s, invalid), Error);
            CHECK_THROWS_AS(lab.step(s, settings, -1), Error);
            CHECK_THROWS_AS(lab.go_to(s, 1234), Error);
            std::vector<int32_t> non_eog;
            for (auto& c : candidates)
                if (!s.is_eog(c.token)) non_eog.push_back(c.token);
            REQUIRE(non_eog.size() >= 2);
            const auto a = lab.step(s, settings, non_eog[0]);
            const auto a_logits = lab.candidates(s, settings, 5);
            const auto piece_a = s.token_to_piece(non_eog[0]);
            CHECK(lab.text_to(a) == piece_a);
            CHECK(lab.path_tokens(a) == std::vector<int32_t>{non_eog[0]});
            lab.go_to(s, 0);
            const auto restored_root = lab.candidates(s, settings, 20);
            REQUIRE(restored_root.size() == candidates.size());
            for (size_t i = 0; i < candidates.size(); ++i) {
                CHECK(restored_root[i].token == candidates[i].token);
                CHECK(restored_root[i].logit == Approx(candidates[i].logit).margin(1e-3));
            }
            const auto b = lab.step(s, settings, non_eog[1]);
            CHECK(a != b);
            CHECK(lab.nodes()[0].children.size() == 2);
            CHECK(lab.text_to(b) == s.token_to_piece(non_eog[1]));
            lab.go_to(s, a);
            const auto restored = lab.candidates(s, settings, 5);
            REQUIRE(restored.size() == a_logits.size());
            for (size_t i = 0; i < restored.size(); ++i) {
                CHECK(restored[i].token == a_logits[i].token);
                CHECK(restored[i].logit == Approx(a_logits[i].logit).margin(1e-3));
            }
            // Revenir à un ancêtre exige aussi de restaurer ses logits.
            const auto depth_two = lab.step(s, settings, non_eog[1]);
            CHECK(lab.text_to(depth_two) == piece_a + s.token_to_piece(non_eog[1]));
            // Sans filtre ni température, une pénalité de présence de 2 multiplie les rapports par exp(-2).
            auto penalized = settings;
            penalized.top_k = 0;
            penalized.temperature = 1;
            penalized.presence_penalty = 2;
            penalized.penalty_last_n = -1;
            const auto all = lab.candidates(s, penalized, s.n_vocab());
            auto prompt_tokens = s.tokenize("The capital of France is", true);
            std::set<int32_t> history(prompt_tokens.begin(), prompt_tokens.end());
            history.insert(non_eog[0]);
            history.insert(non_eog[1]);
            const auto reference = std::find_if(
                all.begin(), all.end(), [&](auto& c) { return !history.contains(c.token) && c.p_raw > 1e-12f; });
            REQUIRE(reference != all.end());
            const auto prompt_candidate = std::find_if(all.begin(), all.end(), [&](auto& c) {
                return std::find(prompt_tokens.begin(), prompt_tokens.end(), c.token) != prompt_tokens.end() &&
                       c.p_raw > 1e-20f;
            });
            REQUIRE(prompt_candidate != all.end());
            const auto generated_candidate =
                std::find_if(all.begin(), all.end(), [&](auto& c) { return c.token == non_eog[1]; });
            REQUIRE(generated_candidate != all.end());
            REQUIRE(generated_candidate->p_raw > 0);
            const double ratio = reference->p_final / reference->p_raw;
            CHECK((prompt_candidate->p_final / prompt_candidate->p_raw) / ratio ==
                  Approx(std::exp(-2.0)).epsilon(1e-4));
            CHECK((generated_candidate->p_final / generated_candidate->p_raw) / ratio ==
                  Approx(std::exp(-2.0)).epsilon(1e-4));
            double raw_sum = 0, penalized_sum = 0;
            for (const auto& c : all) {
                raw_sum += c.p_raw;
                penalized_sum += c.p_final;
            }
            CHECK(raw_sum == Approx(1).margin(1e-5));
            CHECK(penalized_sum == Approx(1).margin(1e-5));
            lab.go_to(s, a);
            const auto ancestor = lab.candidates(s, settings, 5);
            for (size_t i = 0; i < ancestor.size(); ++i)
                CHECK(ancestor[i].logit == Approx(a_logits[i].logit).margin(1e-3));
            lab.go_to(s, 0);
            CHECK(lab.step(s, settings, non_eog[0]) == a);
            CHECK(lab.nodes()[0].children.size() == 2);
            lab.go_to(s, 0);
            const auto sampled = lab.step(s, settings);
            lab.go_to(s, 0);
            CHECK(lab.step(s, settings) == sampled);
            lab.go_to(s, 0);
            auto snap = lab.snapshot(s, settings);
            CHECK_FALSE(snap.at_eog);
            CHECK(snap.text.empty());
            CHECK(snap.current == 0);
            const auto eos = llama_vocab_eos(s.vocab());
            if (eos >= 0 && s.is_eog(eos)) {
                lab.step(s, settings, eos);
                CHECK(lab.snapshot(s, settings).at_eog);
                CHECK_THROWS_AS(lab.step(s, settings), Error);
                lab.go_to(s, 0);
                CHECK_FALSE(lab.snapshot(s, settings).at_eog);
            }
            lab.start(s, "New prompt", false);
            CHECK(lab.nodes().size() == 1);
        })
        .get();
}
