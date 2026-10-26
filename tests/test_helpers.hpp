// Aides communes aux tests. Fichier possédé par Claude.
#pragma once

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <optional>

namespace banc::test {

// Modèle GGUF pour les tests qui ont besoin d'un vrai modèle :
// 1. variable d'environnement BANC_TEST_MODEL ;
// 2. sinon tests/data/tiny-llama.gguf (généré par scripts/make_tiny_model.py) ;
// 3. sinon nullopt -> le test appelle SKIP().
inline std::optional<std::filesystem::path> test_model_path() {
    if (const char* env = std::getenv("BANC_TEST_MODEL"); env && *env) {
        std::filesystem::path p(env);
        if (std::filesystem::is_regular_file(p)) return p;
    }
    std::filesystem::path tiny = std::filesystem::path(BANC_SOURCE_DIR) / "tests" / "data" / "tiny-llama.gguf";
    if (std::filesystem::is_regular_file(tiny)) return tiny;
    return std::nullopt;
}

// Dossier temporaire propre au test, supprimé à la destruction.
struct TempDir {
    std::filesystem::path path;
    explicit TempDir(const char* name) {
        path = std::filesystem::temp_directory_path() / "banc_tests" / name;
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

} // namespace banc::test

#define BANC_REQUIRE_MODEL(var)                                                                                        \
    auto var##_opt = ::banc::test::test_model_path();                                                                  \
    if (!var##_opt) SKIP("Aucun modèle de test (BANC_TEST_MODEL ou tests/data/tiny-llama.gguf)");                      \
    const std::filesystem::path var = *var##_opt
