#include "core/gguf_reader.hpp"

#include <cstdlib>
#include <system_error>

namespace banc {
namespace {

void add_unique(std::vector<std::filesystem::path>& result, const std::filesystem::path& path) {
    if (path.empty()) return;
    const auto normal = path.lexically_normal();
    for (const auto& existing : result)
        if (existing == normal) return;
    result.push_back(normal);
}

} // namespace

std::vector<std::filesystem::path> candidate_model_roots() {
    std::vector<std::filesystem::path> result;
    if (const auto* home = std::getenv("USERPROFILE"); home && *home) {
        const auto base = std::filesystem::path(home);
        add_unique(result, base / "models");
        add_unique(result, base / ".ollama" / "models" / "blobs");
        add_unique(result, base / ".lmstudio" / "models");
        add_unique(result, base / ".cache" / "lm-studio" / "models");
    }
    if (const auto* ollama = std::getenv("OLLAMA_MODELS"); ollama && *ollama) {
        const auto root = std::filesystem::path(ollama);
        std::error_code ec;
        if (std::filesystem::is_directory(root / "blobs", ec))
            add_unique(result, root / "blobs");
        else
            add_unique(result, root);
    }
    return result;
}

std::vector<std::filesystem::path> default_model_roots() {
    std::vector<std::filesystem::path> result;
    for (const auto& path : candidate_model_roots()) {
        std::error_code ec;
        if (std::filesystem::is_directory(path, ec)) result.push_back(path);
    }
    return result;
}

} // namespace banc
