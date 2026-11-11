#include "core/gguf_reader.hpp"

#include <cstdlib>
#include <fstream>
#include <regex>
#include <sstream>
#include <system_error>

namespace waffle {
namespace {

void add_unique(std::vector<std::filesystem::path>& result, const std::filesystem::path& path) {
    if (path.empty()) return;
    const auto normal = path.lexically_normal();
    for (const auto& existing : result)
        if (existing == normal) return;
    result.push_back(normal);
}

// "registry.ollama.ai/library/gemma4/e4b" -> "gemma4:e4b"; another namespace or registry is kept as a prefix.
std::string ollama_tag(const std::filesystem::path& relative) {
    std::vector<std::string> parts;
    for (const auto& part : relative) parts.push_back(part.string());
    if (parts.size() < 4) return relative.generic_string();
    std::string tag = parts[parts.size() - 2] + ":" + parts.back();
    if (parts[parts.size() - 3] != "library") tag = parts[parts.size() - 3] + "/" + tag;
    if (parts.size() == 4 && parts[0] != "registry.ollama.ai") tag = parts[0] + "/" + tag;
    return tag;
}

} // namespace

std::optional<OllamaBlob> find_ollama_blob(const std::filesystem::path& path) {
    const auto file = path.filename().string();
    if (!file.starts_with("sha256-")) return std::nullopt;
    const auto digest = file.substr(7);
    const auto manifests = path.parent_path().parent_path() / "manifests";
    std::error_code ec;
    if (!std::filesystem::is_directory(manifests, ec)) return std::nullopt;
    // Manifests are simple JSON: each layer is an object with no nested object,
    // with "mediaType" (…image.model, …image.draft, …image.projector) and "digest" ("sha256:…").
    static const std::regex layer(R"(\{[^{}]*\})");
    static const std::regex media(R"re("mediaType"\s*:\s*"([^"]*)")re");
    static const std::regex sha(R"re("digest"\s*:\s*"sha256:([0-9a-fA-F]+)")re");
    std::optional<OllamaBlob> found;
    std::filesystem::recursive_directory_iterator it(manifests, std::filesystem::directory_options::skip_permission_denied,
                                                     ec),
        end;
    for (; it != end; it.increment(ec)) {
        if (ec || !it->is_regular_file(ec)) continue;
        std::ifstream in(it->path(), std::ios::binary);
        std::stringstream text;
        text << in.rdbuf();
        const auto json = text.str();
        for (std::sregex_iterator l(json.begin(), json.end(), layer), last; l != last; ++l) {
            const auto object = l->str();
            std::smatch m, d;
            if (!std::regex_search(object, d, sha) || d[1].str() != digest) continue;
            if (!found) found.emplace();
            const auto tag = ollama_tag(std::filesystem::relative(it->path(), manifests, ec));
            if (found->tags.find(tag) == std::string::npos)
                found->tags += (found->tags.empty() ? "" : ", ") + tag;
            if (std::regex_search(object, m, media) && m[1].str().ends_with(".model")) found->model = true;
        }
    }
    return found;
}

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

} // namespace waffle
