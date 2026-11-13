#include "core/gguf_reader.hpp"
#include "core/util.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>
#include <system_error>
#include <unordered_map>

#if !defined(_WIN32) && !defined(__APPLE__)
#include <pwd.h>
#include <unistd.h>
#endif

namespace waffle {
namespace {

void add_unique(std::vector<std::filesystem::path>& result, const std::filesystem::path& path) {
    if (path.empty()) return;
    const auto normal = path.lexically_normal();
    for (const auto& existing : result)
        if (existing == normal) return;
    result.push_back(normal);
}

// OLLAMA_MODELS is the models directory. Blobs live in a subdirectory when Ollama created one.
void add_ollama_models_path(std::vector<std::filesystem::path>& result, const std::filesystem::path& root) {
    if (root.empty()) return;
    std::error_code ec;
    if (std::filesystem::is_directory(root / "blobs", ec))
        add_unique(result, root / "blobs");
    else
        add_unique(result, root);
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

std::filesystem::path profile_directory() {
#ifdef _WIN32
    if (const auto* home = std::getenv("USERPROFILE"); home && *home) return {home};
    return {};
#else
    // Unix sessions use HOME. USERPROFILE is a Windows variable and is ignored here.
    if (const auto* home = std::getenv("HOME"); home && *home) return {home};
    return {};
#endif
}

} // namespace

struct OllamaIndex {
    std::unordered_map<std::string, OllamaBlob> by_digest;
};

struct OllamaCache {
    std::mutex mutex;
    std::unordered_map<std::string, OllamaIndex> by_manifests;
};

OllamaCache& ollama_cache() {
    static OllamaCache cache;
    return cache;
}

OllamaIndex index_manifests(const std::filesystem::path& manifests) {
    // Manifests are simple JSON: each layer is an object with no nested object,
    // with "mediaType" (…image.model, …image.draft, …image.projector) and "digest" ("sha256:…").
    static const std::regex layer(R"(\{[^{}]*\})");
    static const std::regex media(R"re("mediaType"\s*:\s*"([^"]*)")re");
    static const std::regex sha(R"re("digest"\s*:\s*"sha256:([0-9a-fA-F]+)")re");
    OllamaIndex index;
    std::error_code ec;
    std::filesystem::recursive_directory_iterator it(manifests,
                                                     std::filesystem::directory_options::skip_permission_denied, ec),
        end;
    for (; it != end; it.increment(ec)) {
        if (ec || !it->is_regular_file(ec)) continue;
        std::ifstream in(it->path(), std::ios::binary);
        std::stringstream text;
        text << in.rdbuf();
        const auto json = text.str();
        const auto tag = ollama_tag(std::filesystem::relative(it->path(), manifests, ec));
        for (std::sregex_iterator l(json.begin(), json.end(), layer), last; l != last; ++l) {
            const auto object = l->str();
            std::smatch m, d;
            if (!std::regex_search(object, d, sha)) continue;
            auto& blob = index.by_digest[d[1].str()];
            if (blob.tags.find(tag) == std::string::npos) blob.tags += (blob.tags.empty() ? "" : ", ") + tag;
            if (std::regex_search(object, m, media) && m[1].str().ends_with(".model")) blob.model = true;
        }
    }
    return index;
}

void clear_ollama_cache() {
    auto& cache = ollama_cache();
    std::lock_guard lock(cache.mutex);
    cache.by_manifests.clear();
}

std::optional<OllamaBlob> find_ollama_blob(const std::filesystem::path& path) {
    const auto file = utf8_from_path(path.filename());
    if (!file.starts_with("sha256-")) return std::nullopt;
    const auto digest = file.substr(7);
    const auto manifests = path.parent_path().parent_path() / "manifests";
    std::error_code ec;
    if (!std::filesystem::is_directory(manifests, ec)) return std::nullopt;
    auto& cache = ollama_cache();
    std::lock_guard lock(cache.mutex);
    const std::string key = utf8_from_path(manifests);
    auto it = cache.by_manifests.find(key);
    if (it == cache.by_manifests.end()) it = cache.by_manifests.emplace(key, index_manifests(manifests)).first;
    const auto found = it->second.by_digest.find(digest);
    if (found == it->second.by_digest.end()) return std::nullopt;
    return found->second;
}

namespace {

std::string read_text_limited(const std::filesystem::path& path, std::size_t cap = 256 * 1024) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) return {};
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::string text(cap, '\0');
    in.read(text.data(), static_cast<std::streamsize>(cap));
    text.resize(static_cast<std::size_t>(in.gcount()));
    return text;
}

#if !defined(_WIN32) && !defined(__APPLE__)
std::optional<std::filesystem::path> ollama_service_home() {
    long configured = sysconf(_SC_GETPW_R_SIZE_MAX);
    std::size_t bufsize = configured > 0 ? static_cast<std::size_t>(configured) : 16384;
    for (int attempt = 0; attempt < 3; ++attempt) {
        std::string buffer(bufsize, '\0');
        passwd pwd{};
        passwd* result = nullptr;
        const int rc = getpwnam_r("ollama", &pwd, buffer.data(), buffer.size(), &result);
        if (rc == ERANGE) {
            bufsize *= 2;
            continue;
        }
        if (rc != 0 || !result || !result->pw_dir || result->pw_dir[0] == '\0') return std::nullopt;
        return std::filesystem::path(result->pw_dir);
    }
    return std::nullopt;
}

void note_systemd_file(const std::filesystem::path& path, std::optional<std::string>& models) {
    const auto text = read_text_limited(path);
    if (text.empty()) return;
    if (auto value = ollama_models_from_systemd_text(text)) {
        if (value->empty())
            models.reset();
        else
            models = std::move(*value);
    }
}

std::optional<std::filesystem::path> systemd_ollama_models() {
    std::optional<std::string> models;
    const std::filesystem::path units[] = {
        "/usr/lib/systemd/system/ollama.service",
        "/lib/systemd/system/ollama.service",
        "/etc/systemd/system/ollama.service",
    };
    for (const auto& unit : units) note_systemd_file(unit, models);
    std::vector<std::filesystem::path> dropins;
    const std::filesystem::path directories[] = {
        "/usr/lib/systemd/system/ollama.service.d",
        "/lib/systemd/system/ollama.service.d",
        "/etc/systemd/system/ollama.service.d",
    };
    for (const auto& directory : directories) {
        std::error_code ec;
        if (!std::filesystem::is_directory(directory, ec)) continue;
        std::filesystem::directory_iterator it(directory, std::filesystem::directory_options::skip_permission_denied,
                                               ec);
        for (; !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
            if (it->path().extension() == ".conf") dropins.push_back(it->path());
        }
    }
    std::sort(dropins.begin(), dropins.end());
    for (const auto& dropin : dropins) note_systemd_file(dropin, models);
    if (!models || models->empty()) return std::nullopt;
    return std::filesystem::path(*models);
}
#endif

} // namespace

std::optional<std::string> ollama_models_from_systemd_text(std::string_view text) {
    std::optional<std::string> found;
    bool mentioned = false;
    std::size_t line_start = 0;
    while (line_start <= text.size()) {
        auto line_end = text.find('\n', line_start);
        if (line_end == std::string_view::npos) line_end = text.size();
        auto line = text.substr(line_start, line_end - line_start);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        std::size_t indent = 0;
        while (indent < line.size() && (line[indent] == ' ' || line[indent] == '\t')) ++indent;
        line.remove_prefix(indent);
        constexpr std::string_view key = "Environment=";
        if (!line.empty() && line.front() != '#' && line.front() != ';' && line.starts_with(key)) {
            auto rest = line.substr(key.size());
            std::size_t i = 0;
            while (i < rest.size()) {
                while (i < rest.size() && (rest[i] == ' ' || rest[i] == '\t')) ++i;
                if (i >= rest.size()) break;
                std::string token;
                if (rest[i] == '"' || rest[i] == '\'') {
                    const char quote = rest[i++];
                    while (i < rest.size() && rest[i] != quote) {
                        if (rest[i] == '\\' && i + 1 < rest.size()) {
                            token.push_back(rest[i + 1]);
                            i += 2;
                            continue;
                        }
                        token.push_back(rest[i++]);
                    }
                    if (i < rest.size() && rest[i] == quote) ++i;
                } else {
                    while (i < rest.size() && rest[i] != ' ' && rest[i] != '\t') token.push_back(rest[i++]);
                }
                constexpr std::string_view assignment = "OLLAMA_MODELS=";
                if (token.starts_with(assignment)) {
                    mentioned = true;
                    found = token.substr(assignment.size());
                }
            }
        }
        if (line_end == text.size()) break;
        line_start = line_end + 1;
    }
    if (!mentioned) return std::nullopt;
    return found;
}

std::vector<std::filesystem::path> candidate_model_roots() {
    std::vector<std::filesystem::path> result;
    if (const auto base = profile_directory(); !base.empty()) {
        add_unique(result, base / "models");
        add_unique(result, base / ".ollama" / "models" / "blobs");
        add_unique(result, base / ".lmstudio" / "models");
        add_unique(result, base / ".cache" / "lm-studio" / "models");
    }
    if (const auto* ollama = std::getenv("OLLAMA_MODELS"); ollama && *ollama)
        add_ollama_models_path(result, std::filesystem::path(ollama));
#if !defined(_WIN32) && !defined(__APPLE__)
    if (const auto from_unit = systemd_ollama_models()) add_ollama_models_path(result, *from_unit);
    if (const auto home = ollama_service_home()) add_unique(result, *home / ".ollama" / "models" / "blobs");
    // Official install script: useradd -d /usr/share/ollama. Kept when getpwnam cannot see that user.
    add_unique(result, "/usr/share/ollama/.ollama/models/blobs");
#endif
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
