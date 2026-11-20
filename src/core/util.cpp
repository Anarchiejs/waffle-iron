#include "core/util.hpp"

#include "core/types.hpp"

#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iterator>
#include <system_error>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace waffle {

std::string utf8_from_path(const std::filesystem::path& path) {
    const auto u8 = path.u8string();
    return {reinterpret_cast<const char*>(u8.data()), u8.size()};
}

std::filesystem::path path_from_utf8(std::string_view utf8) {
    const std::u8string u8(reinterpret_cast<const char8_t*>(utf8.data()), utf8.size());
    return {u8};
}

namespace {
std::string_view trim_copy_span(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) text.remove_prefix(1);
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t')) text.remove_suffix(1);
    return text;
}
} // namespace

std::filesystem::path expand_user_path(std::string_view text) {
    text = trim_copy_span(text);
    if (text.empty()) return {};
#ifdef _WIN32
    auto wide = path_from_utf8(text).wstring();
    if (wide == L"~" || wide.starts_with(L"~/") || wide.starts_with(L"~\\")) {
        wchar_t profile[32768];
        const DWORD n = GetEnvironmentVariableW(L"USERPROFILE", profile, static_cast<DWORD>(std::size(profile)));
        if (n > 0 && n < std::size(profile)) wide = std::wstring(profile, profile + n) + wide.substr(1);
    }
    const DWORD need = ExpandEnvironmentStringsW(wide.c_str(), nullptr, 0);
    if (need == 0) return path_from_utf8(text);
    std::wstring expanded(need, L'\0');
    if (ExpandEnvironmentStringsW(wide.c_str(), expanded.data(), need) == 0) return path_from_utf8(text);
    if (!expanded.empty() && expanded.back() == L'\0') expanded.pop_back();
    return std::filesystem::path(expanded);
#else
    std::string in(text);
    if (in == "~" || in.starts_with("~/")) {
        if (const char* home = std::getenv("HOME"); home && *home) in = std::string(home) + in.substr(1);
    }
    std::string out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size();) {
        if (in[i] != '$' || i + 1 >= in.size()) {
            out.push_back(in[i++]);
            continue;
        }
        std::string name;
        size_t next = i + 1;
        if (in[next] == '{') {
            const auto end = in.find('}', next + 1);
            if (end == std::string::npos) {
                out.push_back(in[i++]);
                continue;
            }
            name = in.substr(next + 1, end - (next + 1));
            next = end + 1;
        } else {
            size_t j = next;
            while (j < in.size() && (std::isalnum(static_cast<unsigned char>(in[j])) || in[j] == '_')) ++j;
            name = in.substr(next, j - next);
            next = j;
        }
        if (name.empty()) {
            out.push_back(in[i++]);
            continue;
        }
        if (const char* value = std::getenv(name.c_str()); value) {
            out += value;
            i = next;
            continue;
        }
        out.append(in, i, next - i);
        i = next;
    }
    return path_from_utf8(out);
#endif
}

bool model_file_is_gone(const std::filesystem::path& path) {
    std::error_code ec;
    const auto status = std::filesystem::status(path, ec);
    if (std::filesystem::is_regular_file(status)) return false;
    // libstdc++ reports a missing path as no_such_file_or_directory rather than a cleared error.
    // Any other error (permission, I/O) means the file is not confirmed gone.
    if (ec && ec != std::errc::no_such_file_or_directory) return false;
    for (auto parent = path.parent_path(); !parent.empty();) {
        std::error_code parent_ec;
        const auto parent_status = std::filesystem::status(parent, parent_ec);
        if (parent_ec == std::errc::no_such_file_or_directory ||
            (!parent_ec && !std::filesystem::exists(parent_status)))
            return false;
        if (parent_ec) return false;
        const auto next = parent.parent_path();
        if (next == parent) break;
        parent = std::move(next);
    }
    return true;
}

std::filesystem::path canonical_path(const std::filesystem::path& path) {
    if (path.empty()) return {};
    std::error_code ec;
    auto resolved = std::filesystem::weakly_canonical(path, ec);
    if (!ec) return resolved.lexically_normal();
    auto absolute = path;
    if (absolute.is_relative()) {
        auto base = std::filesystem::absolute(absolute, ec);
        if (!ec) absolute = std::move(base);
    }
    return absolute.lexically_normal();
}

std::string path_key(const std::filesystem::path& path) {
    return utf8_from_path(canonical_path(path));
}

std::string narrow_utf8(const std::filesystem::path& path) {
    return utf8_from_path(path);
}

std::string format_bytes(uint64_t bytes) {
    char buf[32];
    if (bytes >= 1024ull * GiB)
        std::snprintf(buf, sizeof buf, "%.2f TiB", bytes / double(1024ull * GiB));
    else if (bytes >= GiB)
        std::snprintf(buf, sizeof buf, "%.2f GiB", bytes / double(GiB));
    else if (bytes >= MiB)
        std::snprintf(buf, sizeof buf, "%.0f MiB", bytes / double(MiB));
    else if (bytes >= KiB)
        std::snprintf(buf, sizeof buf, "%.0f KiB", bytes / double(KiB));
    else
        std::snprintf(buf, sizeof buf, "%llu B", static_cast<unsigned long long>(bytes));
    return buf;
}

std::string format_count(uint64_t count) {
    char buf[32];
    if (count >= 1'000'000'000)
        std::snprintf(buf, sizeof buf, "%.2f G", count / 1e9);
    else if (count >= 1'000'000)
        std::snprintf(buf, sizeof buf, "%.0f M", count / 1e6);
    else if (count >= 1'000)
        std::snprintf(buf, sizeof buf, "%.1f k", count / 1e3);
    else
        std::snprintf(buf, sizeof buf, "%llu", static_cast<unsigned long long>(count));
    return buf;
}

std::string now_iso() {
    const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    if (std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm) == 0) return "1970-01-01T00:00:00Z";
    return buf;
}

std::filesystem::path documents_directory() {
#ifdef _WIN32
    const char* home = std::getenv("USERPROFILE");
    return (home && *home ? std::filesystem::path(home) : std::filesystem::current_path()) / "Documents";
#elif defined(__APPLE__)
    const char* home = std::getenv("HOME");
    return (home && *home ? std::filesystem::path(home) : std::filesystem::current_path()) / "Documents";
#else
    if (const char* xdg = std::getenv("XDG_DOCUMENTS_DIR"); xdg && *xdg) return {xdg};
    const char* home = std::getenv("HOME");
    std::filesystem::path documents =
        (home && *home ? std::filesystem::path(home) : std::filesystem::current_path()) / "Documents";
    if (home && *home) {
        std::ifstream dirs(std::filesystem::path(home) / ".config" / "user-dirs.dirs");
        std::string line;
        while (std::getline(dirs, line)) {
            constexpr std::string_view key = "XDG_DOCUMENTS_DIR=";
            if (!line.starts_with(key)) continue;
            std::string value = line.substr(key.size());
            if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
                value = value.substr(1, value.size() - 2);
            constexpr std::string_view token = "$HOME";
            if (value.starts_with(token)) value = std::string(home) + value.substr(token.size());
            if (!value.empty()) documents = std::filesystem::path(value);
            break;
        }
    }
    return documents;
#endif
}

std::string json_escape(std::string_view text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (unsigned char c : text) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    return out;
}

} // namespace waffle
