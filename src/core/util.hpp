// Shared path, time, and byte formatting used by the store, the CLI, and the interface.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace waffle {

std::string utf8_from_path(const std::filesystem::path& path);
std::filesystem::path path_from_utf8(std::string_view utf8);

// Resolves the existing prefix and appends any missing tail. Used as the library key.
std::filesystem::path canonical_path(const std::filesystem::path& path);
std::string path_key(const std::filesystem::path& path);

// UTF-8 bytes for APIs that open files as UTF-8 (ggml, Dear ImGui on MSVC).
std::string narrow_utf8(const std::filesystem::path& path);

std::string format_bytes(uint64_t bytes);
std::string format_count(uint64_t count);
std::string now_iso();

// Documents/WaffleIron, creating nothing. Empty when the home directory is unknown.
std::filesystem::path documents_directory();

std::string json_escape(std::string_view text);

} // namespace waffle
