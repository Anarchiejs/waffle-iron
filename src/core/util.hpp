// Shared path, time, and byte formatting used by the store, the CLI, and the interface.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace waffle {

std::string utf8_from_path(const std::filesystem::path& path);
std::filesystem::path path_from_utf8(std::string_view utf8);

// User-typed folder. Expands a leading ~ to the home directory, $VAR and ${VAR} on Unix,
// and %VAR% (and a leading ~) on Windows. Unknown variables are left unchanged.
std::filesystem::path expand_user_path(std::string_view text);

// True only when the path is confirmed not to be a regular file and every existing
// ancestor is present. A stat error, or a missing parent (unplugged volume, cut share,
// unreadable directory), returns false so a failed stat is not treated as deletion.
bool model_file_is_gone(const std::filesystem::path& path);

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
