// Reads GGUF metadata without loading weights (ggml gguf API).
#pragma once

#include "core/types.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace waffle {

// True if the file starts with the "GGUF" signature (Ollama blobs have no extension).
bool is_gguf_file(const std::filesystem::path& path);

// Reads the GGUF header. with_tensors=false skips the tensor list (faster for a scan)
// but still fills n_params and bytes_by_type. Throws waffle::Error if the file is unreadable.
ModelInfo read_gguf(const std::filesystem::path& path, bool with_tensors = true);

// Known locations, even if they do not exist yet:
// %USERPROFILE%\models, %USERPROFILE%\.ollama\models\blobs, %USERPROFILE%\.lmstudio\models,
// %USERPROFILE%\.cache\lm-studio\models, and the blobs directory of OLLAMA_MODELS when set
// (or OLLAMA_MODELS itself when it has no blobs subdirectory).
std::vector<std::filesystem::path> candidate_model_roots();

// Subset of candidate_model_roots() whose directory exists on this machine.
std::vector<std::filesystem::path> default_model_roots();

// Ollama blob (a "sha256-…" file in a blobs directory), identified by neighboring manifests
// (<root>/manifests/<registry>/<namespace>/<model>/<tag>), because its file name is only a digest.
struct OllamaBlob {
    std::string tags;   // "gemma4:e4b"; several tags separated by ", "
    bool model = false; // true if a tag uses it as the main model (otherwise: draft, projector…)
};

// nullopt if the file is not a blob referenced by an Ollama manifest.
std::optional<OllamaBlob> find_ollama_blob(const std::filesystem::path& path);

// Recursively walks directories and returns GGUF files (signature checked),
// sorted by path. Skips unreadable files, "-0000N-of-" shards other than the first,
// and Ollama blobs that are not a main model (draft models, projectors).
std::vector<std::filesystem::path> scan_model_dirs(const std::vector<std::filesystem::path>& roots);

// False for GGUF files that are not language models loadable on their own:
// vision projectors (architecture "clip", Ollama mmproj blobs), files with no layers.
bool is_language_model(const ModelInfo& info);

// Translation of general.file_type (llama_ftype enum) to a label: 15 -> "Q4_K_M".
std::string ftype_name(uint32_t ftype);

} // namespace waffle
