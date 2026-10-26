// Lecture des métadonnées GGUF sans charger les poids (API gguf de ggml).
#pragma once

#include "core/types.hpp"

#include <filesystem>
#include <vector>

namespace banc {

// Vrai si le fichier commence par la signature "GGUF" (les blobs Ollama n'ont pas d'extension).
bool is_gguf_file(const std::filesystem::path& path);

// Lit l'en-tête GGUF. with_tensors=false saute la liste des tenseurs (plus rapide pour un scan)
// mais remplit quand même n_params et bytes_by_type. Lève banc::Error si le fichier est illisible.
ModelInfo read_gguf(const std::filesystem::path& path, bool with_tensors = true);

// Dossiers de modèles par défaut qui existent sur cette machine :
// %USERPROFILE%\models, %USERPROFILE%\.ollama\models\blobs, %USERPROFILE%\.lmstudio\models,
// %USERPROFILE%\.cache\lm-studio\models.
std::vector<std::filesystem::path> default_model_roots();

// Parcourt récursivement les dossiers et rend les fichiers GGUF (signature vérifiée),
// triés par chemin. Ignore les fichiers illisibles et les morceaux "-0000N-of-" autres que le premier.
std::vector<std::filesystem::path> scan_model_dirs(const std::vector<std::filesystem::path>& roots);

// Faux pour les fichiers GGUF qui ne sont pas des modèles de langue chargeables seuls :
// projecteurs de vision (architecture "clip", blobs mmproj d'Ollama), fichiers sans couches.
bool is_language_model(const ModelInfo& info);

// Traduction de general.file_type (enum llama_ftype) en libellé : 15 -> "Q4_K_M".
std::string ftype_name(uint32_t ftype);

} // namespace banc
