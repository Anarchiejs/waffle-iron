// Lecture des métadonnées GGUF sans charger les poids (API gguf de ggml).
#pragma once

#include "core/types.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace banc {

// Vrai si le fichier commence par la signature "GGUF" (les blobs Ollama n'ont pas d'extension).
bool is_gguf_file(const std::filesystem::path& path);

// Lit l'en-tête GGUF. with_tensors=false saute la liste des tenseurs (plus rapide pour un scan)
// mais remplit quand même n_params et bytes_by_type. Lève banc::Error si le fichier est illisible.
ModelInfo read_gguf(const std::filesystem::path& path, bool with_tensors = true);

// Emplacements connus, même s'ils n'existent pas encore :
// %USERPROFILE%\models, %USERPROFILE%\.ollama\models\blobs, %USERPROFILE%\.lmstudio\models,
// %USERPROFILE%\.cache\lm-studio\models, et le dossier blobs de OLLAMA_MODELS s'il est défini
// (ou OLLAMA_MODELS lui-même s'il n'a pas de sous-dossier blobs).
std::vector<std::filesystem::path> candidate_model_roots();

// Sous-ensemble de candidate_model_roots() dont le dossier existe sur cette machine.
std::vector<std::filesystem::path> default_model_roots();

// Blob d'Ollama (fichier « sha256-… » d'un dossier blobs), identifié par les manifestes voisins
// (<racine>/manifests/<registre>/<espace>/<modèle>/<étiquette>), car son nom de fichier n'est qu'une empreinte.
struct OllamaBlob {
    std::string tags;   // « gemma4:e4b » ; plusieurs étiquettes séparées par « , »
    bool model = false; // vrai si une étiquette s'en sert comme modèle principal (sinon : draft, projecteur…)
};

// nullopt si le fichier n'est pas un blob référencé par un manifeste d'Ollama.
std::optional<OllamaBlob> find_ollama_blob(const std::filesystem::path& path);

// Parcourt récursivement les dossiers et rend les fichiers GGUF (signature vérifiée),
// triés par chemin. Ignore les fichiers illisibles, les morceaux "-0000N-of-" autres que le premier,
// et les blobs d'Ollama qui ne sont pas un modèle principal (modèles draft, projecteurs).
std::vector<std::filesystem::path> scan_model_dirs(const std::vector<std::filesystem::path>& roots);

// Faux pour les fichiers GGUF qui ne sont pas des modèles de langue chargeables seuls :
// projecteurs de vision (architecture "clip", blobs mmproj d'Ollama), fichiers sans couches.
bool is_language_model(const ModelInfo& info);

// Traduction de general.file_type (enum llama_ftype) en libellé : 15 -> "Q4_K_M".
std::string ftype_name(uint32_t ftype);

} // namespace banc
