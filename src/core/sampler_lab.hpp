// Laboratoire d'échantillonnage : génération pas à pas, top-N candidats, arbre de branches.
// Toutes les méthodes prenant EngineSession& s'appellent DANS un travail du moteur.
// Les accesseurs const (nodes, current…) se lisent sous le mutex de l'appelant : l'interface
// copie l'état via snapshot() rendu par un travail, elle ne lit jamais l'objet en direct.
#pragma once

#include "core/types.hpp"

#include <optional>
#include <string>
#include <vector>

namespace banc {

class EngineSession;

struct SamplerSettings {
    float temperature = 0.8f;
    int top_k = 40;              // 0 = désactivé
    float top_p = 0.95f;         // 1 = désactivé
    float min_p = 0.05f;         // 0 = désactivé
    float repeat_penalty = 1.0f; // 1 = désactivé
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    int penalty_last_n = 64;
    uint32_t seed = 42;

    bool operator==(const SamplerSettings&) const = default;
};

struct Candidate {
    int32_t token = 0;
    std::string piece;
    float logit = 0;
    float p_raw = 0;   // softmax des logits bruts (température 1)
    float p_final = 0; // probabilité après toute la chaîne ; 0 si filtré
};

struct LabNode {
    int id = 0;
    int parent = -1;    // -1 pour la racine (la consigne)
    int32_t token = -1; // -1 pour la racine
    std::string piece;  // texte du token (la consigne entière pour la racine)
    float p_final = 0;  // probabilité du token choisi au moment du choix
    int depth = 0;      // nombre de tokens générés depuis la racine
    std::vector<int> children;
};

struct LabSnapshot {
    std::vector<LabNode> nodes;
    int current = 0;
    std::vector<Candidate> candidates; // top-N au nœud courant avec les réglages donnés
    std::string text;                  // texte généré de la racine au nœud courant
    bool at_eog = false;               // le nœud courant est une fin de génération
};

class SamplerLab {
public:
    // Tokenise la consigne (avec le modèle de conversation si demandé), vide le cache, décode,
    // crée la racine. Efface l'arbre précédent.
    void start(EngineSession& session, const std::string& prompt, bool use_chat_template);
    bool started() const { return !nodes_.empty(); }

    // Top-N candidats au nœud courant, recalculés à partir des logits gardés en mémoire :
    // aucun décodage, donc instantané quand on bouge un curseur.
    std::vector<Candidate> candidates(EngineSession& session, const SamplerSettings& settings, int top_n = 20);

    // Avance d'un token : échantillonne avec settings, ou prend `forced` (choix manuel d'un candidat).
    // Si ce token existe déjà comme enfant du nœud courant, réutilise ce nœud.
    // Décode le token et rend l'id du nouveau nœud courant.
    int step(EngineSession& session, const SamplerSettings& settings, std::optional<int32_t> forced = {});

    // Se place sur un nœud quelconque de l'arbre : vide le cache KV et redécode consigne + chemin
    // (le retrait partiel du cache n'est pas fiable pour toutes les architectures).
    void go_to(EngineSession& session, int node_id);

    LabSnapshot snapshot(EngineSession& session, const SamplerSettings& settings, int top_n = 20);

    const std::vector<LabNode>& nodes() const { return nodes_; }
    int current() const { return current_; }
    std::string text_to(int node_id) const;              // texte généré racine -> nœud (sans la consigne)
    std::vector<int32_t> path_tokens(int node_id) const; // tokens générés racine -> nœud

private:
    std::vector<LabNode> nodes_;
    int current_ = 0;
    std::vector<int32_t> prompt_tokens_;
    std::vector<float> logits_;      // logits du nœud courant (copie)
    std::vector<int32_t> kv_tokens_; // tokens générés actuellement présents dans le cache après la consigne
};

} // namespace banc
