// État partagé par toutes les vues de l'interface. Fichier possédé par Claude.
// Tout est lu et écrit sur le thread de l'interface ; les calculs longs passent par Engine
// (std::future) ou par std::async pour ce qui ne touche pas llama.cpp (scan GGUF, campagne).
#pragma once

#include "core/engine.hpp"
#include "core/telemetry.hpp"
#include "core/types.hpp"
#include "store/store.hpp"

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace banc::ui {

struct Toast {
    std::string text;
    bool error = false;
    double expires_at = 0; // ImGui::GetTime()
};

struct AppContext {
    Engine& engine;
    Telemetry& telemetry;
    Store& store;

    // Bibliothèque : remplie par la vue Bibliothèque (scan + Store), lue par les autres vues.
    std::vector<ModelInfo> library;
    int selected_model = -1; // indice dans library, -1 = aucun

    // Historique de télémétrie (rempli par App à ~10 Hz, gardé 10 minutes) : moniteur, banc d'essai.
    std::deque<GpuSample> gpu_history;

    // Débit de génération en direct, publié par la vue en cours de génération (labo, comparateur).
    std::deque<std::pair<double, double>> tps_history; // (t_seconds, tokens/s)

    // Page demandée par une vue (ex. « Ouvrir dans le planificateur ») ; App l'applique puis remet -1.
    int requested_page = -1;

    std::deque<Toast> toasts;
    void notify(std::string text, bool error = false); // défini dans app.cpp

    const ModelInfo* selected() const {
        return selected_model >= 0 && selected_model < static_cast<int>(library.size())
                   ? &library[static_cast<size_t>(selected_model)]
                   : nullptr;
    }
};

} // namespace banc::ui
