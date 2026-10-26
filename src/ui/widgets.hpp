// Petits outils d'interface partagés par les vues (implémentés par le lot L4 dans widgets.cpp).
#pragma once

#include "core/types.hpp"

#include <chrono>
#include <future>
#include <string>

struct ImVec4;

namespace banc::ui {

std::string format_bytes(uint64_t bytes);  // "4.62 Go", "512 Mo"
std::string format_count(uint64_t n);      // "7.24 G", "350 M"
std::string format_duration_ms(double ms); // "850 ms", "12.4 s"

// Vrai si le futur est prêt (sans bloquer). Faux s'il n'est pas valide.
template <class T> bool is_ready(const std::future<T>& f) {
    return f.valid() && f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

// Sélecteur de modèle de la bibliothèque (combo) ; rend true si l'indice a changé.
struct AppContext;
bool model_combo(const char* label, AppContext& ctx, int& index);

// Curseurs de LoadParams (couches GPU, contexte, types KV, flash attention, ubatch).
// max_layers = nombre de couches du modèle (pour borner le curseur). Rend true si modifié.
bool load_params_editor(LoadParams& params, int max_layers);

// Indicateur d'activité + bouton Annuler si le moteur est occupé ; rend true si Annuler cliqué.
bool busy_indicator(const char* label, float fraction = -1.0f);

// Titre de section dans le style du thème.
void section_header(const char* text);

// Bulle d'aide "(?)" à côté du dernier élément.
void help_marker(const char* text);

} // namespace banc::ui
