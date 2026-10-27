// Petits outils d'interface partagés par les vues : formats, composants du thème, mise en page.
#pragma once

#include "core/types.hpp"

#include <imgui.h>

#include <chrono>
#include <functional>
#include <future>
#include <string>
#include <vector>

namespace banc::ui {

std::string format_bytes(uint64_t bytes);  // "4.62 Go", "512 Mo"
std::string format_count(uint64_t n);      // "7.24 G", "350 M"
std::string format_duration_ms(double ms); // "850 ms", "12.4 s"

// Vrai si le futur est prêt (sans bloquer). Faux s'il n'est pas valide.
template <class T> bool is_ready(const std::future<T>& f) {
    return f.valid() && f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

// Toutes les dimensions de l'interface s'expriment en multiples de la taille de police (suit le DPI).
float em(float k = 1.0f);
// Police du thème à une échelle relative à la taille de base ; toujours suivi de ImGui::PopFont().
void push_font(ImFont* font, float scale = 1.0f);

// Nom lisible d'un modèle et ligne secondaire (fichier, quantification, taille).
std::string model_title(const ModelInfo& m);
std::string model_details(const ModelInfo& m);

// Texte sur une ligne, raccourci par « … » s'il dépasse max_x (font nullptr : police courante).
void text_ellipsis(ImDrawList* dl, ImFont* font, float size, ImVec2 pos, float max_x, ImU32 color,
                   const std::string& text);
// Sélecteur de modèle de la bibliothèque (pleine largeur par défaut) ; rend true si l'indice a changé.
struct AppContext;
bool model_combo(const char* id, AppContext& ctx, int& index, float width = -FLT_MIN);

// Curseurs de LoadParams (couches GPU, contexte, types KV, flash attention, ubatch), libellés au-dessus.
// max_layers = nombre de couches du modèle (pour borner le curseur). Rend true si modifié.
bool load_params_editor(LoadParams& params, int max_layers);

// Barre de progression + bouton Annuler sur toute la largeur ; rend true si Annuler cliqué.
bool busy_indicator(const char* label, float fraction = -1.0f);

// Carte : panneau arrondi du thème. size.y == 0 : hauteur ajustée au contenu. Toujours fermer par end_card().
bool begin_card(const char* id, ImVec2 size = {0, 0}, ImGuiWindowFlags flags = 0);
void end_card();
// Titre de carte (demi-gras) et sous-titre facultatif.
void card_title(const char* title, const char* subtitle = nullptr);

// Petit titre de section en capitales discrètes.
void section_header(const char* text);
// Libellé discret au-dessus d'un champ ; fixe la largeur du champ suivant (défaut : toute la largeur).
void field_label(const char* text, const char* help = nullptr, float width = -FLT_MIN);
// Deux champs côte à côte (empilés si la largeur manque) ; chaque fonction reçoit la largeur de sa colonne.
void field_pair(const std::function<void(float)>& left, const std::function<void(float)>& right);
// Case à cocher compacte et en-tête repliable discret, aux proportions du thème.
bool checkbox(const char* label, bool* value);
bool collapsing(const char* label, ImGuiTreeNodeFlags flags = 0);
// Bulle d'aide "(?)" à côté du dernier élément.
void help_marker(const char* text);
// Texte discret replié à la largeur disponible.
void dim_wrapped(const char* text);

bool primary_button(const char* label, ImVec2 size = {0, 0});
// Pastille colorée en ligne (quantification, état...).
void badge(const char* text, const ImVec4& color);
// Jauge fine arrondie.
void meter(float fraction, const ImVec4& color, float width = -1.0f, float height = 0.0f);
// Message centré pour une zone vide.
void empty_state(const char* title, const char* text);

// Tuiles de statistiques réparties sur autant de colonnes que la largeur le permet.
struct Tile {
    std::string label, value, sub;
    ImVec4 accent;
};
void tile_grid(const std::vector<Tile>& tiles, float min_width);

// Largeur d'une colonne latérale bornée, pour les dispositions maître / détail.
float side_width(float ratio, float min_em, float max_em);

} // namespace banc::ui
