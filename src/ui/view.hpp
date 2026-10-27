// Interface commune des pages de l'application.
#pragma once

#include "ui/app_context.hpp"

#include <memory>

namespace banc::ui {

class View {
public:
    virtual ~View() = default;
    virtual const char* title() const = 0;    // nom de la page (navigation et en-tête), unique et stable
    virtual const char* subtitle() const = 0; // une phrase sous le titre de la page
    virtual const char* nav_title() const { return title(); } // libellé court de la barre latérale
    virtual const char* icon() const = 0;     // glyphe Lucide (UTF-8) ou "" si la police d'icônes est absente
    // Dessine le contenu de la page dans la zone que lui réserve App (sous l'en-tête).
    virtual void draw(AppContext& ctx) = 0;
    // Travail de fond, chaque image, même si la page n'est pas affichée.
    virtual void tick(AppContext&) {}
};

std::unique_ptr<View> make_library_view();
std::unique_ptr<View> make_vram_view();
std::unique_ptr<View> make_bench_view();
std::unique_ptr<View> make_sampler_view();
std::unique_ptr<View> make_compare_view();
std::unique_ptr<View> make_monitor_view();

} // namespace banc::ui
