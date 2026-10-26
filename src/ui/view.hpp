// Interface commune des vues ancrables.
#pragma once

#include "ui/app_context.hpp"

#include <memory>

namespace banc::ui {

class View {
public:
    virtual ~View() = default;
    virtual const char* title() const = 0; // titre de fenêtre ImGui, unique et stable (sert aux dispositions)
    virtual const char* icon() const = 0;  // glyphe Lucide (UTF-8) ou "" si la police d'icônes est absente
    // Dessine le contenu dans la fenêtre déjà ouverte par App (ImGui::Begin/End faits par App).
    virtual void draw(AppContext& ctx) = 0;
    bool open = true;
};

std::unique_ptr<View> make_library_view();
std::unique_ptr<View> make_vram_view();
std::unique_ptr<View> make_bench_view();
std::unique_ptr<View> make_sampler_view();
std::unique_ptr<View> make_compare_view();
std::unique_ptr<View> make_monitor_view();

} // namespace banc::ui
