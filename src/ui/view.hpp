// Shared interface for the application pages.
#pragma once

#include "ui/app_context.hpp"

#include <memory>

namespace waffle::ui {

class View {
public:
    virtual ~View() = default;
    virtual const char* title() const = 0;    // page name (navigation and header), unique and stable
    virtual const char* subtitle() const = 0; // one sentence under the page title
    virtual const char* nav_title() const { return title(); } // short sidebar label
    virtual const char* icon() const = 0;     // Lucide glyph (UTF-8), or "" when the icon font is missing
    // Draws the page into the area App reserves for it (under the header).
    virtual void draw(AppContext& ctx) = 0;
    // Background work, every frame, even when the page is not shown.
    virtual void tick(AppContext&) {}
};

std::unique_ptr<View> make_library_view();
std::unique_ptr<View> make_vram_view();
std::unique_ptr<View> make_bench_view();
std::unique_ptr<View> make_sampler_view();
std::unique_ptr<View> make_compare_view();
std::unique_ptr<View> make_monitor_view();

} // namespace waffle::ui
