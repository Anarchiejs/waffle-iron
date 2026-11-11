// Small interface helpers shared by the views: formats, theme widgets, layout.
#pragma once

#include "core/types.hpp"

#include <imgui.h>

#include <chrono>
#include <functional>
#include <future>
#include <string>
#include <vector>

namespace waffle::ui {

std::string format_bytes(uint64_t bytes);  // "4.62 GiB", "512 MiB"
std::string format_count(uint64_t n);      // "7.24 G", "350 M"
std::string format_duration_ms(double ms); // "850 ms", "12.4 s"

// True when the future is ready (without blocking). False when it is not valid.
template <class T> bool is_ready(const std::future<T>& f) {
    return f.valid() && f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

// Every interface size is a multiple of the font size (it follows the DPI).
float em(float k = 1.0f);
// Theme font at a scale relative to the base size; always followed by ImGui::PopFont().
void push_font(ImFont* font, float scale = 1.0f);

// Readable model name and secondary line (file, quantization, size).
std::string model_title(const ModelInfo& m);
std::string model_details(const ModelInfo& m);

// Single-line text, cut with "…" when it passes max_x (font nullptr: current font).
void text_ellipsis(ImDrawList* dl, ImFont* font, float size, ImVec2 pos, float max_x, ImU32 color,
                   const std::string& text);
// Library model picker (full width by default); returns true when the index changed.
struct AppContext;
bool model_combo(const char* id, AppContext& ctx, int& index, float width = -FLT_MIN);

// LoadParams sliders (GPU layers, context, KV types, flash attention, ubatch), labels above.
// max_layers = model layer count (clamps the slider). Returns true when modified.
bool load_params_editor(LoadParams& params, int max_layers);

// Progress bar plus a full-width Cancel button; returns true when Cancel is clicked.
bool busy_indicator(const char* label, float fraction = -1.0f);

// Card: a rounded theme panel. size.y == 0: height fits the content. Always close with end_card().
bool begin_card(const char* id, ImVec2 size = {0, 0}, ImGuiWindowFlags flags = 0);
void end_card();
// Card title (semibold) and an optional subtitle.
void card_title(const char* title, const char* subtitle = nullptr);

// Small section title in quiet capitals.
void section_header(const char* text);
// Quiet label above a field; sets the width of the next field (default: full width).
void field_label(const char* text, const char* help = nullptr, float width = -FLT_MIN);
// Two fields side by side (stacked when width is short); each function receives its column width.
void field_pair(const std::function<void(float)>& left, const std::function<void(float)>& right);
// Compact checkbox and a quiet collapsing header, at the theme's proportions.
bool checkbox(const char* label, bool* value);
bool collapsing(const char* label, ImGuiTreeNodeFlags flags = 0);
// "(?)" help bubble next to the last item.
void help_marker(const char* text);
// Quiet text wrapped to the available width.
void dim_wrapped(const char* text);

bool primary_button(const char* label, ImVec2 size = {0, 0});
// Colored inline badge (quantization, state, ...).
void badge(const char* text, const ImVec4& color);
// Thin rounded meter.
void meter(float fraction, const ImVec4& color, float width = -1.0f, float height = 0.0f);
// Centered message for an empty area.
void empty_state(const char* title, const char* text);

// Statistic tiles laid out in as many columns as the width allows.
struct Tile {
    std::string label, value, sub;
    ImVec4 accent;
};
void tile_grid(const std::vector<Tile>& tiles, float min_width);

// Width of a clamped side column, for master / detail layouts.
float side_width(float ratio, float min_em, float max_em);

} // namespace waffle::ui
