#include "ui/app.hpp"
#include "ui/theme.hpp"
#include "ui/widgets.hpp"
#include <imgui.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
namespace waffle::ui {
void request_library_scan(View& view);
namespace {
// Navigation marks drawn as strokes (no icon font required).
void draw_nav_icon(ImDrawList* dl, int page, ImVec2 c, float s, ImU32 color) {
    const float t = std::max(1.0f, s * .11f);
    const float h = s * .5f;
    switch (page) {
        case page_library: // three book spines
            dl->AddRect({c.x - h, c.y - h * .8f}, {c.x - h * .4f, c.y + h}, color, 1.5f, 0, t);
            dl->AddRect({c.x - h * .25f, c.y - h}, {c.x + h * .35f, c.y + h}, color, 1.5f, 0, t);
            dl->AddLine({c.x + h * .55f, c.y - h * .65f}, {c.x + h, c.y + h}, color, t);
            break;
        case page_vram: // a chip and its pins
            dl->AddRect({c.x - h * .62f, c.y - h * .62f}, {c.x + h * .62f, c.y + h * .62f}, color, 2.0f, 0, t);
            for (int i = -1; i <= 1; i += 2) {
                const float o = h * .3f * static_cast<float>(i);
                dl->AddLine({c.x + o, c.y - h}, {c.x + o, c.y - h * .62f}, color, t);
                dl->AddLine({c.x + o, c.y + h * .62f}, {c.x + o, c.y + h}, color, t);
                dl->AddLine({c.x - h, c.y + o}, {c.x - h * .62f, c.y + o}, color, t);
                dl->AddLine({c.x + h * .62f, c.y + o}, {c.x + h, c.y + o}, color, t);
            }
            break;
        case page_bench: // a bar chart
            dl->AddRectFilled({c.x - h, c.y + h * .1f}, {c.x - h * .45f, c.y + h}, color, 1.0f);
            dl->AddRectFilled({c.x - h * .27f, c.y - h * .4f}, {c.x + h * .27f, c.y + h}, color, 1.0f);
            dl->AddRectFilled({c.x + h * .45f, c.y - h}, {c.x + h, c.y + h}, color, 1.0f);
            break;
        case page_sampler: // a branch tree
            dl->AddLine({c.x - h * .55f, c.y}, {c.x + h * .55f, c.y - h * .6f}, color, t);
            dl->AddLine({c.x - h * .55f, c.y}, {c.x + h * .55f, c.y + h * .6f}, color, t);
            dl->AddCircleFilled({c.x - h * .65f, c.y}, h * .3f, color);
            dl->AddCircleFilled({c.x + h * .65f, c.y - h * .65f}, h * .3f, color);
            dl->AddCircleFilled({c.x + h * .65f, c.y + h * .65f}, h * .3f, color);
            break;
        case page_compare: // A | B
            dl->AddRect({c.x - h, c.y - h * .8f}, {c.x - h * .12f, c.y + h * .8f}, color, 2.0f, 0, t);
            dl->AddRect({c.x + h * .12f, c.y - h * .8f}, {c.x + h, c.y + h * .8f}, color, 2.0f, 0, t);
            break;
        default: { // monitor: a pulse
            const ImVec2 pts[] = {{c.x - h, c.y},
                                  {c.x - h * .45f, c.y},
                                  {c.x - h * .2f, c.y - h * .8f},
                                  {c.x + h * .15f, c.y + h * .8f},
                                  {c.x + h * .4f, c.y},
                                  {c.x + h, c.y}};
            dl->AddPolyline(pts, 6, color, 0, t);
        }
    }
}
void draw_font_icon(ImDrawList* dl, const char* glyph, ImVec2 center, float size, ImU32 color) {
    ImFont* font = fonts().icons ? fonts().icons : fonts().regular;
    if (!font || !glyph || !glyph[0]) return;
    const ImVec2 text = font->CalcTextSizeA(size, FLT_MAX, 0, glyph);
    dl->AddText(font, size, {center.x - text.x * .5f, center.y - text.y * .5f}, color, glyph);
}
bool nav_item(int page, const char* label, const char* glyph, bool use_font, bool active, bool compact) {
    const float w = ImGui::GetContentRegionAvail().x;
    const float h = em(2.5f);
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::PushID(page);
    const bool clicked = ImGui::InvisibleButton("nav", {w, h});
    const bool hovered = ImGui::IsItemHovered();
    ImGui::PopID();
    auto* dl = ImGui::GetWindowDrawList();
    const float r = em(.6f);
    if (active)
        dl->AddRectFilled(pos, {pos.x + w, pos.y + h}, ImGui::GetColorU32(col::accent_bg), r);
    else if (hovered)
        dl->AddRectFilled(pos, {pos.x + w, pos.y + h}, ImGui::GetColorU32(ImVec4(1, .95f, .85f, .04f)), r);
    const ImU32 icon = ImGui::GetColorU32(active ? col::accent : hovered ? col::text : col::soft);
    const ImVec2 icon_center =
        compact ? ImVec2{pos.x + w * .5f, pos.y + h * .5f} : ImVec2{pos.x + em(1.3f), pos.y + h * .5f};
    if (use_font && glyph && glyph[0])
        draw_font_icon(dl, glyph, icon_center, em(compact ? 1.2f : 1.1f), icon);
    else
        draw_nav_icon(dl, page, icon_center, em(compact ? 1.05f : 1.0f), icon);
    if (compact) {
        // Narrow bar: icon only, label in a tooltip.
        if (hovered) ImGui::SetTooltip("%s  (Ctrl+%d)", label, page + 1);
        return clicked;
    }
    ImFont* font = active ? fonts().semibold : fonts().regular;
    const float fs = em(1.0f);
    dl->AddText(font, fs, {pos.x + em(2.55f), pos.y + (h - fs) * .5f - 1},
                ImGui::GetColorU32(active    ? col::accent
                                   : hovered ? col::text
                                             : col::soft),
                label);
    char key[12];
    std::snprintf(key, sizeof key, "Ctrl %d", page + 1);
    const float ks = em(.74f);
    const float kw = fonts().mono->CalcTextSizeA(ks, FLT_MAX, 0, key).x;
    dl->AddText(fonts().mono, ks, {pos.x + w - kw - em(.7f), pos.y + (h - ks) * .5f}, ImGui::GetColorU32(col::faint),
                key);
    if (hovered) ImGui::SetTooltip("Ctrl+%d", page + 1);
    return clicked;
}
} // namespace
App::App(bool icons)
    : store_(Store::default_path()), ctx_{engine_, telemetry_, store_, {}, -1, {}, {}, -1, {}}, icons_(icons) {
    // Order of the Page enum (view.hpp): understand, measure, tune.
    views_ = {make_library_view(), make_vram_view(),    make_bench_view(),
              make_monitor_view(), make_sampler_view(), make_compare_view()};
    // Starting page (development screenshots: WAFFLE_PAGE=0..5).
    if (const char* page = std::getenv("WAFFLE_PAGE")) page_ = std::clamp(std::atoi(page), 0, 5);
}
App::~App() {
    // Wait for the engine before destroying the views and their context references.
    engine_.cancel_all();
    engine_.submit([](EngineSession&) {}).wait();
}
void App::sidebar(float width, float height, bool compact) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, col::sidebar);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {em(.85f), em(1.2f)});
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 0);
    ImGui::BeginChild("##Navigation", {width, height}, ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    auto* dl = ImGui::GetWindowDrawList();
    // Mark: the waffle square, the name, and the versions.
    {
        const ImVec2 pos = ImGui::GetCursorScreenPos();
        const float s = em(2.3f);
        const ImVec2 p{compact ? pos.x + (ImGui::GetContentRegionAvail().x - s) * .5f : pos.x + em(.35f), pos.y};
        draw_mark(dl, p, s);
        if (compact) {
            ImGui::SetCursorScreenPos(p);
            if (ImGui::InvisibleButton("##mark", {s, s})) about_ = true;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("About Waffle Iron");
            ImGui::SetCursorScreenPos(pos);
        }
        if (!compact)
            dl->AddText(fonts().display, em(1.15f), {p.x + s + em(.75f), p.y - em(.02f)}, ImGui::GetColorU32(col::text),
                        "Waffle Iron");
        char version[96];
        std::snprintf(version, sizeof version, "v%s · llama.cpp %s", WAFFLE_VERSION, WAFFLE_LLAMA_VERSION);
        const ImVec4 clip{pos.x, pos.y, pos.x + width - em(1), pos.y + s + 4};
        if (!compact)
            dl->AddText(fonts().mono, em(.72f), {p.x + s + em(.75f), p.y + em(1.38f)}, ImGui::GetColorU32(col::dim),
                        version, nullptr, 0, &clip);
        ImGui::Dummy({0, s + em(.8f)});
    }
    // Pages, grouped: understand, measure, tune.
    const char* group = nullptr;
    for (int i = 0; i < static_cast<int>(views_.size()); ++i) {
        const auto& view = *views_[static_cast<size_t>(i)];
        if (!compact && group != page_group(i)) {
            group = page_group(i);
            ImGui::Dummy({0, em(.35f)});
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + em(.55f));
            push_font(fonts().semibold, .72f);
            ImGui::PushStyleColor(ImGuiCol_Text, col::faint);
            ImGui::TextUnformatted(group);
            ImGui::PopStyleColor();
            ImGui::PopFont();
        } else if (compact && i && page_group(i) != page_group(i - 1)) {
            // Narrow bar: a short rule between groups.
            const ImVec2 p = ImGui::GetCursorScreenPos();
            const float w = ImGui::GetContentRegionAvail().x;
            dl->AddLine({p.x + w * .3f, p.y + em(.3f)}, {p.x + w * .7f, p.y + em(.3f)},
                        ImGui::GetColorU32(col::border));
            ImGui::Dummy({0, em(.6f)});
        }
        if (nav_item(i, view.nav_title(), view.icon(), icons_, i == page_, compact)) page_ = i;
    }
    if (compact) {
        // A small VRAM meter at the foot of the bar.
        if (!ctx_.gpu_history.empty()) {
            const auto& g = ctx_.gpu_history.back();
            const float f = g.vram_total ? static_cast<float>(static_cast<double>(g.vram_used) / g.vram_total) : 0;
            const float bottom = ImGui::GetWindowHeight() - ImGui::GetStyle().WindowPadding.y - em(2.2f);
            if (ImGui::GetCursorPosY() < bottom) {
                ImGui::SetCursorPosY(bottom);
                char text[16];
                std::snprintf(text, sizeof text, "%.0f %%", 100.0 * f);
                push_font(fonts().mono, .74f);
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() +
                                     (ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(text).x) * .5f);
                ImGui::TextDisabled("%s", text);
                ImGui::PopFont();
                cell_meter(f, load_color(f), 4, -1, em(.38f));
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%s %s / %s", telemetry_.reports_process_memory() ? "Allocation" : "VRAM",
                                      format_bytes(g.vram_used).c_str(), format_bytes(g.vram_total).c_str());
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor();
        const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
        ImGui::GetWindowDrawList()->AddLine({b.x - 1, a.y}, {b.x - 1, b.y}, ImGui::GetColorU32(col::border));
        return;
    }

    // GPU card at the bottom of the sidebar.
    const float card_h = em(8.1f);
    const float bottom = ImGui::GetWindowHeight() - ImGui::GetStyle().WindowPadding.y;
    if (ImGui::GetCursorPosY() + em(1) < bottom - card_h) {
        ImGui::SetCursorPosY(bottom - card_h);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, col::card);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {em(.95f), em(.85f)});
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, em(.8f));
        ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {em(.5f), em(.4f)});
        ImGui::BeginChild("##GpuCard", {0, card_h - em(1.75f)}, ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        const auto name = telemetry_.device_name();
        const float f = !ctx_.gpu_history.empty() && ctx_.gpu_history.back().vram_total
                            ? static_cast<float>(static_cast<double>(ctx_.gpu_history.back().vram_used) /
                                                 ctx_.gpu_history.back().vram_total)
                            : 0;
        {
            push_font(fonts().semibold, .86f);
            const ImVec2 p = ImGui::GetCursorScreenPos();
            const float w = ImGui::GetContentRegionAvail().x;
            char pct[16] = "";
            if (!ctx_.gpu_history.empty()) std::snprintf(pct, sizeof pct, "%.0f %%", 100.0 * f);
            const float pw = fonts().mono->CalcTextSizeA(em(.92f), FLT_MAX, 0, pct).x;
            text_ellipsis(ImGui::GetWindowDrawList(), fonts().semibold, ImGui::GetFontSize(), p, p.x + w - pw - em(.5f),
                          ImGui::GetColorU32(col::text), name.empty() ? "GPU" : name);
            ImGui::GetWindowDrawList()->AddText(fonts().mono, em(.92f), {p.x + w - pw, p.y},
                                                ImGui::GetColorU32(col::dim), pct);
            ImGui::Dummy({w, ImGui::GetTextLineHeight()});
            ImGui::PopFont();
        }
        if (!ctx_.gpu_history.empty()) {
            const auto& g = ctx_.gpu_history.back();
            cell_meter(f, load_color(f), 12, -1, em(.6f));
            push_font(fonts().mono, .74f);
            ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
            ImGui::Text("%s %s / %s", telemetry_.reports_process_memory() ? "Allocation" : "VRAM",
                        format_bytes(g.vram_used).c_str(), format_bytes(g.vram_total).c_str());
            ImGui::Text("%u %% · %u °C · %.0f W", g.gpu_util_pct, g.temperature_c, g.power_w);
            ImGui::PopStyleColor();
            ImGui::PopFont();
        } else {
            push_font(nullptr, .84f);
            const char* status = "Sampling...";
            if (!telemetry_.available()) {
#if defined(__APPLE__)
                status = "GPU telemetry unavailable";
#else
                status = "NVML unavailable";
#endif
            }
            dim_wrapped(status);
            ImGui::PopFont();
        }
        ImGui::EndChild();
        ImGui::PopStyleVar(4);
        ImGui::PopStyleColor();
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + em(.4f));
        push_font(nullptr, .86f);
        ImGui::PushStyleColor(ImGuiCol_TextLink, col::dim);
        if (ImGui::TextLink("About Waffle Iron")) about_ = true;
        ImGui::PopStyleColor();
        ImGui::PopFont();
    }
    ImGui::EndChild();
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor();
    // Separator line on the right of the bar.
    const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
    ImGui::GetWindowDrawList()->AddLine({b.x - 1, a.y}, {b.x - 1, b.y}, ImGui::GetColorU32(col::border));
}
void App::status_bar(float height) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, col::sidebar);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {em(1), 0});
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 0);
    ImGui::BeginChild("##Status", {0, height}, ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    auto* dl = ImGui::GetWindowDrawList();
    const ImVec2 wp = ImGui::GetWindowPos();
    dl->AddLine(wp, {wp.x + ImGui::GetWindowWidth(), wp.y}, ImGui::GetColorU32(col::border));
    push_font(nullptr, .86f);
    const float y = (height - ImGui::GetTextLineHeight()) * .5f;
    ImGui::SetCursorPosY(y);
    const bool idle = engine_.state() == EngineState::Idle;
    const auto status = engine_.status();
    const ImVec2 dot = ImGui::GetCursorScreenPos();
    dl->AddCircleFilled({dot.x + em(.3f), dot.y + ImGui::GetTextLineHeight() * .5f}, em(.26f),
                        ImGui::GetColorU32(idle ? col::good : col::accent));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + em(.85f));
    ImGui::TextUnformatted(status.empty() ? (idle ? "Ready" : "Working...") : status.c_str());
    if (!ctx_.gpu_history.empty()) {
        const auto& g = ctx_.gpu_history.back();
        char right[160];
        std::snprintf(right, sizeof right, "%s · %s %s / %s · %u °C", telemetry_.device_name().c_str(),
                      telemetry_.reports_process_memory() ? "Allocation" : "VRAM", format_bytes(g.vram_used).c_str(),
                      format_bytes(g.vram_total).c_str(), g.temperature_c);
        push_font(fonts().mono, .8f);
        const float w = ImGui::CalcTextSize(right).x;
        ImGui::PopFont();
        const float x = ImGui::GetWindowWidth() - w - em(1);
        if (x > ImGui::GetItemRectMax().x - wp.x + em(2)) {
            ImGui::SameLine();
            ImGui::SetCursorPos({x, y + em(.06f)});
            push_font(fonts().mono, .8f);
            ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
            ImGui::TextUnformatted(right);
            ImGui::PopStyleColor();
            ImGui::PopFont();
        }
    }
    ImGui::PopFont();
    ImGui::EndChild();
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor();
}
void App::toasts(const ImVec2& corner) {
    const double now = ImGui::GetTime();
    while (!ctx_.toasts.empty() && ctx_.toasts.front().expires_at <= now) ctx_.toasts.pop_front();
    float y = corner.y;
    int id = 0;
    for (auto it = ctx_.toasts.rbegin(); it != ctx_.toasts.rend(); ++it) {
        if (it->expires_at <= now) continue;
        ImGui::SetNextWindowPos({corner.x, y}, ImGuiCond_Always, {1, 1});
        ImGui::SetNextWindowSizeConstraints({em(14), 0}, {em(26), FLT_MAX});
        const float alpha = static_cast<float>(std::clamp(it->expires_at - now, 0.0, 1.0));
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, em(.75f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {em(1.5f), em(.8f)});
        ImGui::PushStyleColor(ImGuiCol_WindowBg, col::card_hi);
        ImGui::PushStyleColor(ImGuiCol_Border, it->error ? waffle::ui::alpha(col::bad, .45f) : col::border_hi);
        std::string name = "##Toast" + std::to_string(id++);
        ImGui::Begin(name.c_str(), nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                         ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs |
                         ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);
        const ImVec2 p = ImGui::GetWindowPos();
        const float h = ImGui::GetWindowHeight();
        ImGui::GetWindowDrawList()->AddRectFilled({p.x + em(.55f), p.y + em(.6f)}, {p.x + em(.8f), p.y + h - em(.6f)},
                                                  ImGui::GetColorU32(it->error ? col::bad : col::accent), em(.1f));
        ImGui::PushTextWrapPos(em(24));
        ImGui::TextUnformatted(it->text.c_str());
        ImGui::PopTextWrapPos();
        y -= h + em(.5f);
        ImGui::End();
        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar(4);
    }
}
void App::frame() {
    const double now = ImGui::GetTime();
    if (now - last_sample_ >= .1) {
        last_sample_ = now;
        if (auto sample = telemetry_.sample()) ctx_.gpu_history.push_back(*sample);
        while (
            ctx_.gpu_history.size() > 6000 ||
            (!ctx_.gpu_history.empty() && ctx_.gpu_history.back().t_seconds - ctx_.gpu_history.front().t_seconds > 600))
            ctx_.gpu_history.pop_front();
        while (ctx_.tps_history.size() > 6000 ||
               (!ctx_.tps_history.empty() && now - ctx_.tps_history.front().first > 600))
            ctx_.tps_history.pop_front();
    }
    for (auto& v : views_)
        if (v) {
            try {
                v->tick(ctx_);
            } catch (const std::exception& e) {
                ctx_.notify(e.what(), true);
            } catch (...) {
                ctx_.notify("Unexpected error in the view", true);
            }
        }
    for (int i = 0; i < static_cast<int>(views_.size()); ++i)
        if (ImGui::Shortcut(ImGuiMod_Ctrl | static_cast<ImGuiKey>(ImGuiKey_1 + i), ImGuiInputFlags_RouteGlobal))
            page_ = i;
    if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_R, ImGuiInputFlags_RouteGlobal) && views_[0])
        request_library_scan(*views_[0]);
    // Development screenshots: WAFFLE_SELECT=<index> preselects a library model.
    if (static const char* select = std::getenv("WAFFLE_SELECT"); select && ctx_.selected_model < 0) {
        const int index = std::atoi(select);
        if (index >= 0 && index < static_cast<int>(ctx_.library.size())) ctx_.selected_model = index;
    }
    if (ctx_.requested_page >= 0) {
        page_ = std::clamp(ctx_.requested_page, 0, static_cast<int>(views_.size()) - 1);
        ctx_.requested_page = -1;
    }

    auto* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, 0});
    ImGui::Begin("##Waffle Iron", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                     ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar(2);
    const float status_h = em(2.15f);
    const float body_h = vp->WorkSize.y - status_h;
    // Narrow window: the sidebar shrinks to icons.
    const bool compact = vp->WorkSize.x < em(62);
    const float side_w = compact ? em(4.4f) : std::clamp(vp->WorkSize.x * .17f, em(14.5f), em(16.5f));
    sidebar(side_w, body_h, compact);
    ImGui::SameLine(0, 0);

    // Main area: page header, then content.
    auto& view = *views_[static_cast<size_t>(page_)];
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {em(2.0f), em(1.6f)});
    ImGui::BeginChild("##Main", {0, body_h}, ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();
    {
        // Page header: the group in quiet capitals, the title in the display font, one line of subtitle.
        push_font(fonts().semibold, .76f);
        ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
        ImGui::TextUnformatted(page_group(page_));
        ImGui::PopStyleColor();
        ImGui::PopFont();
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - em(.3f));
        push_font(fonts().display, 1.95f);
        ImGui::TextUnformatted(view.title());
        ImGui::PopFont();
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - em(.35f));
        ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
        ImGui::TextUnformatted(view.subtitle());
        ImGui::PopStyleColor();
        // Engine-state pill, aligned to the right of the title.
        const bool idle = engine_.state() == EngineState::Idle;
        const char* state = idle ? "Engine idle" : "Engine busy";
        push_font(fonts().semibold, .86f);
        const float bw = ImGui::CalcTextSize(state).x + em(2.6f);
        ImGui::PopFont();
        if (ImGui::GetContentRegionAvail().x > bw + em(20)) {
            const ImVec2 keep = ImGui::GetCursorPos();
            ImGui::SetCursorPos({ImGui::GetWindowWidth() - ImGui::GetStyle().WindowPadding.x - bw,
                                 ImGui::GetStyle().WindowPadding.y + em(1.5f)});
            status_pill(state, idle ? col::good : col::accent);
            ImGui::SetCursorPos(keep);
        }
        ImGui::Dummy({0, em(.75f)});
    }
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
    ImGui::BeginChild("##Page", {0, 0}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground);
    ImGui::PopStyleVar();
    try {
        view.draw(ctx_);
    } catch (const std::exception& e) {
        ctx_.notify(e.what(), true);
    } catch (...) {
        ctx_.notify("Unexpected error in the view", true);
    }
    ImGui::EndChild();
    ImGui::EndChild();
    ImGui::PopStyleColor();

    ImGui::SetCursorPos({0, body_h});
    status_bar(status_h);
    ImGui::End();

    if (about_) {
        ImGui::OpenPopup("About");
        about_ = false;
    }
    ImGui::SetNextWindowPos(vp->GetCenter(), ImGuiCond_Appearing, {.5f, .5f});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {em(1.8f), em(1.6f)});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, em(1.0f));
    if (ImGui::BeginPopupModal("About", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar)) {
        {
            const ImVec2 p = ImGui::GetCursorScreenPos();
            const float s = em(2.6f);
            draw_mark(ImGui::GetWindowDrawList(), p, s);
            ImGui::SetCursorScreenPos({p.x + s + em(.8f), p.y - em(.1f)});
            ImGui::BeginGroup();
            push_font(fonts().display, 1.4f);
            ImGui::TextUnformatted("Waffle Iron");
            ImGui::PopFont();
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() - em(.3f));
            push_font(fonts().mono, .8f);
            ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
            ImGui::Text("Version %s · llama.cpp %s", WAFFLE_VERSION, WAFFLE_LLAMA_VERSION);
            ImGui::PopStyleColor();
            ImGui::PopFont();
            ImGui::EndGroup();
            ImGui::SetCursorScreenPos({p.x, std::max(p.y + s, ImGui::GetItemRectMax().y) + em(.2f)});
            ImGui::Dummy({0, 0});
        }
        ImGui::Dummy({0, em(.3f)});
        ImGui::PushTextWrapPos(em(26));
        ImGui::TextUnformatted("Understand, measure, and tune local models on your GPU.");
        ImGui::PopTextWrapPos();
        ImGui::Dummy({0, em(.35f)});
        ImGui::PushStyleColor(ImGuiCol_Text, col::soft);
        ImGui::TextUnformatted("MIT License. Copyright 2026 Anarchy.");
        ImGui::TextWrapped("Built on llama.cpp, Dear ImGui, and ImPlot. Full terms are in THIRD_PARTY.md.");
        ImGui::TextWrapped("No network. GPU readings stay on this machine.");
        ImGui::PopStyleColor();
        ImGui::Dummy({0, em(.3f)});
        ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
        ImGui::TextUnformatted("Shortcuts: Ctrl+1 through Ctrl+6 switch pages, Ctrl+R rescans models.");
        ImGui::PopStyleColor();
        ImGui::Dummy({0, em(.4f)});
        if (primary_button("Close", {em(7), 0}) || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
    ImGui::PopStyleVar(2);

    toasts({vp->WorkPos.x + vp->WorkSize.x - em(1.2f), vp->WorkPos.y + vp->WorkSize.y - status_h - em(1)});
}
} // namespace waffle::ui
