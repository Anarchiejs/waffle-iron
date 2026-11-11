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
        case 0: // library: three book spines
            dl->AddRect({c.x - h, c.y - h * .8f}, {c.x - h * .4f, c.y + h}, color, 1.5f, 0, t);
            dl->AddRect({c.x - h * .25f, c.y - h}, {c.x + h * .35f, c.y + h}, color, 1.5f, 0, t);
            dl->AddLine({c.x + h * .55f, c.y - h * .65f}, {c.x + h, c.y + h}, color, t);
            break;
        case 1: // VRAM: a chip and its pins
            dl->AddRect({c.x - h * .62f, c.y - h * .62f}, {c.x + h * .62f, c.y + h * .62f}, color, 2.0f, 0, t);
            for (int i = -1; i <= 1; i += 2) {
                const float o = h * .3f * static_cast<float>(i);
                dl->AddLine({c.x + o, c.y - h}, {c.x + o, c.y - h * .62f}, color, t);
                dl->AddLine({c.x + o, c.y + h * .62f}, {c.x + o, c.y + h}, color, t);
                dl->AddLine({c.x - h, c.y + o}, {c.x - h * .62f, c.y + o}, color, t);
                dl->AddLine({c.x + h * .62f, c.y + o}, {c.x + h, c.y + o}, color, t);
            }
            break;
        case 2: // bench: a bar chart
            dl->AddRectFilled({c.x - h, c.y + h * .1f}, {c.x - h * .45f, c.y + h}, color, 1.0f);
            dl->AddRectFilled({c.x - h * .27f, c.y - h * .4f}, {c.x + h * .27f, c.y + h}, color, 1.0f);
            dl->AddRectFilled({c.x + h * .45f, c.y - h}, {c.x + h, c.y + h}, color, 1.0f);
            break;
        case 3: // sampler: a branch tree
            dl->AddLine({c.x - h * .55f, c.y}, {c.x + h * .55f, c.y - h * .6f}, color, t);
            dl->AddLine({c.x - h * .55f, c.y}, {c.x + h * .55f, c.y + h * .6f}, color, t);
            dl->AddCircleFilled({c.x - h * .65f, c.y}, h * .3f, color);
            dl->AddCircleFilled({c.x + h * .65f, c.y - h * .65f}, h * .3f, color);
            dl->AddCircleFilled({c.x + h * .65f, c.y + h * .65f}, h * .3f, color);
            break;
        case 4: // comparator: A | B
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
bool nav_item(int page, const char* label, bool active, bool compact) {
    const float w = ImGui::GetContentRegionAvail().x;
    const float h = em(2.35f);
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::PushID(page);
    const bool clicked = ImGui::InvisibleButton("nav", {w, h});
    const bool hovered = ImGui::IsItemHovered();
    ImGui::PopID();
    auto* dl = ImGui::GetWindowDrawList();
    if (active) {
        dl->AddRectFilled(pos, {pos.x + w, pos.y + h}, ImGui::GetColorU32(ImVec4(.23f, .25f, .44f, .75f)), em(.45f));
        dl->AddRectFilled({pos.x, pos.y + em(.55f)}, {pos.x + em(.2f), pos.y + h - em(.55f)},
                          ImGui::GetColorU32(col::accent), em(.1f));
    } else if (hovered)
        dl->AddRectFilled(pos, {pos.x + w, pos.y + h}, ImGui::GetColorU32(ImVec4(1, 1, 1, .045f)), em(.45f));
    const ImU32 icon = ImGui::GetColorU32(active ? col::accent : hovered ? col::text : col::dim);
    if (compact) {
        // Narrow bar: icon only, label in a tooltip.
        draw_nav_icon(dl, page, {pos.x + w * .5f + em(.1f), pos.y + h * .5f}, em(1.05f), icon);
        if (hovered) ImGui::SetTooltip("%s  (Ctrl+%d)", label, page + 1);
        return clicked;
    }
    draw_nav_icon(dl, page, {pos.x + em(1.35f), pos.y + h * .5f}, em(1.0f), icon);
    ImFont* font = active ? fonts().semibold : fonts().regular;
    const float fs = em(1.0f);
    dl->AddText(font, fs, {pos.x + em(2.55f), pos.y + (h - fs) * .5f - 1},
                ImGui::GetColorU32(active || hovered ? col::text : ImVec4(.78f, .80f, .86f, 1)), label);
    char key[4] = {static_cast<char>('1' + page), 0};
    const float ks = em(.78f);
    dl->AddText(fonts().regular, ks, {pos.x + w - em(1.1f), pos.y + (h - ks) * .5f}, ImGui::GetColorU32(col::faint),
                key);
    if (hovered) ImGui::SetTooltip("Ctrl+%d", page + 1);
    return clicked;
}
} // namespace
App::App(bool icons) : store_(Store::default_path()), ctx_{engine_, telemetry_, store_}, icons_(icons) {
    views_ = {make_library_view(), make_vram_view(),    make_bench_view(),
              make_sampler_view(), make_compare_view(), make_monitor_view()};
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
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {em(.75f), em(1.1f)});
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0);
    ImGui::BeginChild("##Navigation", {width, height}, ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    auto* dl = ImGui::GetWindowDrawList();
    // Mark: a "WI" badge and the name.
    {
        const ImVec2 pos = ImGui::GetCursorScreenPos();
        const float s = em(2.2f);
        const ImVec2 p{compact ? pos.x + (ImGui::GetContentRegionAvail().x - s) * .5f : pos.x + em(.35f), pos.y};
        dl->AddRectFilled(p, {p.x + s, p.y + s}, ImGui::GetColorU32(col::accent_bg), em(.55f));
        dl->AddRect(p, {p.x + s, p.y + s}, ImGui::GetColorU32(ImVec4(.58f, .62f, 1, .55f)), em(.55f), 0, 1.0f);
        const float fs = em(.95f);
        const ImVec2 tw = fonts().semibold->CalcTextSizeA(fs, FLT_MAX, 0, "WI");
        dl->AddText(fonts().semibold, fs, {p.x + (s - tw.x) * .5f, p.y + (s - tw.y) * .5f},
                    ImGui::GetColorU32(col::text), "WI");
        if (compact) {
            ImGui::SetCursorScreenPos(p);
            if (ImGui::InvisibleButton("##marque", {s, s})) about_ = true;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("About Waffle Iron");
            ImGui::SetCursorScreenPos(pos);
        }
        if (!compact)
            dl->AddText(fonts().semibold, em(1.1f), {p.x + s + em(.7f), p.y - em(.05f)},
                        ImGui::GetColorU32(col::text), "Waffle Iron");
        char version[96];
        std::snprintf(version, sizeof version, "v%s  ·  llama.cpp %s", WAFFLE_VERSION, WAFFLE_LLAMA_VERSION);
        const ImVec4 clip{pos.x, pos.y, pos.x + width - em(1), pos.y + s + 4};
        if (!compact)
            dl->AddText(fonts().regular, em(.78f), {p.x + s + em(.7f), p.y + em(1.25f)},
                        ImGui::GetColorU32(col::dim), version, nullptr, 0, &clip);
        ImGui::Dummy({0, s + em(.9f)});
    }
    if (!compact) {
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + em(.4f));
        section_header("TOOLS");
    }
    ImGui::Dummy({0, em(.05f)});
    for (int i = 0; i < static_cast<int>(views_.size()); ++i)
        if (nav_item(i, views_[static_cast<size_t>(i)]->nav_title(), i == page_, compact)) page_ = i;
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
                push_font(nullptr, .78f);
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() +
                                     (ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(text).x) * .5f);
                ImGui::TextDisabled("%s", text);
                ImGui::PopFont();
                meter(f, f > .9f ? col::bad : f > .75f ? col::amber : col::accent, -1, em(.32f));
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("VRAM %s / %s", format_bytes(g.vram_used).c_str(),
                                      format_bytes(g.vram_total).c_str());
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();
        const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
        ImGui::GetWindowDrawList()->AddLine({b.x - 1, a.y}, {b.x - 1, b.y}, ImGui::GetColorU32(col::border));
        return;
    }

    // GPU card at the bottom of the sidebar.
    const float card_h = em(7.2f);
    const float bottom = ImGui::GetWindowHeight() - ImGui::GetStyle().WindowPadding.y;
    if (ImGui::GetCursorPosY() + em(1) < bottom - card_h) {
        ImGui::SetCursorPosY(bottom - card_h);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, col::card);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {em(.85f), em(.75f)});
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, em(.6f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {em(.5f), em(.32f)});
        ImGui::BeginChild("##CarteGPU", {0, card_h - em(1.6f)}, ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        push_font(fonts().semibold, .86f);
        const auto name = telemetry_.device_name();
        ImGui::PushStyleColor(ImGuiCol_Text, col::text);
        ImGui::TextUnformatted(name.empty() ? "GPU" : name.c_str());
        ImGui::PopStyleColor();
        ImGui::PopFont();
        if (!ctx_.gpu_history.empty()) {
            const auto& g = ctx_.gpu_history.back();
            const float f = g.vram_total ? static_cast<float>(static_cast<double>(g.vram_used) / g.vram_total) : 0;
            meter(f, f > .9f ? col::bad : f > .75f ? col::amber : col::accent, -1, em(.38f));
            push_font(nullptr, .84f);
            ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
            ImGui::Text("VRAM %s / %s", format_bytes(g.vram_used).c_str(), format_bytes(g.vram_total).c_str());
            ImGui::Text("%u %%  ·  %u °C  ·  %.0f W", g.gpu_util_pct, g.temperature_c, g.power_w);
            ImGui::PopStyleColor();
            ImGui::PopFont();
        } else {
            push_font(nullptr, .84f);
            dim_wrapped(telemetry_.available() ? "Sampling..." : "NVML indisponible");
            ImGui::PopFont();
        }
        ImGui::EndChild();
        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor();
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + em(.4f));
        push_font(nullptr, .86f);
        ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
        if (ImGui::TextLink("About Waffle Iron")) about_ = true;
        ImGui::PopStyleColor();
        ImGui::PopFont();
    }
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
    // Separator line on the right of the bar.
    const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
    ImGui::GetWindowDrawList()->AddLine({b.x - 1, a.y}, {b.x - 1, b.y}, ImGui::GetColorU32(col::border));
}
void App::status_bar(float height) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, col::sidebar);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {em(1), 0});
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0);
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
    dl->AddCircleFilled({dot.x + em(.3f), dot.y + ImGui::GetTextLineHeight() * .5f}, em(.24f),
                        ImGui::GetColorU32(idle ? col::good : col::amber));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + em(.85f));
    ImGui::TextUnformatted(status.empty() ? (idle ? "Ready" : "Working...") : status.c_str());
    if (!ctx_.gpu_history.empty()) {
        const auto& g = ctx_.gpu_history.back();
        char right[160];
        std::snprintf(right, sizeof right, "%s   ·   VRAM %s / %s   ·   %u °C", telemetry_.device_name().c_str(),
                      format_bytes(g.vram_used).c_str(), format_bytes(g.vram_total).c_str(), g.temperature_c);
        const float w = ImGui::CalcTextSize(right).x;
        const float x = ImGui::GetWindowWidth() - w - em(1);
        if (x > ImGui::GetItemRectMax().x - wp.x + em(2)) {
            ImGui::SameLine();
            ImGui::SetCursorPos({x, y});
            ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
            ImGui::TextUnformatted(right);
            ImGui::PopStyleColor();
        }
    }
    ImGui::PopFont();
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
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
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, em(.6f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {em(1.3f), em(.75f)});
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(.115f, .125f, .155f, 1));
        std::string name = "##Toast" + std::to_string(id++);
        ImGui::Begin(name.c_str(), nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDocking |
                         ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoFocusOnAppearing |
                         ImGuiWindowFlags_NoNav);
        const ImVec2 p = ImGui::GetWindowPos();
        const float h = ImGui::GetWindowHeight();
        ImGui::GetWindowDrawList()->AddRectFilled({p.x + em(.35f), p.y + em(.55f)}, {p.x + em(.55f), p.y + h - em(.55f)},
                                                  ImGui::GetColorU32(it->error ? col::bad : col::accent), em(.1f));
        ImGui::PushTextWrapPos(em(24));
        ImGui::TextUnformatted(it->text.c_str());
        ImGui::PopTextWrapPos();
        y -= h + em(.5f);
        ImGui::End();
        ImGui::PopStyleColor();
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
                     ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar(2);
    const float status_h = em(1.95f);
    const float body_h = vp->WorkSize.y - status_h;
    // Narrow window: the sidebar shrinks to icons.
    const bool compact = vp->WorkSize.x < em(62);
    const float side_w = compact ? em(4.2f) : std::clamp(vp->WorkSize.x * .17f, em(13.5f), em(16));
    sidebar(side_w, body_h, compact);
    ImGui::SameLine(0, 0);

    // Main area: page header, then content.
    auto& view = *views_[static_cast<size_t>(page_)];
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {em(1.6f), em(1.2f)});
    ImGui::BeginChild("##Principal", {0, body_h}, ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();
    {
        push_font(fonts().semibold, 1.5f);
        ImGui::TextUnformatted(view.title());
        ImGui::PopFont();
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - em(.35f));
        ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
        ImGui::TextUnformatted(view.subtitle());
        ImGui::PopStyleColor();
        // Engine-state badge, aligned to the right of the title.
        const bool idle = engine_.state() == EngineState::Idle;
        const char* state = idle ? "Engine idle" : "Engine busy";
        const float bw = ImGui::CalcTextSize(state).x + em(.9f);
        if (ImGui::GetContentRegionAvail().x > bw + em(20)) {
            const ImVec2 keep = ImGui::GetCursorPos();
            ImGui::SetCursorPos({ImGui::GetWindowWidth() - ImGui::GetStyle().WindowPadding.x - bw,
                                 ImGui::GetStyle().WindowPadding.y + em(.45f)});
            badge(state, idle ? col::good : col::amber);
            ImGui::SetCursorPos(keep);
        }
        ImGui::Dummy({0, em(.55f)});
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
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {em(1.5f), em(1.2f)});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, em(.7f));
    if (ImGui::BeginPopupModal("About", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar)) {
        push_font(fonts().semibold, 1.3f);
        ImGui::TextUnformatted("Waffle Iron");
        ImGui::PopFont();
        ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
        ImGui::Text("Version %s  ·  llama.cpp %s", WAFFLE_VERSION, WAFFLE_LLAMA_VERSION);
        ImGui::PopStyleColor();
        ImGui::Dummy({0, em(.3f)});
        ImGui::PushTextWrapPos(em(26));
        ImGui::TextUnformatted("Understand, measure, and tune local models on your GPU.");
        ImGui::PopTextWrapPos();
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
