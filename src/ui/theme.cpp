#include "ui/theme.hpp"
#include <implot.h>
#include <cstdlib>
namespace waffle::ui {
namespace {
Fonts g_fonts;
// First existing file among the candidates (assets, then Windows fonts).
std::filesystem::path first_existing(std::initializer_list<std::filesystem::path> candidates) {
    for (const auto& p : candidates)
        if (std::filesystem::exists(p)) return p;
    return {};
}
std::filesystem::path windows_fonts() {
    const char* windir = std::getenv("WINDIR");
    return std::filesystem::path(windir ? windir : "C:\\Windows") / "Fonts";
}
} // namespace
const Fonts& fonts() {
    return g_fonts;
}
bool apply_theme(float dpi, const std::filesystem::path& assets) {
    ImGui::StyleColorsDark();
    auto& s = ImGui::GetStyle();
    s.WindowRounding = 0;
    s.ChildRounding = 10;
    s.FrameRounding = 6;
    s.GrabRounding = 6;
    s.PopupRounding = 8;
    s.TabRounding = 6;
    s.ScrollbarRounding = 8;
    s.WindowPadding = {16, 14};
    s.FramePadding = {10, 7};
    s.ItemSpacing = {10, 9};
    s.ItemInnerSpacing = {8, 6};
    s.CellPadding = {10, 6};
    s.ScrollbarSize = 11;
    s.GrabMinSize = 12;
    s.WindowBorderSize = 0;
    s.ChildBorderSize = 1;
    s.PopupBorderSize = 1;
    s.FrameBorderSize = 0;
    s.TabBarBorderSize = 1;
    s.SeparatorTextBorderSize = 1;
    s.IndentSpacing = 18;
    auto* c = s.Colors;
    c[ImGuiCol_Text] = col::text;
    c[ImGuiCol_TextDisabled] = col::dim;
    c[ImGuiCol_WindowBg] = col::bg;
    c[ImGuiCol_ChildBg] = col::card;
    c[ImGuiCol_PopupBg] = {0.115f, 0.125f, 0.155f, 0.98f};
    c[ImGuiCol_Border] = col::border;
    c[ImGuiCol_FrameBg] = {0.140f, 0.153f, 0.190f, 1};
    c[ImGuiCol_FrameBgHovered] = {0.175f, 0.190f, 0.240f, 1};
    c[ImGuiCol_FrameBgActive] = {0.200f, 0.218f, 0.290f, 1};
    c[ImGuiCol_TitleBg] = c[ImGuiCol_TitleBgActive] = c[ImGuiCol_TitleBgCollapsed] = col::sidebar;
    c[ImGuiCol_MenuBarBg] = col::sidebar;
    c[ImGuiCol_ScrollbarBg] = {0, 0, 0, 0};
    c[ImGuiCol_ScrollbarGrab] = {0.22f, 0.24f, 0.30f, 1};
    c[ImGuiCol_ScrollbarGrabHovered] = {0.30f, 0.33f, 0.42f, 1};
    c[ImGuiCol_ScrollbarGrabActive] = col::accent_bg;
    c[ImGuiCol_CheckMark] = c[ImGuiCol_SliderGrab] = col::accent;
    c[ImGuiCol_SliderGrabActive] = {0.70f, 0.73f, 1, 1};
    c[ImGuiCol_Button] = {0.165f, 0.180f, 0.225f, 1};
    c[ImGuiCol_ButtonHovered] = {0.215f, 0.235f, 0.300f, 1};
    c[ImGuiCol_ButtonActive] = {0.255f, 0.280f, 0.360f, 1};
    c[ImGuiCol_Header] = {0.23f, 0.25f, 0.44f, 0.85f};
    c[ImGuiCol_HeaderHovered] = {0.20f, 0.22f, 0.34f, 1};
    c[ImGuiCol_HeaderActive] = {0.34f, 0.37f, 0.65f, 1};
    c[ImGuiCol_Separator] = c[ImGuiCol_SeparatorHovered] = col::border;
    c[ImGuiCol_SeparatorActive] = col::accent;
    c[ImGuiCol_Tab] = {0, 0, 0, 0};
    c[ImGuiCol_TabHovered] = {0.20f, 0.22f, 0.34f, 1};
    c[ImGuiCol_TabSelected] = {0.23f, 0.25f, 0.44f, 1};
    c[ImGuiCol_TabSelectedOverline] = col::accent;
    c[ImGuiCol_TabDimmed] = {0, 0, 0, 0};
    c[ImGuiCol_TabDimmedSelected] = {0.18f, 0.20f, 0.32f, 1};
    c[ImGuiCol_TableHeaderBg] = {0.120f, 0.131f, 0.162f, 1};
    c[ImGuiCol_TableBorderStrong] = col::border;
    c[ImGuiCol_TableBorderLight] = {0.140f, 0.152f, 0.188f, 1};
    c[ImGuiCol_TableRowBg] = {0, 0, 0, 0};
    c[ImGuiCol_TableRowBgAlt] = {1, 1, 1, 0.022f};
    c[ImGuiCol_PlotHistogram] = col::accent_bg;
    c[ImGuiCol_PlotHistogramHovered] = col::accent;
    c[ImGuiCol_TextSelectedBg] = {0.34f, 0.37f, 0.65f, 0.55f};
    c[ImGuiCol_NavCursor] = col::accent;
    c[ImGuiCol_ModalWindowDimBg] = {0, 0, 0, 0.55f};
    s.ScaleAllSizes(dpi);

    ImPlot::StyleColorsDark();
    auto& ps = ImPlot::GetStyle();
    static const ImVec4 plot_colors[] = {col::accent, col::teal, col::mauve, col::amber};
    ps.Colormap = ImPlot::AddColormap("Waffle Iron", plot_colors, 4);
    ps.PlotPadding = {10 * dpi, 10 * dpi};
    ps.LabelPadding = {6 * dpi, 4 * dpi};
    ps.LegendPadding = {8 * dpi, 6 * dpi};
    ps.LegendInnerPadding = {8 * dpi, 6 * dpi};
    ps.LegendSpacing = {14 * dpi, 4 * dpi};
    ps.PlotMinSize = {120 * dpi, 90 * dpi};
    ps.PlotBorderSize = 0;
    ps.Colors[ImPlotCol_FrameBg] = {0, 0, 0, 0};
    ps.Colors[ImPlotCol_PlotBg] = {0.075f, 0.083f, 0.100f, 1};
    ps.Colors[ImPlotCol_LegendBg] = {0.115f, 0.125f, 0.155f, 0.92f};
    ps.Colors[ImPlotCol_LegendBorder] = col::border;
    ps.Colors[ImPlotCol_AxisText] = col::dim;
    ps.Colors[ImPlotCol_AxisGrid] = {1, 1, 1, 0.06f};
    ps.Colors[ImPlotCol_AxisTick] = {1, 1, 1, 0.10f};

    // Fonts: Inter from assets when present, otherwise Segoe UI shipped with Windows; Cascadia Mono / Consolas for monospace.
    auto& io = ImGui::GetIO();
    const float size = 15.5f * dpi;
    const auto win = windows_fonts();
    const auto regular = first_existing({assets / "fonts/Inter-Regular.ttf", win / "segoeui.ttf"});
    const auto semibold = first_existing({assets / "fonts/Inter-SemiBold.ttf", win / "seguisb.ttf"});
    const auto mono = first_existing({win / "CascadiaMono.ttf", win / "consola.ttf"});
    if (!regular.empty()) g_fonts.regular = io.Fonts->AddFontFromFileTTF(regular.string().c_str(), size);
    if (!g_fonts.regular) {
        ImFontConfig cfg;
        cfg.SizePixels = size;
        g_fonts.regular = io.Fonts->AddFontDefault(&cfg);
    }
    // Symbols for invisible tokens (␣ ↵ ⇥) and arrows: Segoe UI Symbol merged in.
    if (const auto symbols = win / "seguisym.ttf"; std::filesystem::exists(symbols)) {
        ImFontConfig cfg;
        cfg.MergeMode = true;
        io.Fonts->AddFontFromFileTTF(symbols.string().c_str(), size, &cfg);
    }
    bool merged = false;
    const auto icons = assets / "fonts/lucide.ttf";
    if (std::filesystem::exists(icons)) {
        ImFontConfig cfg;
        cfg.MergeMode = true;
        cfg.PixelSnapH = true;
        cfg.GlyphOffset.y = 2 * dpi;
        merged = io.Fonts->AddFontFromFileTTF(icons.string().c_str(), size, &cfg) != nullptr;
    }
    if (!semibold.empty()) g_fonts.semibold = io.Fonts->AddFontFromFileTTF(semibold.string().c_str(), size);
    if (!g_fonts.semibold) g_fonts.semibold = g_fonts.regular;
    if (!mono.empty()) g_fonts.mono = io.Fonts->AddFontFromFileTTF(mono.string().c_str(), size);
    if (!g_fonts.mono) g_fonts.mono = g_fonts.regular;
    io.FontDefault = g_fonts.regular;
    s.FontSizeBase = size;
    return merged;
}
} // namespace waffle::ui
