#include "ui/theme.hpp"
#include <implot.h>
#include <cstdlib>
#include <vector>
namespace waffle::ui {
namespace {
Fonts g_fonts;
// First existing file among the candidates (assets, then platform fonts).
std::filesystem::path first_existing(const std::vector<std::filesystem::path>& candidates) {
    for (const auto& p : candidates)
        if (!p.empty() && std::filesystem::exists(p)) return p;
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

    // Fonts: Inter from assets when present, otherwise a platform UI font, then ImGui's built-in font.
    // Windows: Segoe UI, Cascadia Mono / Consolas. Linux: DejaVu or Liberation. macOS: Arial / Courier New.
    auto& io = ImGui::GetIO();
    const float size = 15.5f * dpi;
    const auto win = windows_fonts();
    std::vector<std::filesystem::path> regular_fonts = {assets / "fonts/Inter-Regular.ttf", win / "segoeui.ttf"};
    std::vector<std::filesystem::path> semibold_fonts = {assets / "fonts/Inter-SemiBold.ttf", win / "seguisb.ttf"};
    std::vector<std::filesystem::path> mono_fonts = {win / "CascadiaMono.ttf", win / "consola.ttf"};
#if defined(__APPLE__)
    regular_fonts.insert(regular_fonts.end(),
                         {"/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"});
    semibold_fonts.insert(semibold_fonts.end(),
                          {"/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf"});
    mono_fonts.insert(mono_fonts.end(),
                      {"/System/Library/Fonts/Supplemental/Courier New.ttf", "/Library/Fonts/Courier New.ttf"});
#elif !defined(_WIN32)
    regular_fonts.insert(regular_fonts.end(),
                         {"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                          "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                          "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
                          "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"});
    semibold_fonts.insert(semibold_fonts.end(), {"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                                 "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
                                                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                                                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"});
    mono_fonts.insert(mono_fonts.end(), {"/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                                         "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
                                         "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                                         "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf"});
#endif
    const auto regular = first_existing(regular_fonts);
    const auto semibold = first_existing(semibold_fonts);
    const auto mono = first_existing(mono_fonts);
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
        // Only the private-use icons. Merging the whole file is unnecessary, and a
        // font that also covered ASCII would replace digits in the UI.
        static const ImWchar icon_ranges[] = {
            0xE038, 0xE038, // activity
            0xE098, 0xE098, // columns-2
            0xE0A9, 0xE0A9, // cpu
            0xE0E2, 0xE0E2, // git-branch
            0xE100, 0xE100, // library
            0xE1BF, 0xE1BF, // gauge
            0,
        };
        ImFontConfig cfg;
        cfg.MergeMode = true;
        cfg.PixelSnapH = true;
        cfg.GlyphOffset.y = 0.5f * dpi;
        merged = io.Fonts->AddFontFromFileTTF(icons.string().c_str(), size, &cfg, icon_ranges) != nullptr;
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
