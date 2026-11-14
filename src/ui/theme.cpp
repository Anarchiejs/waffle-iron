#include "ui/theme.hpp"

#include "core/util.hpp"
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
    s.ChildRounding = 14;
    s.FrameRounding = 9;
    s.GrabRounding = 7;
    s.PopupRounding = 12;
    s.TabRounding = 8;
    s.ScrollbarRounding = 8;
    s.WindowPadding = {18, 16};
    s.FramePadding = {12, 9};
    s.ItemSpacing = {10, 10};
    s.ItemInnerSpacing = {8, 6};
    s.CellPadding = {10, 7};
    s.ScrollbarSize = 11;
    s.GrabMinSize = 14;
    s.WindowBorderSize = 0;
    s.ChildBorderSize = 1;
    s.PopupBorderSize = 1;
    s.FrameBorderSize = 1;
    s.TabBarBorderSize = 1;
    s.TabBarOverlineSize = 0;
    s.SeparatorTextBorderSize = 1;
    s.IndentSpacing = 18;
    auto* c = s.Colors;
    c[ImGuiCol_Text] = col::text;
    c[ImGuiCol_TextDisabled] = col::dim;
    c[ImGuiCol_WindowBg] = col::bg;
    c[ImGuiCol_ChildBg] = col::card;
    c[ImGuiCol_PopupBg] = {0.122f, 0.108f, 0.090f, 0.99f};
    c[ImGuiCol_Border] = col::border;
    c[ImGuiCol_BorderShadow] = {0, 0, 0, 0};
    c[ImGuiCol_FrameBg] = col::field;
    c[ImGuiCol_FrameBgHovered] = {0.165f, 0.147f, 0.122f, 1};
    c[ImGuiCol_FrameBgActive] = {0.188f, 0.167f, 0.137f, 1};
    c[ImGuiCol_TitleBg] = c[ImGuiCol_TitleBgActive] = c[ImGuiCol_TitleBgCollapsed] = col::sidebar;
    c[ImGuiCol_MenuBarBg] = col::sidebar;
    c[ImGuiCol_ScrollbarBg] = {0, 0, 0, 0};
    c[ImGuiCol_ScrollbarGrab] = col::border_hi;
    c[ImGuiCol_ScrollbarGrabHovered] = {0.300f, 0.267f, 0.220f, 1};
    c[ImGuiCol_ScrollbarGrabActive] = col::accent;
    c[ImGuiCol_CheckMark] = col::accent;
    // Integer sliders with few steps get a wide grab: translucent, so the value printed over it stays readable.
    c[ImGuiCol_SliderGrab] = alpha(col::accent, .42f);
    c[ImGuiCol_SliderGrabActive] = alpha(col::accent, .62f);
    c[ImGuiCol_Button] = col::card_hi;
    c[ImGuiCol_ButtonHovered] = {0.188f, 0.167f, 0.137f, 1};
    c[ImGuiCol_ButtonActive] = {0.231f, 0.204f, 0.169f, 1};
    c[ImGuiCol_Header] = alpha(col::accent, .14f);
    c[ImGuiCol_HeaderHovered] = alpha(col::accent, .08f);
    c[ImGuiCol_HeaderActive] = alpha(col::accent, .20f);
    c[ImGuiCol_Separator] = c[ImGuiCol_SeparatorHovered] = col::border;
    c[ImGuiCol_SeparatorActive] = col::accent;
    c[ImGuiCol_ResizeGrip] = {0, 0, 0, 0};
    c[ImGuiCol_ResizeGripHovered] = alpha(col::accent, .4f);
    c[ImGuiCol_ResizeGripActive] = col::accent;
    c[ImGuiCol_Tab] = {0, 0, 0, 0};
    c[ImGuiCol_TabHovered] = alpha(col::accent, .08f);
    c[ImGuiCol_TabSelected] = col::card_hi;
    c[ImGuiCol_TabSelectedOverline] = col::accent;
    c[ImGuiCol_TabDimmed] = {0, 0, 0, 0};
    c[ImGuiCol_TabDimmedSelected] = col::card_hi;
    c[ImGuiCol_TabDimmedSelectedOverline] = {0, 0, 0, 0};
    c[ImGuiCol_TableHeaderBg] = {0, 0, 0, 0};
    c[ImGuiCol_TableBorderStrong] = col::border;
    c[ImGuiCol_TableBorderLight] = {0.149f, 0.129f, 0.098f, 1};
    c[ImGuiCol_TableRowBg] = {0, 0, 0, 0};
    c[ImGuiCol_TableRowBgAlt] = {1, 1, 1, 0.018f};
    c[ImGuiCol_PlotHistogram] = col::accent;
    c[ImGuiCol_PlotHistogramHovered] = col::accent_hi;
    c[ImGuiCol_TextLink] = col::accent;
    c[ImGuiCol_TextSelectedBg] = alpha(col::accent, .30f);
    c[ImGuiCol_DragDropTarget] = col::accent;
    c[ImGuiCol_NavCursor] = col::accent;
    c[ImGuiCol_ModalWindowDimBg] = {0.02f, 0.016f, 0.012f, 0.62f};
    s.ScaleAllSizes(dpi);

    ImPlot::StyleColorsDark();
    auto& ps = ImPlot::GetStyle();
    // Series order: blueberry first so prefill (pp) and generation (tg) read blue then syrup, as in the design.
    static const ImVec4 plot_colors[] = {col::blue, col::accent, col::mint, col::berry, col::butter};
    static int waffle_colormap = -1;
    if (waffle_colormap < 0) waffle_colormap = ImPlot::AddColormap("Gaufrier", plot_colors, 5);
    ps.Colormap = waffle_colormap;
    ps.PlotPadding = {10 * dpi, 10 * dpi};
    ps.LabelPadding = {6 * dpi, 4 * dpi};
    ps.LegendPadding = {8 * dpi, 6 * dpi};
    ps.LegendInnerPadding = {8 * dpi, 6 * dpi};
    ps.LegendSpacing = {14 * dpi, 4 * dpi};
    ps.PlotMinSize = {120 * dpi, 90 * dpi};
    ps.PlotBorderSize = 0;
    ps.Colors[ImPlotCol_FrameBg] = {0, 0, 0, 0};
    ps.Colors[ImPlotCol_PlotBg] = {0.086f, 0.075f, 0.059f, 1};
    ps.Colors[ImPlotCol_LegendBg] = {0.122f, 0.108f, 0.090f, 0.94f};
    ps.Colors[ImPlotCol_LegendBorder] = col::border;
    ps.Colors[ImPlotCol_PlotBorder] = col::border;
    ps.Colors[ImPlotCol_LegendText] = col::soft;
    ps.Colors[ImPlotCol_AxisText] = col::dim;
    ps.Colors[ImPlotCol_AxisGrid] = {1, 0.95f, 0.85f, 0.06f};
    ps.Colors[ImPlotCol_AxisTick] = {1, 0.95f, 0.85f, 0.10f};

    // Fonts: Inter from assets when present, otherwise a platform UI font, then ImGui's built-in font.
    // Windows: Segoe UI, Cascadia Mono / Consolas. Linux: DejaVu or Liberation. macOS: Arial / Courier New.
    // Optional in assets as well: Space Grotesk (page titles) and JetBrains Mono (values).
    auto& io = ImGui::GetIO();
    io.Fonts->Clear();
    g_fonts = {}; // the atlas was cleared: no pointer from a previous build survives
    const float size = 15.5f * dpi;
    auto add_ttf = [&](const std::filesystem::path& file, float pixels, ImFontConfig* config = nullptr,
                       const ImWchar* ranges = nullptr) -> ImFont* {
        if (file.empty()) return nullptr;
        const std::string utf8 = waffle::narrow_utf8(file);
        return io.Fonts->AddFontFromFileTTF(utf8.c_str(), pixels, config, ranges);
    };
    const auto win = windows_fonts();
    std::vector<std::filesystem::path> regular_fonts = {assets / "fonts/Inter-Regular.ttf", win / "segoeui.ttf"};
    std::vector<std::filesystem::path> semibold_fonts = {assets / "fonts/Inter-SemiBold.ttf", win / "seguisb.ttf"};
    std::vector<std::filesystem::path> mono_fonts = {assets / "fonts/JetBrainsMono-Regular.ttf",
                                                     win / "CascadiaMono.ttf", win / "consola.ttf"};
    const std::vector<std::filesystem::path> display_fonts = {assets / "fonts/SpaceGrotesk-SemiBold.ttf",
                                                              assets / "fonts/SpaceGrotesk-Medium.ttf"};
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
    const auto display = first_existing(display_fonts);
    if (!regular.empty()) g_fonts.regular = add_ttf(regular, size);
    if (!g_fonts.regular) {
        ImFontConfig cfg;
        cfg.SizePixels = size;
        g_fonts.regular = io.Fonts->AddFontDefault(&cfg);
    }
    // Symbols for invisible tokens (␣ ↵ ␍ ⇥) and arrows, merged into the text and monospace fonts: Inter and
    // JetBrains Mono do not carry the control pictures. Segoe UI Symbol on Windows, DejaVu elsewhere.
    std::vector<std::filesystem::path> symbol_fonts = {win / "seguisym.ttf"};
#if defined(__APPLE__)
    symbol_fonts.insert(symbol_fonts.end(), {"/System/Library/Fonts/Apple Symbols.ttf",
                                             "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"});
#elif !defined(_WIN32)
    symbol_fonts.insert(symbol_fonts.end(),
                        {"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                         "/usr/share/fonts/noto/NotoSansSymbols2-Regular.ttf"});
#endif
    const auto symbols = first_existing(symbol_fonts);
    static const ImWchar symbol_ranges[] = {0x2190, 0x21FF, 0x2212, 0x2212, 0x2400, 0x243F, 0};
    auto merge_symbols = [&] {
        if (symbols.empty()) return;
        ImFontConfig cfg;
        cfg.MergeMode = true;
        add_ttf(symbols, size, &cfg, symbol_ranges);
    };
    merge_symbols();
    bool merged = false;
    const auto icons = assets / "fonts/lucide.ttf";
    if (std::filesystem::exists(icons)) {
        // Only the private-use icons, in a font of their own: Inter also has private-use glyphs, and in a
        // merged font the first source wins, so the sidebar would show Inter's symbols instead of Lucide's.
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
        cfg.PixelSnapH = true;
        cfg.GlyphOffset.y = 0.5f * dpi;
        g_fonts.icons = add_ttf(icons, size, &cfg, icon_ranges);
        merged = g_fonts.icons != nullptr;
    }
    if (!semibold.empty()) g_fonts.semibold = add_ttf(semibold, size);
    if (!g_fonts.semibold) g_fonts.semibold = g_fonts.regular;
    if (!mono.empty()) g_fonts.mono = add_ttf(mono, size);
    if (g_fonts.mono) merge_symbols();
    if (!g_fonts.mono) g_fonts.mono = g_fonts.regular;
    if (!display.empty()) g_fonts.display = add_ttf(display, size);
    if (!g_fonts.display) g_fonts.display = g_fonts.semibold;
    io.FontDefault = g_fonts.regular;
    s.FontSizeBase = size;
    return merged;
}
} // namespace waffle::ui
