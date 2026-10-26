#include "ui/theme.hpp"
#include <imgui.h>
#include <implot.h>
namespace banc::ui {
bool apply_theme(float dpi, const std::filesystem::path& assets) {
    ImGui::StyleColorsDark();
    auto& s = ImGui::GetStyle();
    s.WindowRounding = 7;
    s.FrameRounding = 5;
    s.GrabRounding = 5;
    s.PopupRounding = 7;
    s.TabRounding = 5;
    s.WindowPadding = {14, 12};
    s.FramePadding = {9, 6};
    s.ItemSpacing = {9, 8};
    s.Colors[ImGuiCol_WindowBg] = {0.078f, 0.086f, 0.102f, 1};
    s.Colors[ImGuiCol_ChildBg] = {0.09f, 0.10f, 0.12f, 1};
    s.Colors[ImGuiCol_Header] = {0.23f, 0.25f, 0.44f, 1};
    s.Colors[ImGuiCol_HeaderHovered] = {0.34f, 0.37f, 0.65f, 1};
    s.Colors[ImGuiCol_Button] = {0.25f, 0.28f, 0.53f, 1};
    s.Colors[ImGuiCol_ButtonHovered] = {0.39f, 0.43f, 0.76f, 1};
    s.Colors[ImGuiCol_CheckMark] = s.Colors[ImGuiCol_SliderGrab] = {0.58f, 0.62f, 1, 1};
    s.ScaleAllSizes(dpi);
    ImPlot::StyleColorsDark();
    static const ImVec4 plot_colors[] = {
        {.58f, .62f, 1, 1}, {.36f, .78f, .78f, 1}, {.85f, .63f, .94f, 1}, {.94f, .73f, .43f, 1}};
    ImPlot::GetStyle().Colormap = ImPlot::AddColormap("Banc Local", plot_colors, 4);
    ImPlot::GetStyle().PlotPadding = {12 * dpi, 12 * dpi};
    auto& io = ImGui::GetIO();
    static const ImWchar ranges[] = {0x20, 0xff, 0x152, 0x153, 0x2010, 0x2026, 0};
    auto regular = assets / "fonts/Inter-Regular.ttf";
    bool custom = false;
    if (std::filesystem::exists(regular))
        custom = io.Fonts->AddFontFromFileTTF(regular.string().c_str(), 16 * dpi, nullptr, ranges) != nullptr;
    if (!custom) {
        ImFontConfig c;
        c.SizePixels = 16 * dpi;
        io.Fonts->AddFontDefault(&c);
    }
    const auto icons = assets / "fonts/lucide.ttf";
    bool merged = false;
    if (std::filesystem::exists(icons)) {
        static const ImWchar icon_ranges[] = {0xe000, 0xf8ff, 0};
        ImFontConfig c;
        c.MergeMode = true;
        c.PixelSnapH = true;
        merged = io.Fonts->AddFontFromFileTTF(icons.string().c_str(), 16 * dpi, &c, icon_ranges) != nullptr;
    }
    const auto bold = assets / "fonts/Inter-SemiBold.ttf";
    if (std::filesystem::exists(bold)) io.Fonts->AddFontFromFileTTF(bold.string().c_str(), 16 * dpi, nullptr, ranges);
    return merged;
}
} // namespace banc::ui
