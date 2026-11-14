#pragma once
#include <imgui.h>
#include <filesystem>
namespace waffle::ui {
// Call after the ImGui and ImPlot contexts are created.
bool apply_theme(float dpi, const std::filesystem::path& assets);

// Fonts loaded by apply_theme: body text, titles (semibold), page titles (display), technical values (monospace).
struct Fonts {
    ImFont* regular = nullptr;
    ImFont* semibold = nullptr;
    ImFont* display = nullptr; // Space Grotesk when present in assets, otherwise the semibold font
    ImFont* mono = nullptr;
    ImFont* icons = nullptr; // Lucide private-use glyphs (sidebar), when assets/fonts/lucide.ttf loaded
};
const Fonts& fonts();

// "Gaufrier" palette: warm carbon ground, one syrup accent, numbers in monospace.
// Series: syrup, blueberry, mint, berry. States: mint (good), butter (warning), raspberry (error).
namespace col {
inline constexpr ImVec4 bg{0.075f, 0.067f, 0.055f, 1};           // #13110E page background
inline constexpr ImVec4 sidebar{0.059f, 0.051f, 0.043f, 1};      // #0F0D0B sidebar and status bar
inline constexpr ImVec4 card{0.106f, 0.094f, 0.078f, 1};         // #1B1814 cards
inline constexpr ImVec4 card_hi{0.141f, 0.125f, 0.102f, 1};      // #24201A tiles, secondary buttons
inline constexpr ImVec4 field{0.133f, 0.118f, 0.098f, 1};        // #221E19 inputs
inline constexpr ImVec4 border{0.180f, 0.161f, 0.133f, 1};       // #2E2922
inline constexpr ImVec4 border_hi{0.231f, 0.204f, 0.169f, 1};    // #3B342B button and control outlines
inline constexpr ImVec4 cell_off{0.165f, 0.145f, 0.118f, 1};     // #2A251E empty waffle cell
inline constexpr ImVec4 text{0.953f, 0.929f, 0.890f, 1};         // #F3EDE3
inline constexpr ImVec4 soft{0.851f, 0.812f, 0.753f, 1};         // #D9CFC0 navigation labels
inline constexpr ImVec4 dim{0.663f, 0.616f, 0.549f, 1};          // #A99D8C labels, help
inline constexpr ImVec4 faint{0.553f, 0.506f, 0.439f, 1};        // #8D8170 hints, shortcuts
inline constexpr ImVec4 accent{0.949f, 0.663f, 0.231f, 1};       // #F2A93B syrup
inline constexpr ImVec4 accent_hi{1.000f, 0.776f, 0.420f, 1};    // #FFC66B hover
inline constexpr ImVec4 accent_ink{0.102f, 0.071f, 0.027f, 1};   // #1A1207 text on syrup
inline constexpr ImVec4 accent_bg{0.949f, 0.663f, 0.231f, .12f}; // syrup wash: selection, active item
inline constexpr ImVec4 blue{0.510f, 0.659f, 0.961f, 1};         // #82A8F5
inline constexpr ImVec4 mint{0.369f, 0.827f, 0.647f, 1};         // #5ED3A5
inline constexpr ImVec4 berry{0.898f, 0.545f, 0.784f, 1};        // #E58BC8
inline constexpr ImVec4 butter{0.949f, 0.824f, 0.420f, 1};       // #F2D26B
inline constexpr ImVec4 good = mint;
inline constexpr ImVec4 warn = butter;
inline constexpr ImVec4 bad{0.949f, 0.502f, 0.561f, 1}; // #F2808F
// Names kept from the previous palette, mapped onto Gaufrier.
inline constexpr ImVec4 teal = blue;
inline constexpr ImVec4 mauve = berry;
inline constexpr ImVec4 amber = butter;
} // namespace col
// Same color with another alpha.
inline ImVec4 alpha(const ImVec4& c, float a) {
    return {c.x, c.y, c.z, a};
}
} // namespace waffle::ui
