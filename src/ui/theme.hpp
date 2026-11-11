#pragma once
#include <imgui.h>
#include <filesystem>
namespace waffle::ui {
// Call after the ImGui and ImPlot contexts are created.
bool apply_theme(float dpi, const std::filesystem::path& assets);

// Fonts loaded by apply_theme: body text, titles (semibold), technical values (monospace).
struct Fonts {
    ImFont* regular = nullptr;
    ImFont* semibold = nullptr;
    ImFont* mono = nullptr;
};
const Fonts& fonts();

// Art-direction palette: slate background, indigo accent, lavender / teal / mauve / amber series.
namespace col {
inline constexpr ImVec4 bg{0.078f, 0.086f, 0.102f, 1};       // page background
inline constexpr ImVec4 sidebar{0.060f, 0.066f, 0.080f, 1};  // sidebar and status bar
inline constexpr ImVec4 card{0.102f, 0.112f, 0.137f, 1};     // cards
inline constexpr ImVec4 card_hi{0.130f, 0.142f, 0.175f, 1};  // tiles, hover
inline constexpr ImVec4 border{0.170f, 0.185f, 0.230f, 1};
inline constexpr ImVec4 text{0.905f, 0.915f, 0.945f, 1};
inline constexpr ImVec4 dim{0.560f, 0.585f, 0.660f, 1};
inline constexpr ImVec4 faint{0.380f, 0.400f, 0.460f, 1};
inline constexpr ImVec4 accent{0.58f, 0.62f, 1.00f, 1};      // indigo lavender
inline constexpr ImVec4 accent_bg{0.25f, 0.28f, 0.53f, 1};   // primary buttons
inline constexpr ImVec4 teal{0.36f, 0.78f, 0.78f, 1};
inline constexpr ImVec4 mauve{0.85f, 0.63f, 0.94f, 1};
inline constexpr ImVec4 amber{0.94f, 0.73f, 0.43f, 1};
inline constexpr ImVec4 good{0.35f, 0.85f, 0.62f, 1};
inline constexpr ImVec4 bad{1.00f, 0.50f, 0.50f, 1};
} // namespace col
} // namespace waffle::ui
