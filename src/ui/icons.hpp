#pragma once

// Private-use codepoints in assets/fonts/lucide.ttf (Lucide 0.544.0).
// The sidebar draws these only after that font has been merged. Without it, a
// private-use character is a missing glyph (often a box that looks like "00").
namespace waffle::ui {

inline constexpr const char* icon_library = "\uE100"; // library
inline constexpr const char* icon_vram = "\uE0A9";    // cpu
inline constexpr const char* icon_bench = "\uE1BF";   // gauge
inline constexpr const char* icon_sampler = "\uE0E2"; // git-branch
inline constexpr const char* icon_compare = "\uE098"; // columns-2
inline constexpr const char* icon_monitor = "\uE038"; // activity

} // namespace waffle::ui
