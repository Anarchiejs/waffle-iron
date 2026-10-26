#pragma once
#include <filesystem>
namespace banc::ui {
// À appeler après la création des contextes ImGui et ImPlot.
bool apply_theme(float dpi, const std::filesystem::path& assets);
} // namespace banc::ui
