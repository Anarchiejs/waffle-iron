// Native folder picker and drag-and-drop paths. The picker returns nullopt when the
// platform has no dialog or the user cancels; the text field stays available.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace waffle::ui {

std::optional<std::filesystem::path> pick_folder(const std::string& title);

void push_dropped_paths(std::vector<std::filesystem::path> paths);
std::vector<std::filesystem::path> take_dropped_paths();

} // namespace waffle::ui
