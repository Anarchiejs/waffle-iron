// Native folder picker and drag-and-drop paths. Cancelled leaves the text field as it is.
// Unavailable means this machine has no folder dialog (on Linux, neither zenity nor kdialog).
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace waffle::ui {

enum class FolderPickStatus { Picked, Cancelled, Unavailable };

struct FolderPick {
    FolderPickStatus status = FolderPickStatus::Cancelled;
    std::optional<std::filesystem::path> path;
};

FolderPick pick_folder(const std::string& title);

void push_dropped_paths(std::vector<std::filesystem::path> paths);
std::vector<std::filesystem::path> take_dropped_paths();

} // namespace waffle::ui
