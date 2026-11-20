#include "ui/platform.hpp"

#include "core/util.hpp"

#include <cstdlib>
#include <cstdio>
#include <mutex>

#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <shobjidl.h>
#include <windows.h>
#endif

namespace waffle::ui {
namespace {

std::mutex g_drop_mutex;
std::vector<std::filesystem::path> g_dropped;

#if !defined(_WIN32)
bool executable_on_path(const char* name) {
    const char* path = std::getenv("PATH");
    if (!path || !*path) return false;
    const std::string dirs = path;
    std::size_t start = 0;
    while (start <= dirs.size()) {
        auto end = dirs.find(':', start);
        if (end == std::string::npos) end = dirs.size();
        const auto dir = dirs.substr(start, end - start);
        if (!dir.empty()) {
            const auto candidate = (std::filesystem::path(dir) / name).string();
            if (::access(candidate.c_str(), X_OK) == 0) return true;
        }
        if (end == dirs.size()) break;
        start = end + 1;
    }
    return false;
}

std::optional<std::filesystem::path> read_command(const std::string& command) {
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) return std::nullopt;
    std::string text;
    char buf[1024];
    while (std::fgets(buf, sizeof buf, pipe)) text += buf;
    const int rc = pclose(pipe);
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
    if (rc != 0 || text.empty()) return std::nullopt;
    return path_from_utf8(text);
}
#endif

} // namespace

void push_dropped_paths(std::vector<std::filesystem::path> paths) {
    std::lock_guard lock(g_drop_mutex);
    for (auto& path : paths) g_dropped.push_back(std::move(path));
}

std::vector<std::filesystem::path> take_dropped_paths() {
    std::lock_guard lock(g_drop_mutex);
    std::vector<std::filesystem::path> paths;
    paths.swap(g_dropped);
    return paths;
}

FolderPick pick_folder(const std::string& title) {
    FolderPick result;
#ifdef _WIN32
    const HRESULT init = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    const bool uninit = SUCCEEDED(init);
    IFileDialog* dialog = nullptr;
    if (!SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog)))) {
        if (uninit) CoUninitialize();
        result.status = FolderPickStatus::Unavailable;
        return result;
    }
    DWORD options = 0;
    dialog->GetOptions(&options);
    dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
    const std::wstring wide(title.begin(), title.end());
    dialog->SetTitle(wide.c_str());
    const HRESULT shown = dialog->Show(nullptr);
    if (shown == HRESULT_FROM_WIN32(ERROR_CANCELLED)) {
        result.status = FolderPickStatus::Cancelled;
    } else if (!SUCCEEDED(shown)) {
        result.status = FolderPickStatus::Unavailable;
    } else {
        IShellItem* item = nullptr;
        if (SUCCEEDED(dialog->GetResult(&item))) {
            PWSTR display = nullptr;
            if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &display)) && display) {
                result.path = std::filesystem::path(display);
                result.status = FolderPickStatus::Picked;
                CoTaskMemFree(display);
            }
            item->Release();
        }
        if (result.status != FolderPickStatus::Picked) result.status = FolderPickStatus::Unavailable;
    }
    dialog->Release();
    if (uninit) CoUninitialize();
    return result;
#elif defined(__APPLE__)
    (void)title;
    if (!executable_on_path("osascript")) {
        result.status = FolderPickStatus::Unavailable;
        return result;
    }
    if (auto picked = read_command(
            "osascript -e 'try' -e 'POSIX path of (choose folder with prompt \"Add a model folder\")' -e 'end try'")) {
        result.status = FolderPickStatus::Picked;
        result.path = std::move(picked);
        return result;
    }
    result.status = FolderPickStatus::Cancelled;
    return result;
#else
    (void)title;
    const bool zenity = executable_on_path("zenity");
    const bool kdialog = executable_on_path("kdialog");
    if (!zenity && !kdialog) {
        result.status = FolderPickStatus::Unavailable;
        return result;
    }
    std::optional<std::filesystem::path> picked;
    if (zenity) picked = read_command("zenity --file-selection --directory --title='Add a model folder' 2>/dev/null");
    if (!picked && kdialog) picked = read_command("kdialog --getexistingdirectory . 2>/dev/null");
    if (!picked) {
        result.status = FolderPickStatus::Cancelled;
        return result;
    }
    result.status = FolderPickStatus::Picked;
    result.path = std::move(picked);
    return result;
#endif
}

} // namespace waffle::ui
