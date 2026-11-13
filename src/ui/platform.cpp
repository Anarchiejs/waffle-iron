#include "ui/platform.hpp"

#include "core/util.hpp"

#include <cstdio>
#include <mutex>

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

std::optional<std::filesystem::path> pick_folder(const std::string& title) {
#ifdef _WIN32
    const HRESULT init = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    const bool uninit = SUCCEEDED(init);
    IFileDialog* dialog = nullptr;
    std::optional<std::filesystem::path> result;
    if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog)))) {
        DWORD options = 0;
        dialog->GetOptions(&options);
        dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
        const std::wstring wide(title.begin(), title.end());
        dialog->SetTitle(wide.c_str());
        if (SUCCEEDED(dialog->Show(nullptr))) {
            IShellItem* item = nullptr;
            if (SUCCEEDED(dialog->GetResult(&item))) {
                PWSTR display = nullptr;
                if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &display)) && display) {
                    result = std::filesystem::path(display);
                    CoTaskMemFree(display);
                }
                item->Release();
            }
        }
        dialog->Release();
    }
    if (uninit) CoUninitialize();
    return result;
#elif defined(__APPLE__)
    (void)title;
    return read_command(
        "osascript -e 'try' -e 'POSIX path of (choose folder with prompt \"Add a model folder\")' -e 'end try'");
#else
    (void)title;
    if (auto picked = read_command("zenity --file-selection --directory --title='Add a model folder' 2>/dev/null"))
        return picked;
    return read_command("kdialog --getexistingdirectory . 2>/dev/null");
#endif
}

} // namespace waffle::ui
