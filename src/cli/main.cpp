#include "cli/cli.hpp"

#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

namespace {

std::vector<std::string> arguments(int argc, char** argv) {
#ifdef _WIN32
    (void)argc;
    (void)argv;
    int count = 0;
    LPWSTR* wide = CommandLineToArgvW(GetCommandLineW(), &count);
    if (!wide) return {};
    std::vector<std::string> args;
    args.reserve(count > 0 ? static_cast<size_t>(count - 1) : 0);
    for (int i = 1; i < count; ++i) {
        const int bytes = WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, nullptr, 0, nullptr, nullptr);
        if (bytes <= 1) {
            args.emplace_back();
            continue;
        }
        std::string utf8(static_cast<size_t>(bytes - 1), '\0');
        WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, utf8.data(), bytes, nullptr, nullptr);
        args.push_back(std::move(utf8));
    }
    LocalFree(wide);
    return args;
#else
    std::vector<std::string> args;
    args.reserve(argc > 0 ? static_cast<size_t>(argc - 1) : 0);
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    return args;
#endif
}

} // namespace

int main(int argc, char** argv) {
    return waffle::cli::run(arguments(argc, argv));
}
