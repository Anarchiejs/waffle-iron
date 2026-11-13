// Linux and macOS shell. Windows keeps src/ui/main_win32.cpp (DirectX 11).
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

#include "core/util.hpp"
#include "ui/app.hpp"
#include "ui/platform.hpp"
#include "ui/theme.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace {
bool shot_requested = false;
std::string shot_path;

// Test script (development): WAFFLE_SCRIPT=<file>, one command per line, played frame by frame by
// injecting events straight into ImGui (independent of the real cursor):
//   wait <frames> | click <x> <y> | move <x> <y> | scroll <x> <y> <steps> | type <text> | key enter|backspace|ctrl+a
//   shot <file.bmp> | quit
struct Script {
    std::vector<std::string> lines;
    size_t next = 0;
    int wait = 0;
    ImVec2 mouse{-FLT_MAX, -FLT_MAX};
    bool done = false;
    std::vector<std::function<void(ImGuiIO&)>> pending;
};

std::unique_ptr<Script> load_script() {
    const char* path = std::getenv("WAFFLE_SCRIPT");
    if (!path) return nullptr;
    std::ifstream in(path);
    auto script = std::make_unique<Script>();
    for (std::string line; std::getline(in, line);)
        if (!line.empty() && line[0] != '#') script->lines.push_back(line);
    return script;
}

void step_script(Script& sc, ImGuiIO& io) {
    if (sc.mouse.x > -FLT_MAX) io.AddMousePosEvent(sc.mouse.x, sc.mouse.y);
    if (!sc.pending.empty()) {
        sc.pending.front()(io);
        sc.pending.erase(sc.pending.begin());
        return;
    }
    if (sc.wait > 0) {
        --sc.wait;
        return;
    }
    if (sc.next >= sc.lines.size()) {
        sc.done = true;
        return;
    }
    std::istringstream line(sc.lines[sc.next++]);
    std::string cmd;
    line >> cmd;
    if (cmd == "wait") {
        line >> sc.wait;
    } else if (cmd == "move" || cmd == "click") {
        float x = 0, y = 0;
        line >> x >> y;
        sc.mouse = {x, y};
        io.AddMousePosEvent(x, y);
        if (cmd == "click") {
            sc.pending.push_back([](ImGuiIO&) {});
            sc.pending.push_back([](ImGuiIO& i) { i.AddMouseButtonEvent(0, true); });
            sc.pending.push_back([](ImGuiIO&) {});
            sc.pending.push_back([](ImGuiIO& i) { i.AddMouseButtonEvent(0, false); });
            sc.pending.push_back([](ImGuiIO&) {});
        }
    } else if (cmd == "scroll") {
        float x = 0, y = 0, d = 0;
        line >> x >> y >> d;
        sc.mouse = {x, y};
        io.AddMousePosEvent(x, y);
        sc.pending.push_back([d](ImGuiIO& i) { i.AddMouseWheelEvent(0, d); });
    } else if (cmd == "type") {
        std::string text;
        std::getline(line, text);
        if (!text.empty() && text[0] == ' ') text.erase(0, 1);
        io.AddInputCharactersUTF8(text.c_str());
    } else if (cmd == "key") {
        std::string k;
        line >> k;
        if (k == "ctrl+a") {
            io.AddKeyEvent(ImGuiMod_Ctrl, true);
            io.AddKeyEvent(ImGuiKey_A, true);
            sc.pending.push_back([](ImGuiIO& i) {
                i.AddKeyEvent(ImGuiKey_A, false);
                i.AddKeyEvent(ImGuiMod_Ctrl, false);
            });
        } else {
            const ImGuiKey key = k == "enter"       ? ImGuiKey_Enter
                                 : k == "backspace" ? ImGuiKey_Backspace
                                                    : ImGuiKey_Escape;
            io.AddKeyEvent(key, true);
            sc.pending.push_back([key](ImGuiIO& i) { i.AddKeyEvent(key, false); });
        }
    } else if (cmd == "shot") {
        line >> shot_path;
        shot_requested = true;
    } else if (cmd == "quit") {
        sc.done = true;
    }
}

void put_u32(unsigned char* dest, uint32_t value) {
    dest[0] = static_cast<unsigned char>(value);
    dest[1] = static_cast<unsigned char>(value >> 8);
    dest[2] = static_cast<unsigned char>(value >> 16);
    dest[3] = static_cast<unsigned char>(value >> 24);
}

// Top-down 32-bit BMP, matching the Windows capture (BGRA, alpha forced to 255).
bool save_rgba_bmp(const char* path, int width, int height, const unsigned char* rgba) {
    if (!path || width <= 0 || height <= 0 || !rgba) return false;
    const uint32_t row = static_cast<uint32_t>(width) * 4;
    const uint32_t pixels = row * static_cast<uint32_t>(height);
    unsigned char file_header[14] = {};
    unsigned char info_header[40] = {};
    file_header[0] = 'B';
    file_header[1] = 'M';
    put_u32(file_header + 2, 14u + 40u + pixels);
    put_u32(file_header + 10, 14u + 40u);
    put_u32(info_header, 40);
    put_u32(info_header + 4, static_cast<uint32_t>(width));
    put_u32(info_header + 8, static_cast<uint32_t>(-height));
    info_header[12] = 1;
    info_header[14] = 32;
    FILE* file = std::fopen(path, "wb");
    if (!file) return false;
    std::fwrite(file_header, 1, sizeof file_header, file);
    std::fwrite(info_header, 1, sizeof info_header, file);
    std::vector<unsigned char> line(row);
    for (int y = height - 1; y >= 0; --y) {
        const unsigned char* src = rgba + static_cast<size_t>(y) * row;
        for (int x = 0; x < width; ++x) {
            line[static_cast<size_t>(x) * 4 + 0] = src[x * 4 + 2];
            line[static_cast<size_t>(x) * 4 + 1] = src[x * 4 + 1];
            line[static_cast<size_t>(x) * 4 + 2] = src[x * 4 + 0];
            line[static_cast<size_t>(x) * 4 + 3] = 255;
        }
        std::fwrite(line.data(), 1, line.size(), file);
    }
    std::fclose(file);
    return true;
}

bool save_framebuffer_bmp(const char* path) {
    GLint viewport[4] = {};
    glGetIntegerv(GL_VIEWPORT, viewport);
    const int width = viewport[2];
    const int height = viewport[3];
    if (width <= 0 || height <= 0) return false;
    std::vector<unsigned char> rgba(static_cast<size_t>(width) * static_cast<size_t>(height) * 4);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    return save_rgba_bmp(path, width, height, rgba.data());
}

std::filesystem::path executable_directory() {
#if defined(__linux__)
    std::error_code ec;
    const auto exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec) return exe.parent_path();
#elif defined(__APPLE__)
    char buf[4096];
    uint32_t size = sizeof buf;
    if (_NSGetExecutablePath(buf, &size) == 0) {
        std::error_code ec;
        const auto canonical = std::filesystem::weakly_canonical(buf, ec);
        if (!ec) return canonical.parent_path();
        return std::filesystem::path(buf).parent_path();
    }
#endif
    return std::filesystem::current_path();
}

void glfw_error(int code, const char* description) {
    std::fprintf(stderr, "GLFW error %d: %s\n", code, description ? description : "");
}
} // namespace

// Capture mode (development): WAFFLE_CAPTURE=<file.bmp> saves the rendered image, then quits.
// WAFFLE_CAPTURE_SIZE=WxH sets the client size, WAFFLE_CAPTURE_FRAMES the frames before the capture.
int main() {
    const char* capture_path = std::getenv("WAFFLE_CAPTURE");
    int capture_w = 0, capture_h = 0, capture_frames = 90;
    if (const char* size = std::getenv("WAFFLE_CAPTURE_SIZE")) std::sscanf(size, "%dx%d", &capture_w, &capture_h);
    if (const char* frames = std::getenv("WAFFLE_CAPTURE_FRAMES")) capture_frames = std::atoi(frames);

    glfwSetErrorCallback(glfw_error);
    if (!glfwInit()) {
        std::fprintf(stderr, "Cannot initialize GLFW.\n");
        return 1;
    }

    float monitor_x = 1.f, monitor_y = 1.f;
    if (GLFWmonitor* monitor = glfwGetPrimaryMonitor()) glfwGetMonitorContentScale(monitor, &monitor_x, &monitor_y);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
#ifdef __APPLE__
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#endif
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1400, 900, "Waffle Iron", nullptr, nullptr);
    if (window) {
        glfwSetDropCallback(window, [](GLFWwindow*, int count, const char** paths) {
            std::vector<std::filesystem::path> dropped;
            for (int i = 0; i < count; ++i) dropped.push_back(waffle::path_from_utf8(paths[i]));
            waffle::ui::push_dropped_paths(std::move(dropped));
        });
    }
    if (!window) {
        std::fprintf(stderr, "Cannot open a window. On Linux the interface needs X11 or XWayland and OpenGL.\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    int window_w = 0, window_h = 0, framebuffer_w = 0, framebuffer_h = 0;
    glfwGetWindowSize(window, &window_w, &window_h);
    glfwGetFramebufferSize(window, &framebuffer_w, &framebuffer_h);
    // Retina windows already map one point to several pixels. Elsewhere, grow the window with the monitor scale
    // the way the Windows shell grows it with the DPI.
    const bool retina = window_w > 0 && framebuffer_w > window_w;
    float dpi = !retina && monitor_x > 0.f ? monitor_x : 1.f;
    if (!retina && dpi > 1.f) glfwSetWindowSize(window, static_cast<int>(1400 * dpi), static_cast<int>(900 * dpi));
    if (capture_w > 0 && capture_h > 0) glfwSetWindowSize(window, capture_w, capture_h);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    auto& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    const auto directory = executable_directory();
    std::string ini = waffle::narrow_utf8(directory / "imgui.ini");
    io.IniFilename = ini.c_str();
    const bool icons = waffle::ui::apply_theme(dpi, directory / "assets");

    const char* glsl = "#version 150";
    bool glfw_ok = ImGui_ImplGlfw_InitForOpenGL(window, true);
    bool gl_ok = glfw_ok && ImGui_ImplOpenGL3_Init(glsl);
    int result = 0;
    if (gl_ok) {
        glfwShowWindow(window);
        try {
            waffle::ui::App app(icons);
            auto script = load_script();
            float current_dpi = dpi;
            while (!glfwWindowShouldClose(window) && !app.quitting()) {
                glfwPollEvents();
                if (glfwGetWindowAttrib(window, GLFW_ICONIFIED)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
                float xscale = 1.f, yscale = 1.f;
                glfwGetWindowContentScale(window, &xscale, &yscale);
                const float next_dpi = !retina && xscale > 0.f ? xscale : 1.f;
                if (next_dpi != current_dpi) {
                    waffle::ui::apply_theme(next_dpi, directory / "assets");
                    ImGui_ImplOpenGL3_DestroyDeviceObjects();
                    current_dpi = next_dpi;
                }
                ImGui_ImplOpenGL3_NewFrame();
                ImGui_ImplGlfw_NewFrame();
                if (script) step_script(*script, io);
                ImGui::NewFrame();
                app.frame();
                ImGui::Render();
                int display_w = 0, display_h = 0;
                glfwGetFramebufferSize(window, &display_w, &display_h);
                glViewport(0, 0, display_w, display_h);
                glClearColor(0.078f, 0.086f, 0.102f, 1.f);
                glClear(GL_COLOR_BUFFER_BIT);
                ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
                if (shot_requested) {
                    shot_requested = false;
                    save_framebuffer_bmp(shot_path.c_str());
                }
                if (script && script->done) break;
                if (capture_path && --capture_frames <= 0) {
                    save_framebuffer_bmp(capture_path);
                    break;
                }
                glfwSwapBuffers(window);
            }
        } catch (const std::exception& error) {
            std::fprintf(stderr, "Waffle Iron: %s\n", error.what());
            result = 1;
        }
    } else {
        std::fprintf(stderr, "Cannot initialize ImGui.\n");
        result = 1;
    }
    if (gl_ok) ImGui_ImplOpenGL3_Shutdown();
    if (glfw_ok) ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return result;
}
