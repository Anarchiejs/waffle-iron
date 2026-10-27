#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <imgui.h>
#include <implot.h>
#include <imgui_impl_win32.h>
#include <imgui_impl_dx11.h>
#include "ui/app.hpp"
#include "ui/theme.hpp"
#include <filesystem>
#include <memory>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <functional>
#include <sstream>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);
namespace {
ID3D11Device* device = nullptr;
ID3D11DeviceContext* context = nullptr;
IDXGISwapChain* swap_chain = nullptr;
ID3D11RenderTargetView* target = nullptr;
UINT resize_width = 0, resize_height = 0;
bool shot_requested = false;
std::string shot_path;
// Scénario de test (développement) : BANC_SCRIPT=<fichier>, une commande par ligne, jouée image par image en
// injectant les évènements directement dans ImGui (indépendant du curseur réel) :
//   wait <images> | click <x> <y> | move <x> <y> | scroll <x> <y> <crans> | type <texte> | key enter|backspace|ctrl+a
//   shot <fichier.bmp> | quit
struct Script {
    std::vector<std::string> lines;
    size_t next = 0;
    int wait = 0;
    ImVec2 mouse{-FLT_MAX, -FLT_MAX};
    bool done = false;
    // Suite d'évènements élémentaires différés (un par image) pour qu'ImGui voie chaque transition.
    std::vector<std::function<void(ImGuiIO&)>> pending;
};
std::unique_ptr<Script> load_script() {
    const char* path = std::getenv("BANC_SCRIPT");
    if (!path) return nullptr;
    std::ifstream in(path);
    auto script = std::make_unique<Script>();
    for (std::string line; std::getline(in, line);)
        if (!line.empty() && line[0] != '#') script->lines.push_back(line);
    return script;
}
void step_script(Script& sc, ImGuiIO& io) {
    // La position simulée est réaffirmée à chaque image (le moteur Win32 peut injecter le curseur réel).
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
            const ImGuiKey key = k == "enter" ? ImGuiKey_Enter : k == "backspace" ? ImGuiKey_Backspace : ImGuiKey_Escape;
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
void release_target() {
    if (target) {
        target->Release();
        target = nullptr;
    }
}
bool create_target() {
    ID3D11Texture2D* buffer = nullptr;
    if (FAILED(swap_chain->GetBuffer(0, IID_PPV_ARGS(&buffer)))) return false;
    const auto hr = device->CreateRenderTargetView(buffer, nullptr, &target);
    buffer->Release();
    return SUCCEEDED(hr);
}
void cleanup() {
    release_target();
    if (swap_chain) swap_chain->Release();
    if (context) context->Release();
    if (device) device->Release();
}
LRESULT WINAPI window_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wp, lp)) return 1;
    switch (msg) {
        case WM_SIZE:
            if (wp != SIZE_MINIMIZED) {
                resize_width = LOWORD(lp);
                resize_height = HIWORD(lp);
            }
            return 0;
        case WM_DPICHANGED: {
            const auto* rect = reinterpret_cast<RECT*>(lp);
            SetWindowPos(hwnd, nullptr, rect->left, rect->top, rect->right - rect->left, rect->bottom - rect->top,
                         SWP_NOZORDER | SWP_NOACTIVATE);
            return 0;
        }
        case WM_SYSCOMMAND:
            if ((wp & 0xfff0) == SC_KEYMENU) return 0;
            break;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}
} // namespace
// Mode capture (développement) : BANC_CAPTURE=<fichier.bmp> enregistre l'image rendue puis quitte.
// BANC_CAPTURE_SIZE=LxH fixe la taille de la zone cliente, BANC_CAPTURE_FRAMES le nombre d'images avant capture.
bool save_backbuffer_bmp(const char* path) {
    ID3D11Texture2D* buffer = nullptr;
    if (FAILED(swap_chain->GetBuffer(0, IID_PPV_ARGS(&buffer)))) return false;
    D3D11_TEXTURE2D_DESC desc{};
    buffer->GetDesc(&desc);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;
    ID3D11Texture2D* staging = nullptr;
    bool ok = SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &staging));
    if (ok) {
        context->CopyResource(staging, buffer);
        D3D11_MAPPED_SUBRESOURCE map{};
        ok = SUCCEEDED(context->Map(staging, 0, D3D11_MAP_READ, 0, &map));
        if (ok) {
            const int w = static_cast<int>(desc.Width), h = static_cast<int>(desc.Height);
            BITMAPFILEHEADER fh{};
            BITMAPINFOHEADER ih{};
            ih.biSize = sizeof ih;
            ih.biWidth = w;
            ih.biHeight = -h;
            ih.biPlanes = 1;
            ih.biBitCount = 32;
            fh.bfType = 0x4d42;
            fh.bfOffBits = sizeof fh + sizeof ih;
            fh.bfSize = fh.bfOffBits + static_cast<DWORD>(w * h * 4);
            if (FILE* f = std::fopen(path, "wb")) {
                std::fwrite(&fh, sizeof fh, 1, f);
                std::fwrite(&ih, sizeof ih, 1, f);
                std::vector<unsigned char> row(static_cast<size_t>(w) * 4);
                for (int y = 0; y < h; ++y) {
                    const auto* src = static_cast<const unsigned char*>(map.pData) + static_cast<size_t>(y) * map.RowPitch;
                    for (int x = 0; x < w; ++x) { // RGBA -> BGRA
                        row[x * 4 + 0] = src[x * 4 + 2];
                        row[x * 4 + 1] = src[x * 4 + 1];
                        row[x * 4 + 2] = src[x * 4 + 0];
                        row[x * 4 + 3] = 255;
                    }
                    std::fwrite(row.data(), 1, row.size(), f);
                }
                std::fclose(f);
            } else
                ok = false;
            context->Unmap(staging, 0);
        }
        staging->Release();
    }
    buffer->Release();
    return ok;
}
int WINAPI WinMain(HINSTANCE instance, HINSTANCE, LPSTR, int show) {
    const char* capture_path = std::getenv("BANC_CAPTURE");
    int capture_w = 0, capture_h = 0, capture_frames = 90;
    if (const char* size = std::getenv("BANC_CAPTURE_SIZE")) std::sscanf(size, "%dx%d", &capture_w, &capture_h);
    if (const char* frames = std::getenv("BANC_CAPTURE_FRAMES")) capture_frames = std::atoi(frames);
    ImGui_ImplWin32_EnableDpiAwareness();
    WNDCLASSEXW wc{sizeof(WNDCLASSEXW),
                   CS_CLASSDC,
                   window_proc,
                   0,
                   0,
                   instance,
                   LoadIconW(nullptr, MAKEINTRESOURCEW(32512)),
                   LoadCursorW(nullptr, MAKEINTRESOURCEW(32512)),
                   nullptr,
                   nullptr,
                   L"BancLocal",
                   nullptr};
    if (!RegisterClassExW(&wc)) return 1;
    const float dpi = ImGui_ImplWin32_GetDpiScaleForMonitor(MonitorFromPoint({0, 0}, MONITOR_DEFAULTTOPRIMARY));
    HWND hwnd =
        CreateWindowW(wc.lpszClassName, L"Banc Local", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                      static_cast<int>(1400 * dpi), static_cast<int>(900 * dpi), nullptr, nullptr, instance, nullptr);
    if (!hwnd) {
        UnregisterClassW(wc.lpszClassName, instance);
        return 1;
    }
    if (capture_w > 0 && capture_h > 0) {
        RECT r{0, 0, capture_w, capture_h};
        AdjustWindowRectEx(&r, WS_OVERLAPPEDWINDOW, FALSE, 0);
        SetWindowPos(hwnd, nullptr, 0, 0, r.right - r.left, r.bottom - r.top, SWP_NOMOVE | SWP_NOZORDER);
    }
    DXGI_SWAP_CHAIN_DESC sd{};
    sd.BufferCount = 2;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hwnd;
    sd.SampleDesc.Count = 1;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    D3D_FEATURE_LEVEL level;
    HRESULT hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
                                               D3D11_SDK_VERSION, &sd, &swap_chain, &device, &level, &context);
    if (hr == DXGI_ERROR_UNSUPPORTED)
        hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION,
                                           &sd, &swap_chain, &device, &level, &context);
    if (FAILED(hr) || !create_target()) {
        MessageBoxW(hwnd, L"Impossible d'initialiser DirectX 11.", L"Banc Local", MB_ICONERROR);
        cleanup();
        DestroyWindow(hwnd);
        UnregisterClassW(wc.lpszClassName, instance);
        return 1;
    }
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    auto& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    wchar_t exe[MAX_PATH]{};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    const auto directory = std::filesystem::path(exe).parent_path();
    std::string ini = (directory / "imgui.ini").string();
    io.IniFilename = ini.c_str();
    const bool icons = banc::ui::apply_theme(dpi, directory / "assets");
    bool win32_ok = ImGui_ImplWin32_Init(hwnd);
    bool dx_ok = win32_ok && ImGui_ImplDX11_Init(device, context);
    int result = 0;
    if (dx_ok) {
        ShowWindow(hwnd, show);
        UpdateWindow(hwnd);
        try {
            banc::ui::App app(icons);
            auto script = load_script();
            bool done = false;
            float current_dpi = dpi;
            while (!done && !app.quitting()) {
                MSG msg;
                while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
                    TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                    if (msg.message == WM_QUIT) done = true;
                }
                if (done) break;
                if (IsIconic(hwnd)) {
                    Sleep(10);
                    continue;
                }
                if (resize_width && resize_height) {
                    release_target();
                    hr = swap_chain->ResizeBuffers(0, resize_width, resize_height, DXGI_FORMAT_UNKNOWN, 0);
                    resize_width = resize_height = 0;
                    if (FAILED(hr) || !create_target())
                        throw std::runtime_error("Redimensionnement DirectX impossible");
                }
                const float next_dpi = ImGui_ImplWin32_GetDpiScaleForHwnd(hwnd);
                if (next_dpi > 0 && next_dpi != current_dpi) {
                    ImGui::GetStyle().ScaleAllSizes(next_dpi / current_dpi);
                    ImGui::GetStyle().FontScaleMain = next_dpi / dpi;
                    current_dpi = next_dpi;
                }
                ImGui_ImplDX11_NewFrame();
                ImGui_ImplWin32_NewFrame();
                if (script) step_script(*script, io);
                ImGui::NewFrame();
                app.frame();
                ImGui::Render();
                const float clear[] = {.078f, .086f, .102f, 1};
                context->OMSetRenderTargets(1, &target, nullptr);
                context->ClearRenderTargetView(target, clear);
                ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
                if (shot_requested) {
                    shot_requested = false;
                    save_backbuffer_bmp(shot_path.c_str());
                }
                if (script && script->done) break;
                if (capture_path && --capture_frames <= 0) {
                    save_backbuffer_bmp(capture_path);
                    break;
                }
                hr = swap_chain->Present(1, 0);
                if (hr == DXGI_STATUS_OCCLUDED)
                    Sleep(10);
                else if (FAILED(hr))
                    throw std::runtime_error("Le périphérique DirectX a été perdu");
            }
        } catch (const std::exception& e) {
            MessageBoxA(hwnd, e.what(), "Banc Local", MB_ICONERROR);
            result = 1;
        }
    } else {
        MessageBoxW(hwnd, L"Initialisation ImGui impossible.", L"Banc Local", MB_ICONERROR);
        result = 1;
    }
    if (dx_ok) ImGui_ImplDX11_Shutdown();
    if (win32_ok) ImGui_ImplWin32_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    cleanup();
    DestroyWindow(hwnd);
    UnregisterClassW(wc.lpszClassName, instance);
    return result;
}
