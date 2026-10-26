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
#include <cstdlib>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);
namespace {
ID3D11Device* device = nullptr;
ID3D11DeviceContext* context = nullptr;
IDXGISwapChain* swap_chain = nullptr;
ID3D11RenderTargetView* target = nullptr;
UINT resize_width = 0, resize_height = 0;
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
int WINAPI WinMain(HINSTANCE instance, HINSTANCE, LPSTR, int show) {
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
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable | ImGuiConfigFlags_NavEnableKeyboard;
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
                ImGui::NewFrame();
                app.frame();
                ImGui::Render();
                const float clear[] = {.078f, .086f, .102f, 1};
                context->OMSetRenderTargets(1, &target, nullptr);
                context->ClearRenderTargetView(target, clear);
                ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
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
