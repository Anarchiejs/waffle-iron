# Third-party notices

Waffle Iron's own source is MIT, copyright 2026 Anarchy (`LICENSE`). The pieces below stay under their own terms. Do not relicense them.

## llama.cpp and ggml

Pinned submodule `external/llama.cpp` at `4b851fbee8867136a0c561c8a3521f2fe7d53252` (26 Oct 2026).

MIT License. Copyright (c) 2023-2026 The ggml authors.

https://github.com/ggml-org/llama.cpp

The build sets `LLAMA_CURL` off, so libcurl is not linked. Keep the submodule's `LICENSE` with any redistribution of that code.

## Dear ImGui

Fetched with vcpkg. Windows uses the `dx11-binding` and `win32-binding` features. Linux and macOS use `glfw-binding` and `opengl3-binding`, which pull in GLFW. Not vendored in this repository.

MIT License. Copyright (c) Omar Cornut.

https://github.com/ocornut/imgui

## ImPlot

Fetched with vcpkg (`implot`). Not vendored in this repository.

MIT License. Copyright (c) Evan Pezent.

https://github.com/epezent/implot

## SQLite

Fetched with vcpkg (`sqlite3`) and linked as `unofficial::sqlite3`.

Public domain.

https://www.sqlite.org

## Catch2

Fetched with vcpkg (`catch2`) and linked only into the test binary.

Boost Software License 1.0.

https://github.com/catchorg/Catch2

## Fonts

`assets/fonts/lucide.ttf` (Lucide 0.544.0) is included. The build copies it next to `WaffleIron`, and `NOTICE` carries the full ISC text plus the MIT notice for the Feather icons incorporated into Lucide. https://github.com/lucide-icons/lucide

Inter (`Inter-Regular.ttf`, `Inter-SemiBold.ttf`) is not included. If you add it, keep the SIL Open Font License 1.1 notice with any redistribution. https://github.com/rsms/inter

Without Inter, the interface uses a system font or Dear ImGui's built-in font. Sidebar icons fall back to drawn strokes when `lucide.ttf` is missing beside the binary.

## GLFW

Fetched with vcpkg on Linux and macOS as a dependency of Dear ImGui's GLFW binding. Not used by the Windows build.

Zlib License. Copyright (c) Camilla Löwy and the GLFW contributors.

https://www.glfw.org

## Platform and GPU components

DirectX 11 (`d3d11`, `dxgi`) comes from the Windows SDK and is used as a system library. It is not copied into this repository.

OpenGL is a system library on Linux (Mesa or the vendor driver) and on macOS (the OpenGL framework). It is not copied into this repository.

NVML (`nvml.dll` on Windows, `libnvidia-ml.so.1` on Linux) is loaded at run time from the NVIDIA driver. The CUDA backend is optional and is compiled only when `WAFFLE_CUDA` is on. This repository does not ship the CUDA Toolkit or a CUDA runtime. A binary you build with CUDA is subject to NVIDIA's terms for redistributing that runtime.

Metal and Foundation are macOS system frameworks. They are linked when the build runs on Apple, for the monitor's device query, and when `WAFFLE_METAL` is on, for llama.cpp's Metal backend. They are not copied into this repository.
