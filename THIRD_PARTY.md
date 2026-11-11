# Third-party notices

Waffle Iron's own source is MIT, copyright 2026 Anarchy (`LICENSE`). The pieces below stay under their own terms. Do not relicense them.

## llama.cpp and ggml

Pinned submodule `external/llama.cpp` at `4b851fbee8867136a0c561c8a3521f2fe7d53252` (26 Oct 2026).

MIT License. Copyright (c) 2023-2026 The ggml authors.

https://github.com/ggml-org/llama.cpp

The build sets `LLAMA_CURL` off, so libcurl is not linked. Keep the submodule's `LICENSE` with any redistribution of that code.

## Dear ImGui

Fetched with vcpkg (`imgui`, features `dx11-binding` and `win32-binding`). Not vendored in this repository.

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

`assets/fonts/` ships no font files, only instructions. If you add them, keep their notices with any redistribution:

- Inter (`Inter-Regular.ttf`, `Inter-SemiBold.ttf`): SIL Open Font License 1.1. https://github.com/rsms/inter
- Lucide (`lucide.ttf`): ISC License. https://github.com/lucide-icons/lucide

Without those files, the interface uses Dear ImGui's built-in font.

## Windows and NVIDIA components

DirectX 11 (`d3d11`, `dxgi`) comes from the Windows SDK and is used as a system library. It is not copied into this repository.

NVML (`nvml.dll`) is loaded at run time from the NVIDIA driver. The CUDA backend is optional and is compiled only when `WAFFLE_CUDA` is on. This repository does not ship the CUDA Toolkit or a CUDA runtime. A binary you build with CUDA is subject to NVIDIA's terms for redistributing that runtime.
