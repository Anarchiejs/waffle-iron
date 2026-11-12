# Changelog

## Unreleased

- Linux and macOS builds beside Windows. CPU presets on all three, CUDA on Windows and Linux, Metal on macOS.
- GLFW and OpenGL interface on Linux and macOS. The Windows interface stays on DirectX 11.
- Model folders follow `$HOME` on Linux and macOS, plus the Ollama systemd library on Linux. The library database uses XDG on Linux and Application Support on macOS.
- `install.sh`, `launch.sh`, and `scripts/dev.sh` next to the Windows scripts. CI builds and tests `cpu-release` on Linux and macOS as well as Windows.
- The Lucide icon font ships in `assets/fonts/` and is copied next to `WaffleIron`. Sidebar icons use it on every platform.
- On macOS, GPU copy no longer says NVML. **Verify for real** still loads the model when Metal does not report a memory sample.
- Linux install docs and `install.sh` require `zip` and the GLFW X11 packages. A shallow vcpkg clone is fetched up to the `builtin-baseline` commit.
- The AddressSanitizer preset on Linux and macOS links release vcpkg libraries.
- Linux Ollama libraries are discovered from `OLLAMA_MODELS`, the systemd unit, and the `ollama` user's home, with `/usr/share/ollama/.ollama/models/blobs` kept as a fallback.

## 0.1.0

- Windows workbench for local GGUF models: library, VRAM planner, bench, sampler lab, comparator, and GPU monitor, plus the `waffle` command line.
- Built on a pinned llama.cpp submodule, Dear ImGui, ImPlot, and DirectX 11. CUDA is optional. The monitor uses NVML from the driver.
- English interface. The product name is Waffle Iron.
- One-command install (`install.cmd`) and a Windows CPU CI build.
- MIT license for this project's code. Third-party terms are in `THIRD_PARTY.md`. Copyright 2026 Anarchy.
