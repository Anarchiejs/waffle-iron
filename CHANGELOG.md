# Changelog

## Unreleased

- New interface theme, "Gaufrier": warm carbon background, a single syrup accent, values and paths in monospace. Every page is redrawn with it; no control or feature was removed.
- The sidebar groups the pages as Understand (Library, VRAM planner), Measure (Bench, Monitor), and Tune (Sampler lab, Comparator). Ctrl+1 through Ctrl+6 follow that order, so Monitor moves to Ctrl+4, Sampler lab to Ctrl+5, and Comparator to Ctrl+6. The mark is a waffle grid.
- Waffle charts: the VRAM planner's memory breakdown (96 cells = total VRAM, striped memory already in use, cells over capacity outlined), the Library's weight layout (100 cells), and cell meters for VRAM in the sidebar.
- The comparator's word diff and the sampler's generated text flow as one paragraph with tinted spans. The sampler's "Retour au parent" button is now "Back to parent".
- Optional fonts: Space Grotesk for page titles and JetBrains Mono for values, placed in `assets/fonts/` like Inter.

- UTF-8 paths for model loads, GGUF names, fonts, and the interface settings file. The library key is the canonical path.
- Benchmark results store the SWA cache mode and the seed. The history label includes them. VRAM peaks are the increase over the baseline, and the planner check shares one load-and-decode measurement with the command line.
- `waffle --version`, `--ubatch`, `--swa-full`, `--threads`, `--warmup`, `--seed`, `--ctv`, JSON export and import, perplexity, and a non-zero exit when a scan or GPU sample fails.
- Sampler lab: Mirostat v2 and DRY. The comparator shows a word diff. The bench grid can measure several models, and history charts draw standard-deviation bars.
- DirectX 11 uses a flip discard swap chain and rebuilds the device if it is removed. Fonts rebuild when the DPI changes. A folder dialog and file drop add library folders.
- Schema v2, one query for tags and one for campaign names, and the command line is a library covered by tests.
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
