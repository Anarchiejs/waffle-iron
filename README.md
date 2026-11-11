# Waffle Iron

Waffle Iron is a desktop workbench for **understanding, measuring, and tuning the GGUF models** that already run on your machine. It is for anyone who wants to know what a model costs in memory, how fast it answers, and how the cache, the context, or the quantization changes the result — without leaving Windows and without sending the files anywhere else.

The name is a pun: GGUF is pronounced like *gaufre*, the French word for waffle.

The tool is native (C++20). It is built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (`libllama` and `ggml`, with the CUDA backend when the GPU and the toolkit are present) and draws its views with Dear ImGui and DirectX 11.

Two programs:

| Program | Role |
|---|---|
| `WaffleIron.exe` | The interface: library, memory, measurements, and generation |
| `waffle.exe` | The same core on the command line, for scripting a scan, a plan, or a campaign |

Windows is the only target platform.

The GitHub repository is [Banc-Local](https://github.com/Anarchiejs/Banc-Local). If that repository is renamed to `waffle-iron`, GitHub keeps the old URL working.

## Limits

- Windows x64 only. The interface needs DirectX 11. The command line reads its arguments with the Win32 API.
- A CPU build runs models on the processor. CUDA and the monitor need an NVIDIA driver. The monitor reads GPU index 0 through NVML. Other GPUs have no monitor and no CUDA backend.
- The checked-in `cuda-release` preset compiles for `sm_89` only. That cubin does not run on `sm_86` or on Blackwell (`sm_100`, `sm_120`). `install.cmd` writes a gitignored preset for the GPU that is actually installed.
- Waffle Iron does not download models and does not contact Ollama, LM Studio, or `registry.ollama.ai`. That name is only a prefix in local manifest paths. The scan reads GGUF files on disk and, when those folders exist, Ollama blobs and manifest JSON, so it can name a digest and skip draft and projector blobs.
- Nothing is uploaded. "Telemetry" in the source is local NVML sampling. The first build does use the network, for the llama.cpp submodule and the vcpkg ports.
- The library database is `%LOCALAPPDATA%\WaffleIron\waffle.db`. It stores the full path of each file. Interface state is `imgui.ini` next to the executable. Exports go to the path you choose.
- Memory estimates are approximate. **Verify for real** checks them on the installed GPU. A result from one card does not transfer to another.
- Waffle Iron is not an official project of ggml, Ollama, LM Studio, or NVIDIA. No model weights are included, except the random-weight test fixture `tests\data\tiny-llama.gguf`.

## Requirements

| Tool | Required | Where to install it |
|---|---|---|
| Git for Windows | yes | https://git-scm.com/download/win |
| Visual Studio 2022 or newer (2026 is fine), **Desktop development with C++** workload | yes | https://visualstudio.microsoft.com/downloads/ |
| MSVC C++ tools (x64) | yes, included in the workload | Visual Studio Installer component |
| C++ CMake tools for Windows (CMake ≥ 3.25 and Ninja) | yes | same installer, selected with the C++ workload |
| vcpkg package manager | yes | same installer: "vcpkg package manager" |
| A recent NVIDIA driver | only for the GPU monitor and a CUDA build | GeForce Experience or the NVIDIA site |
| CUDA Toolkit | no | https://developer.nvidia.com/cuda-downloads |

The Visual Studio Build Tools are enough instead of the IDE, with the same components. CMake and Ninja installed separately (`https://cmake.org/download/`, `https://ninja-build.org/`) are picked up when they are on `PATH` or in their usual location.

Without the CUDA Toolkit the interface still builds: models run on the CPU. The monitor can still read the GPU, because NVML comes from the driver, not from the toolkit.

## Quick install

```bat
git clone --recurse-submodules https://github.com/Anarchiejs/Banc-Local.git waffle-iron
cd waffle-iron
install.cmd
```

`install.cmd` is the only command to run after the clone. It:

1. checks Git, Visual Studio, MSVC, CMake ≥ 3.25, Ninja, and vcpkg, and says what to select or download if something is missing;
2. fetches the `external/llama.cpp` submodule if `--recurse-submodules` was forgotten (a GitHub zip is not enough: it does not contain the submodule);
3. queries `nvidia-smi` and looks for `nvcc`;
4. chooses a CUDA build for the GPU architecture (`cuda-release-local`), or `cpu-release` when there is no usable NVIDIA GPU with a toolkit;
5. configures, builds, then opens `WaffleIron.exe`.

The first run downloads the vcpkg dependencies (SQLite, Dear ImGui, ImPlot, Catch2): it needs the internet. A CUDA build of llama.cpp is long and resumes where it stopped if you run `install.cmd` again. The log is `build\install.log`.

To open the interface later, without rebuilding:

```bat
launch.cmd
```

Variants:

```bat
install.cmd -NoLaunch
install.cmd -Cpu
install.cmd -Cuda
install.cmd -Architecture 86
install.cmd -Architecture 8.9
install.cmd -Tests
install.cmd -Help
```

`-Architecture` accepts `89` or `8.9`, and a list (`86,89`) when several GPUs must be targeted. `-Cpu` and `-Cuda` together are rejected.

## First launch

The database `%LOCALAPPDATA%\WaffleIron\waffle.db` is created on first start. If it is empty, the library stays empty for a few moments: a scan runs in the background, without blocking the window, then the models it found appear. A toast says "Scan finished".

That scan runs again on every launch, even when the Library page is not the one on screen. There is no need to run `waffle scan` by hand for the first fill. **Rescan**, or Ctrl+R, starts it again after you add files.

The sidebar lists the six pages. Ctrl+1 through Ctrl+6 switch pages. The status bar shows the work in progress, the GPU name, and VRAM when NVML answers.

A single thread talks to llama.cpp. The interface sends it work and keeps drawing during a load.

## Where to put models

Waffle Iron does not download anything. It indexes the GGUF files already on disk. The folders it looks at, when they exist:

| Location | Usual contents |
|---|---|
| `%USERPROFILE%\models` | `.gguf` files you put there yourself |
| `%USERPROFILE%\.ollama\models\blobs` | Ollama blobs (often with no extension; the `GGUF` signature is enough) |
| `%USERPROFILE%\.lmstudio\models` | LM Studio models |
| `%USERPROFILE%\.cache\lm-studio\models` | older LM Studio cache folder |
| `%OLLAMA_MODELS%\blobs`, or `%OLLAMA_MODELS%` when there is no `blobs` subfolder | a moved Ollama library |

Example: copy `gemma-4-E4B-Q4_K_M.gguf` into `%USERPROFILE%\models`, then open Waffle Iron (or click **Rescan**). The model shows up with its name, architecture, quantization, size, parameter count, and context.

The Library ignores files that are not GGUF, vision projectors (`mmproj`, architecture `clip`), and split model parts other than the first (`…-00002-of-…`). Another folder is added with **Add a folder** (UTF-8 path). The record of a selected model accepts comma-separated tags and notes, stored in the database.

If none of the folders in the list exist, the page shows them as "absent": create `%USERPROFILE%\models` and put the files there.

## A tour of the interface

The names below are the page titles and the controls.

### Library

Purpose: see what the machine already has, without loading the weights.

Example. You have a Q4_K_M for everyday use and a Q8_0 to compare. After the scan, filter with the search field, click the Q4_K_M row, set the tag `everyday` and a note ("fits in 8 GB at 8k"), then **Save**. The weight-layout bar shows bytes per tensor type. **GGUF metadata** opens the GGUF keys, with a filter.

### VRAM planner

Purpose: see where the memory goes before loading, then check it on the GPU.

Example. Choose a Q4_K_M model. Set the context (4k, 8k, 32k…), the K cache and the V cache (`f16`, `q8_0`, `q4_0`…), flash attention, and **Full SWA cache**. The chart stacks weights, KV cache, buffers, and overhead, on the GPU and the CPU, with total VRAM and VRAM already in use. **Max layers that fit** suggests an offload that stays inside free memory. **Verify for real** loads the model, measures the gap, unloads it, and writes that into the history. The NVML sample is the whole GPU: another program can move it. The gap belongs to that machine.

Sliding-window models (Gemma): llama.cpp allocates the cache of those layers for the whole context by default (box checked, `--swa-full 1`). Unchecking **Full SWA cache** (`--swa-full 0`) limits it to the window and uses less VRAM at long context. Rewinding in the sequence is then no longer possible.

A CPU build says that verification does not validate a CUDA estimate.

### Bench

Purpose: compare settings, not a single generation.

Three tabs: **Campaign**, **Results**, **History**.

Example. Q4_K_M model, name "layers and context", GPU layers `0,20,99`, contexts `4096`, KV types `f16,q8_0`, flash attention `1`, ubatch `512`, 512 prompt tokens, 128 generated, 5 repetitions. The page reports how many configurations that is (the product of the lists, capped at 10,000). **Run campaign** runs the measurements. **Results** shows, for each setting, the mean and standard deviation of prefill (pp) and generation (tg), time to first token, peak VRAM, and outliers. **Save** stores the campaign. **CSV** or **Markdown** writes next to the path you give, by default under `Documents\WaffleIron`. **History** plots one model's throughput over time; hovering a point recalls the config and the llama.cpp version.

### Sampler lab

Purpose: see the next token, not only the final text.

Example. Select the model, leave the prompt "Describe a landscape in spring." (or write your own), check **Chat template** if the GGUF has a chat template, then **Start**. Twenty candidates appear: a pale bar for the raw probability, an accent bar after temperature, top-k, top-p, min-p, and penalties. Click a candidate to force that token and open a branch. **Next step** lets the sampler decide. **Generate** runs several steps. The colored text follows the probability of the chosen token. The **Branches** tree jumps back to a node (the cache is recomputed). The sliders apply without reloading the model: temperature 0.8, top-k 40, top-p 0.95, min-p 0.05, seed 42 at the start.

### Comparator

Purpose: the same prompt, two models or two quantizations, one after the other so their VRAM is not added together.

Example. Column A: the Q4_K_M. Column B: the Q8_0 of the same model. Prompt "Explain the water cycle in a few sentences.", 256 tokens, temperature 0, seed 42. **Run**. The texts fill in one after the other. The table marks with a star the best prefill throughput, the best generation throughput, the shortest load, and, if you check **Measure perplexity**, the lowest perplexity on the reference text (a neutral passage already in the field, which you can replace). A warning appears when both sides have exactly the same settings.

### Monitor

Purpose: the GPU while another page is working.

The history window runs from 1 to 10 minutes. The curves cover GPU utilization, VRAM (and its capacity), temperature, power, SM clock, and the generation throughput published by the sampler lab. Without NVML, the page explains why (driver missing, no NVIDIA GPU) and the other pages stay usable.

## Command line

`waffle.exe` sits next to `WaffleIron.exe` in the build directory (`build\cuda-release-local` or `build\cpu-release`). From a terminal:

```text
waffle scan [folders…]         index GGUF models
waffle list                    list the library
waffle info <model>            GGUF metadata
waffle plan <model> --ctx 8192 --ctk q8_0 --ctv q8_0 --fa 1 [--swa-full 0] [--verify]
waffle bench <model> --ngl 0,20,99 --ctx 4096 --reps 5
waffle campaigns | history <model> | export <id> --md
waffle gpu                     GPU state (NVML)
```

`waffle` and `waffle <command> --help` list the options. `<model>` is a file path or a unique substring of the name in the database. `--db PATH` changes the database (default: `%LOCALAPPDATA%\WaffleIron\waffle.db`).

`waffle plan --verify` actually loads the model, measures the VRAM it took, and prints the gap against the estimate, plus the allocations llama.cpp reported. That gap is for the GPU in the machine, not a portable specification.

`waffle bench` accepts comma-separated lists for `--ngl`, `--ctx`, `--ctk`, `--fa`, `--pp`, `--tg`, and `--reps`. The campaign is saved and the Markdown table is printed. `waffle export <id> --csv -o file.csv` or `--md` writes a campaign back out.

## Advanced build

The versioned presets are in `CMakePresets.json`. `install.cmd` does not modify them.

| Preset | Use |
|---|---|
| `cpu-debug` | development, tests |
| `cpu-release` | CPU Release, the one CI uses |
| `cuda-release` | CUDA Release, architecture `sm_89` only (Ada). It does not run on `sm_86` or on Blackwell |
| `asan` | CPU Debug with AddressSanitizer |

When a GPU is detected, `install.cmd` writes `CMakeUserPresets.json` (gitignored): a `cuda-release-local` preset that inherits `cuda-release` and replaces `CMAKE_CUDA_ARCHITECTURES` (for example `86` for Ampere, `89` for Ada, `120` for Blackwell). The binary goes to `build\cuda-release-local`. The repository's `cuda-release` preset stays on `sm_89` and its `build\cuda-release` directory. Use that preset only for an `sm_89` GPU. `install.cmd` is the path that matches the installed card.

```bat
scripts\dev.cmd cmake --preset cpu-debug
scripts\dev.cmd cmake --build build\cpu-debug
scripts\dev.cmd ctest --preset cpu-debug
```

`scripts\dev.cmd` opens the MSVC x64 environment (Visual Studio 2022 or newer, found with `vswhere`, otherwise the Visual Studio 2026 Community path) and points `VCPKG_ROOT` at that install's vcpkg when it is not already set.

To force an architecture without the installer: `install.cmd -Architecture 75`, or `scripts\dev.cmd cmake --preset cuda-release -DCMAKE_CUDA_ARCHITECTURES=75`. The toolkit must know that architecture (Ada `sm_89`: CUDA 11.8 or newer; Blackwell `sm_100` / `sm_120`: CUDA 12.8 or newer).

Tests:

```bat
scripts\dev.cmd ctest --preset cpu-release
install.cmd -Tests
```

Tests that load a model use `WAFFLE_TEST_MODEL` (path to a GGUF), otherwise `tests\data\tiny-llama.gguf` (random weights, produced by `scripts\make_tiny_model.py`).

AddressSanitizer:

```bat
scripts\dev.cmd cmake --preset asan
scripts\dev.cmd cmake --build build\asan
scripts\dev.cmd ctest --preset asan
```

## Where the data lives

| What | Where |
|---|---|
| Library, campaigns, VRAM checks | `%LOCALAPPDATA%\WaffleIron\waffle.db` |
| Interface state | `imgui.ini` next to `WaffleIron.exe` |
| Optional fonts | `assets\fonts\` next to the executable (copied at build time) |
| Binaries | `build\cpu-release\`, `build\cuda-release\`, or `build\cuda-release-local\` |
| Last executable started by the installer | `build\waffle-launch.txt` |
| Local CUDA preset | `CMakeUserPresets.json` at the root (not versioned) |
| Install log | `build\install.log` |
| Campaign exports | the path you choose, by default `Documents\WaffleIron` |

Deleting `waffle.db` starts from an empty library; the next scan rebuilds it. The GGUF files are not in that database, only their records.

## Troubleshooting

**A prerequisite is missing.** `install.cmd` stops and names the component and the link. After changing Visual Studio, open a new terminal: an old `PATH` does not see CMake or Ninja. Run `install.cmd` again.

**`cl.exe`, CMake, or Ninja are missing even though Visual Studio is installed.** Modify the installation and select the MSVC tools and "C++ CMake tools for Windows".

**vcpkg not found.** Select "vcpkg package manager", or clone https://github.com/microsoft/vcpkg , run `bootstrap-vcpkg.bat`, then `set VCPKG_ROOT=C:\vcpkg` before `install.cmd`.

**The submodule is empty.** `install.cmd` runs `git submodule update --init --recursive`. If the folder is not a clone (`git clone`), get the repository that way rather than from a zip.

**The CUDA build fails on the architecture.** The `nvcc` message often means the toolkit is too old for the GPU. Install a suitable toolkit, or set `-Architecture`. With no GPU detected, `-Cuda` falls back to the `sm_89` preset.

**The build is killed (memory).** Run `install.cmd` again: Ninja resumes. If it keeps happening: `scripts\dev.cmd cmake --build build\cuda-release-local --parallel 2`.

**The interface shows no models.** Look at the list on the Library page. The folder must exist before the scan. `mmproj` vision files and shards other than the first are ignored. If Ollama is not under the profile, `OLLAMA_MODELS` must point at the library. **Add a folder** covers a custom LM Studio location.

**"Cannot initialize DirectX 11."** The graphics driver does not provide DirectX 11. Update it. A WARP fallback is tried only in some cases where no hardware driver is present.

**The monitor says NVML is unavailable.** No NVIDIA GPU, or a driver that is too old. The rest of the workbench still works. A CPU build does not block NVML.

**The console shows the wrong characters.** `install.cmd` switches the code page to UTF-8 (`chcp 65001`). Windows Terminal usually does this on its own. The sources and this file are UTF-8.

**The CUDA build does not seem to target the right GPU.** Check the preset printed by `install.cmd` and `CMakeUserPresets.json`. `cmake --preset cuda-release` rebuilds the other directory, the one fixed at `sm_89`.

**Two `WaffleIron.exe`.** `launch.cmd` opens the one recorded in `build\waffle-launch.txt`, otherwise the newest among the known directories.

## Architecture

```
external/llama.cpp/   pinned submodule 4b851fb (updated on purpose, tests as a safety net)
src/core/             no interface code: gguf_reader, model_roots, engine, vram_planner, bench, sampler_lab, comparator, telemetry
src/store/            SQLite
src/cli/              waffle.exe
src/ui/               Dear ImGui + ImPlot + DirectX 11
tests/                Catch2
scripts/dev.cmd       MSVC environment for manual commands
install.cmd           install and launch
launch.cmd            open the last built interface
```

Central rule: **a single thread talks to llama.cpp** (`Engine`). The interface submits work and receives results through `std::future`.

CI (`.github/workflows/ci.yml`) builds and tests the `cpu-release` preset on Windows. It does not use `install.cmd`.

## Fonts

The interface uses Inter and the Lucide icons when `Inter-Regular.ttf`, `Inter-SemiBold.ttf`, and `lucide.ttf` are in `assets/fonts/` (see that folder's README). Otherwise, ImGui's built-in font. Nothing is downloaded at install time.

- Inter: https://github.com/rsms/inter (SIL Open Font License 1.1)
- Lucide: https://github.com/lucide-icons/lucide (ISC, private-use range)

## Developer switches

These environment variables are read on the local machine. They do not send anything anywhere.

| Variable | Effect |
|---|---|
| `WAFFLE_PAGE` | Starting page, 0 through 5 |
| `WAFFLE_SELECT` | Preselect a library row by index |
| `WAFFLE_SCRIPT` | Play a local UI script. The command list is in the comment at the top of `src/ui/main_win32.cpp` |
| `WAFFLE_CAPTURE` | Save a BMP of the window on disk, then quit |
| `WAFFLE_CAPTURE_SIZE` | Client size, written as `WIDTHxHEIGHT` |
| `WAFFLE_CAPTURE_FRAMES` | Frames to draw before the capture |
| `WAFFLE_TEST_MODEL` | GGUF path for tests that load a model |

## License

Waffle Iron is MIT. The text is in `LICENSE` (copyright 2026 Anarchy). Third-party terms are in `THIRD_PARTY.md`. Optional fonts, if you add them under `assets/fonts/`, keep their own licenses.
