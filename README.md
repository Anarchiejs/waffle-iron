# Waffle Iron

Waffle Iron is a desktop workbench for **understanding, measuring, and tuning the GGUF models** that already run on your machine. It is for anyone who wants to know what a model costs in memory, how fast it answers, and how the cache, the context, or the quantization changes the result — on Windows, Linux, or macOS, without sending the files anywhere else.

The name is a pun: GGUF is pronounced like *gaufre*, the French word for waffle.

The tool is native (C++20). It is built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (`libllama` and `ggml`). The CUDA backend is used when an NVIDIA GPU and the CUDA Toolkit are present (Windows and Linux). macOS uses the Metal backend for Apple GPUs. The views are Dear ImGui: DirectX 11 on Windows, GLFW and OpenGL on Linux and macOS.

Two programs:

| Program | Role |
|---|---|
| `WaffleIron` (`WaffleIron.exe` on Windows) | The interface: library, memory, measurements, and generation |
| `waffle` (`waffle.exe` on Windows) | The same core on the command line, for scripting a scan, a plan, or a campaign |

Supported platforms: Windows x64, Linux x64, and macOS (Apple Silicon and Intel).

The GitHub repository is [waffle-iron](https://github.com/Anarchiejs/waffle-iron).

## Limits

- Windows x64, Linux x64, and macOS. The Windows interface needs DirectX 11. The Linux and macOS interface needs OpenGL 3.2 (or 3.3) and, on Linux, an X11 session or XWayland. The command line does not need a display. On Windows it reads arguments with the Win32 API; on Linux and macOS it uses the process argument vector.
- A CPU build runs models on the processor. CUDA (Windows and Linux) needs an NVIDIA driver and the CUDA Toolkit. The monitor reads NVIDIA GPU index 0 through NVML. On macOS the monitor reads the Metal device name and this process's allocation against the recommended working set; utilization, temperature, power, and SM clock stay at zero because those readings are NVML-only. Other GPUs have no CUDA backend and no NVML monitor.
- The checked-in `cuda-release` preset compiles for `sm_89` only. That cubin does not run on `sm_86` or on Blackwell (`sm_100`, `sm_120`). `install.cmd` and `./install.sh` write a gitignored preset for the NVIDIA GPU that is actually installed. macOS does not use that preset: `metal-release` targets the Apple GPU.
- Waffle Iron does not download models and does not contact Ollama, LM Studio, or `registry.ollama.ai`. That name is only a prefix in local manifest paths. The scan reads GGUF files on disk and, when those folders exist, Ollama blobs and manifest JSON, so it can name a digest and skip draft and projector blobs.
- Nothing is uploaded. "Telemetry" in the source is local NVML sampling, or a local Metal query on macOS. The first build does use the network, for the llama.cpp submodule and the vcpkg ports.
- The library database is per user (see [Where the data lives](#where-the-data-lives)). It stores the full path of each file. Interface state is `imgui.ini` next to the executable. Exports go to the path you choose.
- Memory estimates are approximate. **Verify for real** checks them on the installed GPU when a memory reading is available. On macOS that reading is the Metal allocation of this process; if Metal does not report one, the model still loads and the gap is left unmeasured. A result from one card does not transfer to another. The fixed GPU overhead in the estimate was calibrated against CUDA.
- Waffle Iron is not an official project of ggml, Ollama, LM Studio, NVIDIA, or Apple. No model weights are included, except the random-weight test fixture `tests/data/tiny-llama.gguf`.

## Requirements

CMake ≥ 3.25, Ninja, Git, and vcpkg are required on every platform. A C++20 compiler is required. The first configure downloads the vcpkg ports (SQLite, Dear ImGui, ImPlot, Catch2, and GLFW on Linux and macOS).

### Windows

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

### Linux

| Tool | Required | Notes |
|---|---|---|
| Git | yes | distribution package |
| GCC 11+ or Clang 14+ | yes | Debian/Ubuntu: `build-essential`. Fedora: `gcc-c++`. Arch: `base-devel` |
| CMake ≥ 3.25 and Ninja | yes | `cmake` and `ninja-build` |
| zip | yes | vcpkg's `bootstrap-vcpkg.sh` uses it. Debian/Ubuntu and Arch: `zip`. Fedora: `zip` |
| vcpkg | yes | see the install steps below. Clone it without `--depth` |
| OpenGL and GLFW's X11 headers | yes, for the interface | see the package line below |
| Autotools | yes, for the interface | vcpkg builds some X11 dependencies from source |
| NVIDIA driver | only for NVML and a CUDA build | the monitor loads `libnvidia-ml.so.1` from the driver |
| CUDA Toolkit (`nvcc`) | no | https://developer.nvidia.com/cuda-downloads |

Debian/Ubuntu:

```bash
sudo apt-get install -y build-essential git cmake ninja-build pkg-config zip \
  autoconf autoconf-archive automake libtool xutils-dev \
  libgl1-mesa-dev libglu1-mesa-dev \
  libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libxext-dev libxxf86vm-dev
```

Fedora:

```bash
sudo dnf install git gcc-c++ cmake ninja-build pkgconf zip \
  autoconf autoconf-archive automake libtool xorg-x11-util-macros \
  mesa-libGL-devel mesa-libGLU-devel \
  libX11-devel libXrandr-devel libXinerama-devel libXcursor-devel libXi-devel libXext-devel libXxf86vm-devel
```

Arch Linux:

```bash
sudo pacman -S --needed base-devel git cmake ninja pkgconf zip \
  mesa glu libx11 libxrandr libxinerama libxcursor libxi libxext libxxf86vm
```

The interface needs an X11 display or XWayland. A Wayland-only session without XWayland can still build and use `waffle`. Pass `-DWAFFLE_BUILD_UI=OFF` to configure a command-line-only build.

### macOS

| Tool | Required | Notes |
|---|---|---|
| Xcode or the Command Line Tools | yes | `xcode-select --install`. The Metal compiler (`xcrun metal`) ships with Xcode and is required for `metal-release` |
| CMake ≥ 3.25 and Ninja | yes | `brew install cmake ninja` if they are not already installed |
| vcpkg | yes | see the install steps below |
| CUDA Toolkit | no | not used. Apple GPUs go through Metal |

A CPU build (`--cpu`, preset `cpu-release`) does not need the Metal compiler. The default `./install.sh` selects Metal when `xcrun metal` is available.

## Quick install

Windows:

```bat
git clone --recurse-submodules https://github.com/Anarchiejs/waffle-iron.git waffle-iron
cd waffle-iron
install.cmd
```

Linux and macOS, after CMake, Ninja, a C++ compiler, and vcpkg are installed:

```bash
git clone --recurse-submodules https://github.com/Anarchiejs/waffle-iron.git waffle-iron
cd waffle-iron
# if vcpkg is not already on the machine, ./install.sh clones it.
# Do not use git clone --depth: vcpkg manifest mode needs the builtin-baseline
# commit recorded in vcpkg.json, and a shallow clone does not contain it.
./install.sh
```

`install.cmd` (Windows) and `./install.sh` (Linux and macOS) are the commands to run after the clone. Each one:

1. checks the platform toolchain (Visual Studio and MSVC on Windows; a C++ compiler, CMake ≥ 3.25, Ninja, and vcpkg everywhere) and says what to select or download if something is missing;
2. fetches the `external/llama.cpp` submodule if `--recurse-submodules` was forgotten (a GitHub zip is not enough: it does not contain the submodule);
3. on Windows and Linux, queries `nvidia-smi` and looks for `nvcc`; on macOS, looks for `xcrun metal`;
4. chooses a CUDA build for the GPU architecture (`cuda-release-local`), a Metal build (`metal-release`) on macOS, or `cpu-release` when there is no usable GPU backend;
5. configures, builds, then opens `WaffleIron`.

The first run downloads the vcpkg dependencies (SQLite, Dear ImGui, ImPlot, Catch2, and GLFW on Linux and macOS): it needs the internet. A CUDA or Metal build of llama.cpp is long and resumes where it stopped if you run the installer again. The log is `build/install.log` (`build\install.log` on Windows).

To open the interface later, without rebuilding:

```bat
launch.cmd
```

```bash
./launch.sh
```

Windows variants:

```bat
install.cmd -NoLaunch
install.cmd -Cpu
install.cmd -Cuda
install.cmd -Architecture 86
install.cmd -Architecture 8.9
install.cmd -Tests
install.cmd -Help
```

Linux and macOS variants (the Windows names, such as `-Cpu`, are also accepted):

```bash
./install.sh --no-launch
./install.sh --cpu
./install.sh --cuda
./install.sh --architecture 86
./install.sh --architecture 8.9
./install.sh --tests
./install.sh --help
```

`--architecture` / `-Architecture` accepts `89` or `8.9`, and a list (`86,89`) when several NVIDIA GPUs must be targeted. CPU and CUDA together are rejected. On macOS, `--cuda` and `--architecture` are rejected: the GPU build is Metal, and `--cpu` forces the CPU preset.

## First launch

The library database (see [Where the data lives](#where-the-data-lives)) is created on first start. If it is empty, the library stays empty for a few moments: a scan runs in the background, without blocking the window, then the models it found appear. A toast says "Scan finished".

That scan runs again on every launch, even when the Library page is not the one on screen. There is no need to run `waffle scan` by hand for the first fill. **Rescan**, or Ctrl+R, starts it again after you add files.

The sidebar lists the six pages. Ctrl+1 through Ctrl+6 switch pages. The status bar shows the work in progress, the GPU name, and VRAM when NVML answers.

A single thread talks to llama.cpp. The interface sends it work and keeps drawing during a load.

## Where to put models

Waffle Iron does not download anything. It indexes the GGUF files already on disk. The folders it looks at, when they exist:

The profile directory is `%USERPROFILE%` on Windows and `$HOME` on Linux and macOS.

| Location | Usual contents |
|---|---|
| `<profile>/models` | `.gguf` files you put there yourself |
| `<profile>/.ollama/models/blobs` | Ollama blobs (often with no extension; the `GGUF` signature is enough) |
| `<profile>/.lmstudio/models` | LM Studio models |
| `<profile>/.cache/lm-studio/models` | older LM Studio cache folder |
| `/usr/share/ollama/.ollama/models/blobs` | Linux fallback when the service user is the official `ollama` account. The scan also reads `OLLAMA_MODELS` from the process and from the systemd unit, and the home directory of the `ollama` user |
| `$OLLAMA_MODELS/blobs`, or `$OLLAMA_MODELS` when there is no `blobs` subfolder | a moved Ollama library (`OLLAMA_MODELS` on every platform) |

On Windows those profile paths are `%USERPROFILE%\models`, `%USERPROFILE%\.ollama\models\blobs`, `%USERPROFILE%\.lmstudio\models`, and `%USERPROFILE%\.cache\lm-studio\models`.

Example: copy `gemma-4-E4B-Q4_K_M.gguf` into the profile `models` folder, then open Waffle Iron (or click **Rescan**). The model shows up with its name, architecture, quantization, size, parameter count, and context.

The Library ignores files that are not GGUF, vision projectors (`mmproj`, architecture `clip`), and split model parts other than the first (`…-00002-of-…`). Another folder is added with **Add a folder** (UTF-8 path). The record of a selected model accepts comma-separated tags and notes, stored in the database.

If none of the folders in the list exist, the page shows them as "absent": create the profile `models` folder and put the files there.

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

`waffle` sits next to `WaffleIron` in the build directory (`build/cuda-release-local`, `build/metal-release`, or `build/cpu-release`; Windows paths use backslashes and a `.exe` suffix). From a terminal:

```text
waffle scan [folders…]         index GGUF models
waffle list                    list the library
waffle info <model>            GGUF metadata
waffle plan <model> --ctx 8192 --ctk q8_0 --ctv q8_0 --fa 1 [--swa-full 0] [--verify]
waffle bench <model> --ngl 0,20,99 --ctx 4096 --reps 5
waffle campaigns | history <model> | export <id> --md
waffle gpu                     GPU state (NVML)
```

`waffle` and `waffle <command> --help` list the options. `<model>` is a file path or a unique substring of the name in the database. `--db PATH` changes the database. The default path is in [Where the data lives](#where-the-data-lives).

`waffle plan --verify` actually loads the model, measures the VRAM it took, and prints the gap against the estimate, plus the allocations llama.cpp reported. That gap is for the GPU in the machine, not a portable specification.

`waffle bench` accepts comma-separated lists for `--ngl`, `--ctx`, `--ctk`, `--fa`, `--pp`, `--tg`, and `--reps`. The campaign is saved and the Markdown table is printed. `waffle export <id> --csv -o file.csv` or `--md` writes a campaign back out.

## Advanced build

The versioned presets are in `CMakePresets.json`. `install.cmd` and `./install.sh` do not modify them. `metal-release` is listed only on macOS. `cuda-release` is listed on Windows and Linux.

| Preset | Use |
|---|---|
| `cpu-debug` | development, tests. `WAFFLE_METAL` is off, including on macOS |
| `cpu-release` | CPU Release, the one CI uses on Windows, Linux, and macOS |
| `cuda-release` | CUDA Release, architecture `sm_89` only (Ada). Windows and Linux. It does not run on `sm_86` or on Blackwell |
| `metal-release` | Metal Release on macOS |
| `asan` | CPU Debug with AddressSanitizer (MSVC `/fsanitize=address`, GCC/Clang `-fsanitize=address`). On Linux and macOS this links the release vcpkg libraries, so a release-only vcpkg install is enough |

When an NVIDIA GPU is detected, the installer writes `CMakeUserPresets.json` (gitignored): a `cuda-release-local` preset that inherits `cuda-release` and replaces `CMAKE_CUDA_ARCHITECTURES` (for example `86` for Ampere, `89` for Ada, `120` for Blackwell). The binary goes to `build/cuda-release-local`. The repository's `cuda-release` preset stays on `sm_89` and its `build/cuda-release` directory. Use that preset only for an `sm_89` GPU. The installer is the path that matches the installed card. On macOS the installer selects `metal-release` and does not write a user preset.

Windows:

```bat
scripts\dev.cmd cmake --preset cpu-debug
scripts\dev.cmd cmake --build build\cpu-debug
scripts\dev.cmd ctest --preset cpu-debug
```

`scripts\dev.cmd` opens the MSVC x64 environment (Visual Studio 2022 or newer, found with `vswhere`, otherwise the Visual Studio 2026 Community path) and points `VCPKG_ROOT` at that install's vcpkg when it is not already set.

Linux and macOS (`scripts/dev.sh` exports `VCPKG_ROOT` when it can find a checkout):

```bash
export VCPKG_ROOT="$HOME/vcpkg"   # if it is not already set
scripts/dev.sh cmake --preset cpu-debug
scripts/dev.sh cmake --build --preset cpu-debug
scripts/dev.sh ctest --preset cpu-debug
```

On macOS, `cmake --preset metal-release` builds the Metal backend. A configure without a preset also turns Metal on, because that is llama.cpp's Apple default. The CPU presets force it off.

To force an NVIDIA architecture without the installer: `install.cmd -Architecture 75` or `./install.sh --architecture 75`, or `cmake --preset cuda-release -DCMAKE_CUDA_ARCHITECTURES=75`. The toolkit must know that architecture (Ada `sm_89`: CUDA 11.8 or newer; Blackwell `sm_100` / `sm_120`: CUDA 12.8 or newer).

Tests:

```bat
scripts\dev.cmd ctest --preset cpu-release
install.cmd -Tests
```

```bash
scripts/dev.sh ctest --preset cpu-release
./install.sh --tests
```

Tests that load a model use `WAFFLE_TEST_MODEL` (path to a GGUF), otherwise `tests/data/tiny-llama.gguf` (random weights, produced by `scripts/make_tiny_model.py`).

AddressSanitizer:

```bash
scripts/dev.sh cmake --preset asan
scripts/dev.sh cmake --build --preset asan
scripts/dev.sh ctest --preset asan
```

On Windows, run those three commands through `scripts\dev.cmd` instead of `scripts/dev.sh`.

## Where the data lives

| What | Windows | Linux | macOS |
|---|---|---|---|
| Library, campaigns, VRAM checks | `%LOCALAPPDATA%\WaffleIron\waffle.db` | `$XDG_DATA_HOME/WaffleIron/waffle.db`, or `~/.local/share/WaffleIron/waffle.db` | `~/Library/Application Support/WaffleIron/waffle.db` |
| Interface state | `imgui.ini` next to the executable | same | same |
| Icon font and optional Inter | `assets/fonts/` next to the executable (copied at build time; `lucide.ttf` is included) | same | same |
| Binaries | `build\cpu-release\`, `build\cuda-release\`, or `build\cuda-release-local\` | `build/cpu-release/`, `build/cuda-release/`, or `build/cuda-release-local/` | `build/cpu-release/` or `build/metal-release/` |
| Last executable started by the installer | `build\waffle-launch.txt` | `build/waffle-launch.txt` | `build/waffle-launch.txt` |
| Local CUDA preset | `CMakeUserPresets.json` at the root (not versioned) | same | not written (Metal has no CUDA architecture) |
| Install log | `build\install.log` | `build/install.log` | `build/install.log` |
| Campaign exports | the path you choose, by default `Documents\WaffleIron` | `XDG_DOCUMENTS_DIR/WaffleIron` when `~/.config/user-dirs.dirs` or `XDG_DOCUMENTS_DIR` says so, otherwise `~/Documents/WaffleIron` | `~/Documents/WaffleIron` |

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

**The CUDA build does not seem to target the right GPU.** Check the preset printed by `install.cmd` or `./install.sh` and `CMakeUserPresets.json`. `cmake --preset cuda-release` rebuilds the other directory, the one fixed at `sm_89`.

**Two `WaffleIron` binaries.** `launch.cmd` and `./launch.sh` open the one recorded in `build/waffle-launch.txt`, otherwise the newest among the known directories.

**Linux: the interface does not open.** The GLFW build uses X11. A Wayland session needs XWayland. `waffle` does not need a display. If CMake cannot find OpenGL, install the Mesa development package listed under Requirements.

**macOS: `./install.sh` built the CPU preset.** `xcrun metal` was not found. Install Xcode (the command-line tools alone do not always include the Metal compiler) and run `./install.sh` again, or configure `metal-release` yourself.

**`VCPKG_ROOT` is empty on Linux or macOS.** `./install.sh` clones vcpkg into `$HOME/vcpkg` without `--depth`, bootstraps it, and fetches the `builtin-baseline` commit when that commit is missing. A shallow clone (`git clone --depth 1`) does not contain that commit, and manifest mode then fails. `scripts/dev.sh` looks in `$HOME/vcpkg`, `/opt/vcpkg`, and `/usr/local/vcpkg` when the variable is unset.

**Linux: vcpkg or GLFW fails while configuring.** Install `zip` and the X11 development packages in the Linux requirements. `./install.sh` checks for them before configure.

## Architecture

```
external/llama.cpp/   pinned submodule 4b851fb (updated on purpose, tests as a safety net)
src/core/             no interface code: gguf_reader, model_roots, engine, vram_planner, bench, sampler_lab, comparator, telemetry
src/store/            SQLite
src/cli/              waffle
src/ui/               Dear ImGui + ImPlot; DirectX 11 on Windows (main_win32.cpp), GLFW + OpenGL elsewhere (main_glfw.cpp)
tests/                Catch2
scripts/dev.cmd       MSVC environment for manual commands
scripts/dev.sh        VCPKG_ROOT for manual commands on Linux and macOS
scripts/install.ps1   Windows installer used by install.cmd
scripts/install.sh    Linux and macOS installer used by install.sh
install.cmd           Windows install and launch
launch.cmd            Windows: open the last built interface
install.sh            Linux and macOS install and launch
launch.sh             Linux and macOS: open the last built interface
```

Central rule: **a single thread talks to llama.cpp** (`Engine`). The interface submits work and receives results through `std::future`.

CI (`.github/workflows/ci.yml`) builds and tests the `cpu-release` preset on Windows, Linux, and macOS. It does not use `install.cmd` or `./install.sh` to compile. The Linux and macOS jobs do run `./install.sh --auto-test`, which checks GPU-backend selection without compiling. CUDA and Metal presets are not built in CI: the hosted runners have no NVIDIA GPU, and the macOS job stays on the CPU preset so it matches the Windows and Linux test.

## Fonts

`lucide.ttf` is shipped in `assets/fonts/` and copied next to `WaffleIron` for every preset that builds the interface (`NOTICE` has the ISC and Feather terms). Inter is optional: place `Inter-Regular.ttf` and `Inter-SemiBold.ttf` in that same folder. Otherwise the interface uses a system font (Segoe UI on Windows, DejaVu or Liberation on Linux, Arial on macOS) and then ImGui's built-in font. Sidebar icons use the Lucide font when it was copied; without it they are drawn as strokes, because a missing private-use glyph shows up as a box.

- Inter: https://github.com/rsms/inter (SIL Open Font License 1.1). Not included.
- Lucide: https://github.com/lucide-icons/lucide (ISC, plus MIT for the Feather icons it incorporates). Included as `assets/fonts/lucide.ttf`.

## Developer switches

These environment variables are read on the local machine. They do not send anything anywhere.

| Variable | Effect |
|---|---|
| `WAFFLE_PAGE` | Starting page, 0 through 5 |
| `WAFFLE_SELECT` | Preselect a library row by index |
| `WAFFLE_SCRIPT` | Play a local UI script. The command list is in the comment at the top of `src/ui/main_win32.cpp` and `src/ui/main_glfw.cpp` |
| `WAFFLE_CAPTURE` | Save a BMP of the window on disk, then quit |
| `WAFFLE_CAPTURE_SIZE` | Client size, written as `WIDTHxHEIGHT` |
| `WAFFLE_CAPTURE_FRAMES` | Frames to draw before the capture |
| `WAFFLE_TEST_MODEL` | GGUF path for tests that load a model |

## License

Waffle Iron is MIT. The text is in `LICENSE` (copyright 2026 Anarchy). Third-party terms are in `THIRD_PARTY.md`. The included Lucide font is covered by `NOTICE`. Optional fonts, if you add them under `assets/fonts/`, keep their own licenses.
