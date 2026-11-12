# Contributing

Waffle Iron builds on Windows x64 (MSVC), Linux x64 (GCC or Clang), and macOS (Apple Clang). The setup for each platform is in `README.md`.

Windows:

```bat
git clone --recurse-submodules https://github.com/Anarchiejs/Banc-Local.git waffle-iron
cd waffle-iron
install.cmd -Tests
```

Linux and macOS:

```bash
git clone --recurse-submodules https://github.com/Anarchiejs/Banc-Local.git waffle-iron
cd waffle-iron
export VCPKG_ROOT="$HOME/vcpkg"   # after bootstrap-vcpkg.sh, if it is not already set
./install.sh --tests
```

`install.cmd -Tests` and `./install.sh --tests` configure the CPU, CUDA, or Metal preset for this machine and run the Catch2 suite. CI runs the `cpu-release` preset on Windows, Linux, and macOS. It does not run a full installer build, a CUDA build, a Metal build, or the graphical interface.

Before sending a change:

- Format C and C++ sources with clang-format 21.1.2, using the repository `.clang-format`. CI rejects a diff. Do not run clang-format on `CMakeLists.txt`.
- Keep new user-visible strings in English.
- Do not commit `build/`, `CMakeUserPresets.json`, `.env`, `imgui.ini`, logs, or font binaries. Optional fonts stay under their own licenses (`THIRD_PARTY.md`, `assets/fonts/README.md`).
- Do not add personal paths, email addresses, or machine names. The public identity of this project is Anarchy (https://github.com/Anarchiejs).
- Keep Windows behavior the same when adding a Linux or macOS path. `install.cmd`, `scripts/dev.cmd`, and `src/ui/main_win32.cpp` stay the Windows entry points. Shell scripts sit next to them; they do not replace them.

The llama.cpp submodule is updated on purpose. A bump needs the tests still passing on `cpu-release`.

The GitHub repository is named Banc-Local today. If it is renamed to `waffle-iron`, use the new clone URL. GitHub redirects the old one.
