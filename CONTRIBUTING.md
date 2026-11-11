# Contributing

Waffle Iron builds on Windows x64 with MSVC, CMake, Ninja, and vcpkg. The setup is in `README.md`.

```bat
git clone --recurse-submodules https://github.com/Anarchiejs/Banc-Local.git waffle-iron
cd waffle-iron
install.cmd -Tests
```

`install.cmd -Tests` configures the CPU or CUDA preset for this machine and runs the Catch2 suite. CI runs the `cpu-release` preset only. It does not run `install.cmd`, a CUDA build, or the DirectX interface.

Before sending a change:

- Format C and C++ sources with clang-format 21.1.2, using the repository `.clang-format`. CI rejects a diff.
- Keep new user-visible strings in English.
- Do not commit `build\`, `CMakeUserPresets.json`, `.env`, `imgui.ini`, logs, or font binaries. Optional fonts stay under their own licenses (`THIRD_PARTY.md`, `assets/fonts/README.md`).
- Do not add personal paths, email addresses, or machine names. The public identity of this project is Anarchy (https://github.com/Anarchiejs).

The llama.cpp submodule is updated on purpose. A bump needs the tests still passing on `cpu-release`.

The GitHub repository is named Banc-Local today. If it is renamed to `waffle-iron`, use the new clone URL. GitHub redirects the old one.
