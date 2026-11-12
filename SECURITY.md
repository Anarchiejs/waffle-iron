# Security

Waffle Iron is a local program (Windows, Linux, and macOS). It does not open a network service, and it does not upload models, prompts, or GPU readings.

It parses GGUF files with the pinned [llama.cpp](https://github.com/ggml-org/llama.cpp) submodule (`external/llama.cpp`, see `THIRD_PARTY.md`). A malformed model file is handled by that parser. Treat model files as input you chose to open.

## Reporting a vulnerability

Please use GitHub private vulnerability reporting:

https://github.com/Anarchiejs/waffle-iron/security/advisories/new

If the repository is renamed to `waffle-iron`, use the security page of the new URL. GitHub redirects the old one.

Include the Waffle Iron version, the llama.cpp version shown in the About box, whether the build is CPU, CUDA, or Metal, and the operating system version. Do not send model weights or private paths unless they are required to understand the bug, and then only through the private report.
