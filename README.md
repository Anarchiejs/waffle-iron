# Banc Local

Atelier de bureau natif (C++20, Windows) pour **comprendre, mesurer et régler les modèles locaux** sur sa propre
carte graphique, construit directement sur la bibliothèque de [llama.cpp](https://github.com/ggml-org/llama.cpp)
(`libllama` + `ggml`, backend CUDA).

| Module | Ce qu'il fait |
|---|---|
| Bibliothèque | scanne vos dossiers (`%USERPROFILE%\models`, Ollama, LM Studio), lit les métadonnées GGUF sans charger le modèle, étiquettes et notes |
| Planificateur de VRAM | où part la mémoire (poids GPU/CPU, cache KV, tampons) avant de charger, puis vérification réelle et écart affiché |
| Banc d'essai | campagnes en grille (couches GPU, contexte, cache KV, flash attention), répétitions, moyenne ± écart-type, aberrations, historique SQLite, export CSV / Markdown |
| Laboratoire d'échantillonnage | génération pas à pas, 20 candidats et leurs probabilités, curseurs en direct, arbre de branches |
| Comparateur | même consigne à deux modèles (ou deux quantifications) chargés l'un après l'autre, vitesse et perplexité |
| Moniteur | utilisation GPU, VRAM, température, consommation (NVML), tokens/s en direct |

Deux exécutables : `BancLocal.exe` (interface Dear ImGui, fenêtres ancrables) et `banc.exe` (ligne de commande).

## Construire

Prérequis : Visual Studio 2026 (C++ desktop, vcpkg inclus), CMake ≥ 3.25, Ninja, Git ; CUDA Toolkit pour le build GPU.

```bat
git clone --recurse-submodules <url> banc-local
cd banc-local
scripts\dev.cmd cmake --preset cuda-release
scripts\dev.cmd cmake --build build\cuda-release
```

`scripts\dev.cmd` ouvre l'environnement MSVC x64 et pointe `VCPKG_ROOT` vers le vcpkg de Visual Studio.
Presets : `cpu-debug` (développement), `cpu-release` (CI), `cuda-release` (RTX série 40, `sm_89` — changez
`CMAKE_CUDA_ARCHITECTURES` pour une autre carte), `asan` (AddressSanitizer).

Tests :

```bat
scripts\dev.cmd ctest --preset cpu-debug
```

Les tests qui chargent un modèle utilisent `BANC_TEST_MODEL` (chemin d'un GGUF), sinon le modèle minuscule
`tests/data/tiny-llama.gguf` (poids aléatoires, généré par `scripts/make_tiny_model.py`).

## Ligne de commande

```text
banc scan [dossiers…]          indexe les modèles GGUF
banc list                      liste la bibliothèque
banc info <modèle>             métadonnées GGUF
banc plan <modèle> --ctx 8192 --ctk q8_0 --ctv q8_0 --fa 1 [--swa-full 0] [--verify]
banc bench <modèle> --ngl 0,20,99 --ctx 4096 --reps 5
banc campaigns | history <modèle> | export <id> --md
banc gpu                       état de la carte (NVML)
```

`banc <commande> --help` détaille chaque option. Base de données : `%LOCALAPPDATA%\BancLocal\banc.db`.

`banc plan --verify` charge réellement le modèle, mesure la VRAM prise et affiche l'écart avec l'estimation,
ainsi que les allocations annoncées par llama.cpp. Mesuré sur RTX 4060 Ti avec gemma-4-E4B Q4_K_M : écart
de −1,1 % à +0,4 % tout sur GPU (contexte 4k à 32k, cache f16/q8_0, avec ou sans flash attention) ; +7 % en
déchargement partiel (`--ngl 20`).

Modèles à fenêtre glissante (Gemma) : llama.cpp alloue par défaut le cache SWA sur tout le contexte
(`--swa-full 1`). `--swa-full 0` (case « Cache SWA complet » dans l'interface) le limite à la fenêtre :
5,3 Go → 4,0 Go de VRAM à 32k tokens pour gemma-4-E4B.

## Architecture

```
external/llama.cpp/   sous-module épinglé (mis à jour volontairement, tests en garde-fou)
src/core/             aucun code d'interface : gguf_reader, engine, vram_planner, bench, sampler_lab, comparator, telemetry
src/store/            SQLite
src/cli/              banc.exe
src/ui/               Dear ImGui (docking) + ImPlot + DirectX 11
tests/                Catch2
```

Règle centrale : **un seul thread parle à llama.cpp** (`Engine`). L'interface lui soumet des travaux et reçoit les
résultats par `std::future` ; elle ne se fige jamais pendant un chargement.

## Polices

L'interface utilise Inter et les icônes Lucide si elles sont présentes dans `assets/fonts/` (voir le README de ce
dossier) ; sinon la police intégrée d'ImGui.

## Licence

MIT, comme llama.cpp.
