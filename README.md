# Banc Local

Banc Local est un atelier de bureau pour **comprendre, mesurer et régler les modèles GGUF** qui tournent déjà sur votre machine. Il s'adresse à qui veut savoir ce qu'un modèle coûte en mémoire, à quelle vitesse il répond, et comment le cache, le contexte ou la quantification changent le résultat — sans quitter Windows et sans envoyer les fichiers ailleurs.

L'outil est natif (C++20). Il s'appuie sur [llama.cpp](https://github.com/ggml-org/llama.cpp) (`libllama` et `ggml`, backend CUDA si la carte et le toolkit sont là) et affiche ses vues avec Dear ImGui et DirectX 11.

Deux programmes :

| Programme | Rôle |
|---|---|
| `BancLocal.exe` | L'interface : fenêtres ancrables, bibliothèque, mémoire, mesures, génération |
| `banc.exe` | La même base en ligne de commande, pour scripter un scan, un plan ou une campagne |

Windows est la seule plateforme visée.

## Prérequis

| Outil | Obligatoire | Où l'installer |
|---|---|---|
| Git pour Windows | oui | https://git-scm.com/download/win |
| Visual Studio 2022 ou plus récent (2026 convient), charge de travail **Développement Desktop en C++** | oui | https://visualstudio.microsoft.com/fr/downloads/ |
| Outils MSVC C++ (x64) | oui, inclus dans la charge | composant de l'installateur Visual Studio |
| Outils CMake C++ pour Windows (CMake ≥ 3.25 et Ninja) | oui | même installateur, coché avec la charge C++ |
| Gestionnaire de package vcpkg | oui | même installateur : « Gestionnaire de package vcpkg » |
| Pilote NVIDIA récent | seulement pour le moniteur GPU et un build CUDA | via GeForce Experience ou le site NVIDIA |
| CUDA Toolkit | non | https://developer.nvidia.com/cuda-downloads |

Les Build Tools Visual Studio suffisent à la place de l'IDE, avec les mêmes composants. CMake et Ninja installés à part (`https://cmake.org/download/`, `https://ninja-build.org/`) sont reconnus s'ils sont sur le `PATH` ou à leur emplacement habituel.

Sans CUDA Toolkit, l'interface se compile quand même : les modèles tournent sur le processeur. Le moniteur peut tout de même lire la carte, parce que NVML vient du pilote, pas du toolkit.

## Installation rapide

```bat
git clone --recurse-submodules <url> banc-local
cd banc-local
install.cmd
```

`install.cmd` est la seule commande à lancer après le clone. Elle :

1. vérifie Git, Visual Studio, MSVC, CMake ≥ 3.25, Ninja et vcpkg, et dit quoi cocher ou télécharger s'il manque quelque chose ;
2. récupère le sous-module `external/llama.cpp` si `--recurse-submodules` a été oublié (un zip GitHub ne suffit pas : il n'a pas le sous-module) ;
3. interroge `nvidia-smi` et cherche `nvcc` ;
4. choisit un build CUDA pour l'architecture de la carte (`cuda-release-local`), ou `cpu-release` s'il n'y a pas de GPU NVIDIA utilisable avec un toolkit ;
5. configure, compile, puis ouvre `BancLocal.exe`.

Le premier passage télécharge les dépendances vcpkg (SQLite, spdlog, Dear ImGui, ImPlot, Catch2) : il faut Internet. Une compilation CUDA de llama.cpp est longue et reprend là où elle s'est arrêtée si vous relancez `install.cmd`. Le journal est `build\install.log`.

Pour rouvrir l'interface plus tard, sans recompiler :

```bat
lancer.cmd
```

Variantes :

```bat
install.cmd -SansLancement
install.cmd -Cpu
install.cmd -Cuda
install.cmd -Architecture 86
install.cmd -Architecture 8.9
install.cmd -Tests
install.cmd -Aide
```

`-Architecture` accepte `89` ou `8.9`, et une liste (`86,89`) si plusieurs cartes doivent être visées. `-Cpu` et `-Cuda` ensemble sont refusés.

## Premier lancement

La base `%LOCALAPPDATA%\BancLocal\banc.db` est créée au premier démarrage. Si elle est vide, la bibliothèque est vide pendant quelques instants : un scan tourne en arrière-plan, sans bloquer la fenêtre, puis les modèles trouvés apparaissent. Un bandeau indique « Scan terminé ».

Ce scan se relance à chaque ouverture, même si le panneau Bibliothèque a été replié. Inutile de lancer `banc scan` à la main pour le premier remplissage. Le menu **Fichier → Rescanner les modèles**, ou le bouton **Rescanner**, recommence quand vous avez ajouté des fichiers.

La disposition de départ place la Bibliothèque à gauche, le Moniteur en bas, et les quatre autres modules en onglets au centre. Le menu **Affichage** rouvre un panneau ou réinitialise la disposition. La barre d'état montre le travail en cours, le nom de la carte et la VRAM si NVML répond.

Un seul thread parle à llama.cpp. L'interface lui envoie le travail et continue de s'afficher pendant un chargement.

## Où mettre les modèles

Banc Local ne télécharge rien. Il indexe les fichiers GGUF déjà sur le disque. Les dossiers examinés, s'ils existent :

| Emplacement | Contenu habituel |
|---|---|
| `%USERPROFILE%\models` | vos fichiers `.gguf` déposés à la main |
| `%USERPROFILE%\.ollama\models\blobs` | blobs Ollama (souvent sans extension ; la signature `GGUF` suffit) |
| `%USERPROFILE%\.lmstudio\models` | modèles LM Studio |
| `%USERPROFILE%\.cache\lm-studio\models` | ancien dossier de cache LM Studio |
| `%OLLAMA_MODELS%\blobs`, ou `%OLLAMA_MODELS%` s'il n'y a pas de sous-dossier `blobs` | bibliothèque Ollama déplacée |

Exemple : copier `gemma-4-E4B-Q4_K_M.gguf` dans `C:\Users\Vous\models`, puis ouvrir Banc Local (ou cliquer **Rescanner**). Le modèle apparaît avec son nom, son architecture, sa quantification, sa taille, son nombre de paramètres et son contexte.

La Bibliothèque ignore les fichiers qui ne sont pas des GGUF, les projecteurs de vision (`mmproj`, architecture `clip`) et les morceaux de modèles découpés autres que le premier (`…-00002-of-…`). Un autre dossier s'ajoute avec **Ajouter un dossier...** (chemin UTF-8). La fiche d'un modèle sélectionné accepte des étiquettes séparées par des virgules et des notes, enregistrées dans la base.

Si aucun dossier de la liste n'existe, le panneau les affiche avec la mention « absent » : créez `%USERPROFILE%\models` et déposez-y les fichiers.

## Visite de l'interface

Les noms ci-dessous sont ceux des panneaux et des contrôles.

### Bibliothèque

But : voir ce que la machine possède déjà, sans charger les poids.

Exemple. Vous avez un Q4_K_M pour le quotidien et un Q8_0 pour comparer. Après le scan, filtrez avec « Filtrer les modèles... », cliquez la ligne Q4_K_M, mettez l'étiquette `quotidien` et une note (« tient en 8 Go à 8k »), puis **Enregistrer**. Le bandeau de répartition montre les octets par type de tenseur. **Métadonnées** ouvre les clés GGUF, avec un filtre.

### Planificateur de VRAM

But : savoir où part la mémoire avant de charger, puis vérifier sur la carte.

Exemple, sur une RTX 4060 Ti avec gemma-4-E4B en Q4_K_M. Choisissez le modèle. Réglez le contexte (4k, 8k, 32k…), le cache K et le cache V (`f16`, `q8_0`, `q4_0`…), la flash attention et la case **Cache SWA complet**. Le graphique empile poids, cache KV, tampons et overhead, côté GPU et CPU, avec la VRAM totale et la VRAM déjà prise. **Couches max qui tiennent** propose un déchargement qui reste dans la mémoire libre. **Vérifier réellement** charge le modèle, mesure l'écart, le décharge et l'inscrit dans l'historique.

Mesures de référence sur cette carte, gemma-4-E4B Q4_K_M : écart de −1,1 % à +0,4 % tout sur GPU (contexte 4k à 32k, cache f16 ou q8_0, avec ou sans flash attention) ; +7 % en déchargement partiel (`--ngl 20`). La mesure NVML est globale : un autre programme peut la faire bouger.

Modèles à fenêtre glissante (Gemma) : llama.cpp alloue par défaut le cache de ces couches sur tout le contexte (case cochée, `--swa-full 1`). Décocher **Cache SWA complet** (`--swa-full 0`) le limite à la fenêtre. Sur gemma-4-E4B, à 32k tokens : 5,3 Go → 4,0 Go de VRAM. Revenir en arrière dans la séquence n'est alors plus possible.

Un build CPU affiche que la vérification ne valide pas une estimation CUDA.

### Banc d'essai

But : comparer des réglages, pas une seule génération.

Trois onglets : **Campagne**, **Résultats**, **Historique**.

Exemple. Modèle Q4_K_M, nom « couches et contexte », couches GPU `0,20,99`, contextes `4096`, types KV `f16,q8_0`, flash attention `1`, ubatch `512`, 512 tokens de consigne, 128 générés, 5 répétitions. Le panneau annonce le nombre de configurations (produit des listes, plafond 10 000). **Lancer** enchaîne les mesures. **Résultats** montre, pour chaque réglage, la moyenne et l'écart-type du préremplissage (pp) et de la génération (tg), le temps jusqu'au premier token, le pic de VRAM et les aberrations. **Enregistrer** range la campagne. **Exporter CSV** ou **Exporter Markdown** écrit à côté du chemin proposé, par défaut sous `Documents\BancLocal`. **Historique** trace les débits d'un modèle dans le temps ; le survol d'un point rappelle la config et la version de llama.cpp.

### Laboratoire d'échantillonnage

But : voir le prochain token, pas seulement le texte final.

Exemple. Sélectionnez le modèle, laissez la consigne « Décris un paysage au printemps. » (ou la vôtre), cochez **Modèle de conversation** si le GGUF a un modèle de chat, puis **Démarrer**. Vingt candidats s'affichent : barre pâle pour la probabilité brute, barre accentuée après température, top-k, top-p, min-p et pénalités. Cliquez un candidat pour forcer ce token et ouvrir une branche. **Pas suivant** laisse le tirage décider. **Générer N** enchaîne. Le texte coloré suit la probabilité du token choisi. L'arbre **Branches** permet de revenir sur un nœud (le cache est recalculé). Les curseurs se règlent sans relancer le modèle : température 0,8, top-k 40, top-p 0,95, min-p 0,05, graine 42 au départ.

### Comparateur

But : la même consigne, deux modèles ou deux quantifications, l'un après l'autre pour ne pas additionner leur VRAM.

Exemple. Colonne A : le Q4_K_M. Colonne B : le Q8_0 du même modèle. Consigne « Explique le cycle de l'eau en quelques phrases. », 256 tokens, température 0, graine 42. **Lancer**. Les textes se remplissent tour à tour. Le tableau marque d'une étoile le meilleur débit de lecture, le meilleur débit de génération, le chargement le plus court et, si vous cochez **Mesurer la perplexité**, la perplexité la plus basse sur le texte de référence (un récit neutre déjà présent dans le champ, que vous pouvez remplacer). Un avertissement s'affiche si les deux côtés ont exactement les mêmes réglages.

### Moniteur

But : la carte pendant qu'un autre module travaille.

Le curseur **Historique (minutes)** va de 1 à 10. Les courbes couvrent l'utilisation GPU, la VRAM (et sa capacité), la température, la puissance, la fréquence SM, et le débit de génération publié par le laboratoire. Sans NVML, le panneau explique pourquoi (pilote absent, pas de carte NVIDIA) et les autres modules restent utilisables.

## Ligne de commande

`banc.exe` est à côté de `BancLocal.exe` dans le dossier de build (`build\cuda-release-local` ou `build\cpu-release`). Depuis un terminal :

```text
banc scan [dossiers…]          indexe les modèles GGUF
banc list                      liste la bibliothèque
banc info <modèle>             métadonnées GGUF
banc plan <modèle> --ctx 8192 --ctk q8_0 --ctv q8_0 --fa 1 [--swa-full 0] [--verify]
banc bench <modèle> --ngl 0,20,99 --ctx 4096 --reps 5
banc campaigns | history <modèle> | export <id> --md
banc gpu                       état de la carte (NVML)
```

`banc` et `banc <commande> --help` détaillent les options. `<modèle>` est un chemin de fichier ou une sous-chaîne unique du nom en base. `--db CHEMIN` change la base (défaut : `%LOCALAPPDATA%\BancLocal\banc.db`).

`banc plan --verify` charge réellement le modèle, mesure la VRAM prise et affiche l'écart avec l'estimation, ainsi que les allocations annoncées par llama.cpp. Les chiffres cités plus haut (RTX 4060 Ti, gemma-4-E4B Q4_K_M) viennent de cette vérification.

`banc bench` accepte des listes séparées par des virgules pour `--ngl`, `--ctx`, `--ctk`, `--fa`, `--pp`, `--tg` et `--reps`. La campagne est enregistrée et le tableau Markdown est imprimé. `banc export <id> --csv -o fichier.csv` ou `--md` ressort une campagne.

## Compilation avancée

Les presets versionnés sont dans `CMakePresets.json`. `install.cmd` ne les modifie pas.

| Preset | Usage |
|---|---|
| `cpu-debug` | développement, tests |
| `cpu-release` | Release CPU, celui de la CI |
| `cuda-release` | Release CUDA, architecture `sm_89` (RTX série 40) |
| `asan` | Debug CPU avec AddressSanitizer |

Quand une carte est détectée, `install.cmd` écrit `CMakeUserPresets.json` (ignoré par Git) : un preset `cuda-release-local` qui hérite de `cuda-release` et remplace `CMAKE_CUDA_ARCHITECTURES` (par exemple `86` pour une RTX série 30, `89` pour une RTX série 40, `120` pour une RTX série 50). Le binaire va dans `build\cuda-release-local`. Le preset `cuda-release` du dépôt reste sur `sm_89` et son dossier `build\cuda-release`.

```bat
scripts\dev.cmd cmake --preset cpu-debug
scripts\dev.cmd cmake --build build\cpu-debug
scripts\dev.cmd ctest --preset cpu-debug
```

`scripts\dev.cmd` ouvre l'environnement MSVC x64 (Visual Studio 2022 ou plus récent, repéré avec `vswhere`, sinon le chemin Visual Studio 2026 Community) et pointe `VCPKG_ROOT` vers le vcpkg de cette installation s'il n'est pas déjà défini.

Pour forcer une architecture sans passer par l'installeur : `install.cmd -Architecture 75`, ou `scripts\dev.cmd cmake --preset cuda-release -DCMAKE_CUDA_ARCHITECTURES=75`. Le toolkit doit connaître cette architecture (Ada `sm_89` : CUDA 11.8 ou plus ; Blackwell `sm_100` / `sm_120` : CUDA 12.8 ou plus).

Tests :

```bat
scripts\dev.cmd ctest --preset cpu-release
install.cmd -Tests
```

Les tests qui chargent un modèle utilisent `BANC_TEST_MODEL` (chemin d'un GGUF), sinon `tests\data\tiny-llama.gguf` (poids aléatoires, produit par `scripts\make_tiny_model.py`).

AddressSanitizer :

```bat
scripts\dev.cmd cmake --preset asan
scripts\dev.cmd cmake --build build\asan
scripts\dev.cmd ctest --preset asan
```

## Où sont les données

| Quoi | Où |
|---|---|
| Bibliothèque, campagnes, vérifications VRAM | `%LOCALAPPDATA%\BancLocal\banc.db` |
| Disposition de l'interface | `imgui.ini` à côté de `BancLocal.exe` |
| Polices facultatives | `assets\fonts\` à côté de l'exécutable (copiées au build) |
| Binaires | `build\cpu-release\`, `build\cuda-release\` ou `build\cuda-release-local\` |
| Dernier exécutable lancé par l'installeur | `build\banc-launch.txt` |
| Preset CUDA local | `CMakeUserPresets.json` à la racine (non versionné) |
| Journal d'installation | `build\install.log` |
| Exports de campagnes | le chemin choisi, par défaut `Documents\BancLocal` |

Effacer `banc.db` repart d'une bibliothèque vide ; le scan suivant la reconstruit. Les fichiers GGUF ne sont pas dans cette base, seulement leurs fiches.

## Dépannage

**Un prérequis manque.** `install.cmd` s'arrête avec le composant et le lien. Après une modification de Visual Studio, ouvrez un nouveau terminal : un ancien `PATH` ne voit pas CMake ni Ninja. Relancez `install.cmd`.

**`cl.exe`, CMake ou Ninja absents alors que Visual Studio est installé.** Modifiez l'installation et cochez les outils MSVC et « Outils CMake C++ pour Windows ».

**vcpkg introuvable.** Cochez « Gestionnaire de package vcpkg », ou clonez https://github.com/microsoft/vcpkg , lancez `bootstrap-vcpkg.bat`, puis `set VCPKG_ROOT=C:\vcpkg` avant `install.cmd`.

**Le sous-module est vide.** `install.cmd` lance `git submodule update --init --recursive`. Si le dossier n'est pas un clone (`git clone`), récupérez le dépôt ainsi plutôt qu'avec un zip.

**La compilation CUDA échoue sur l'architecture.** Le message de `nvcc` indique souvent un toolkit trop ancien pour la carte. Installez un toolkit adapté, ou imposez `-Architecture`. Sans GPU détecté, `-Cuda` retombe sur le preset `sm_89`.

**La compilation est tuée (mémoire).** Relancez `install.cmd` : Ninja reprend. Si ça se reproduit : `scripts\dev.cmd cmake --build build\cuda-release-local --parallel 2`.

**L'interface ne montre aucun modèle.** Regardez la liste du panneau Bibliothèque. Le dossier doit exister avant le scan. Les visionneurs `mmproj` et les shards autres que le premier sont ignorés. Si Ollama n'est pas dans le profil, la variable `OLLAMA_MODELS` doit pointer vers la bibliothèque. **Ajouter un dossier...** couvre un emplacement personnalisé de LM Studio.

**« Impossible d'initialiser DirectX 11 ».** Le pilote graphique ne fournit pas DirectX 11. Mettez-le à jour. Un repli WARP est tenté seulement dans certains cas d'absence de pilote matériel.

**Le moniteur dit NVML indisponible.** Pas de carte NVIDIA, ou pilote trop ancien. Le reste de l'atelier fonctionne. Un build CPU n'empêche pas NVML.

**Les accents sont illisibles dans `cmd.exe`.** La console est encore en page OEM. `install.cmd` passe en UTF-8. Dans un terminal déjà ouvert : `chcp 65001`. Windows Terminal le fait en général tout seul. Les sources et ce fichier sont en UTF-8.

**Le build CUDA ne semble pas utiliser la bonne carte.** Vérifiez le preset affiché par `install.cmd` et `CMakeUserPresets.json`. `cmake --preset cuda-release` reconstruit l'autre dossier, celui figé en `sm_89`.

**Deux `BancLocal.exe`.** `lancer.cmd` ouvre celui noté dans `build\banc-launch.txt`, sinon le plus récent parmi les dossiers connus.

## Architecture

```
external/llama.cpp/   sous-module épinglé (mis à jour volontairement, tests en garde-fou)
src/core/             aucun code d'interface : gguf_reader, model_roots, engine, vram_planner, bench, sampler_lab, comparator, telemetry
src/store/            SQLite
src/cli/              banc.exe
src/ui/               Dear ImGui (docking) + ImPlot + DirectX 11
tests/                Catch2
scripts/dev.cmd       environnement MSVC pour les commandes manuelles
install.cmd           installation et lancement
```

Règle centrale : **un seul thread parle à llama.cpp** (`Engine`). L'interface soumet des travaux et reçoit les résultats par `std::future`.

La CI (`.github/workflows/ci.yml`) compile et teste le preset `cpu-release` sous Windows. Elle n'utilise pas `install.cmd`.

## Polices

L'interface utilise Inter et les icônes Lucide si `Inter-Regular.ttf`, `Inter-SemiBold.ttf` et `lucide.ttf` sont dans `assets/fonts/` (voir le README de ce dossier). Sinon, la police intégrée d'ImGui. Aucun téléchargement n'est fait à l'installation.

- Inter : https://github.com/rsms/inter (SIL Open Font License 1.1)
- Lucide : https://github.com/lucide-icons/lucide (ISC, plage privée)

## Licence

MIT, comme llama.cpp. Le texte est dans `LICENSE` (copyright 2026 Anarchy). Les polices, si vous les redistribuez, gardent leur propre licence.
