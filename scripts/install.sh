#!/usr/bin/env bash
# Install Waffle Iron on Linux or macOS and open the interface.
# Entry point for install.sh and launch.sh.
# --auto-test checks the CPU/CUDA/Metal selection logic without compiling anything.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

LAUNCH=0
NO_LAUNCH=0
FORCE_CPU=0
FORCE_CUDA=0
RUN_TESTS=0
SHOW_HELP=0
AUTO_TEST=0
REQUESTED_ARCH=

OS_NAME=Linux
HAS_METAL=0

PLAN_OK=1
PLAN_ERROR=
PLAN_KIND=cpu
PLAN_PRESET=cpu-release
PLAN_ARCH=
PLAN_NOTE=
PLAN_WRITE=0
PLAN_EXE="build/cpu-release/WaffleIron"
DETECTED_ARCHS=()
GPU_LABELS=()

usage() {
    cat <<'EOF'
Waffle Iron — install and launch (Linux and macOS)

  ./install.sh
      Checks prerequisites, fetches llama.cpp if needed, chooses CPU, CUDA, or Metal,
      builds, then opens the interface.

  ./install.sh --no-launch
      Builds without opening the interface.

  ./install.sh --cpu
      Forces a CPU build (preset cpu-release).

  ./install.sh --cuda
      Linux: forces a CUDA build. With no GPU identified, the architecture stays sm_89.
      macOS: rejected. Apple GPUs use Metal.

  ./install.sh --architecture 86
      Linux: sets the CUDA architecture (86, 89, 120...). 8.9 is accepted and becomes 89.
      Several values: --architecture 86,89

  ./install.sh --tests
      Runs the tests after the build.

  ./install.sh --help
      Shows this help.

  ./launch.sh
      Opens an already built WaffleIron, without rebuilding.

The first CUDA or Metal build is long. Running ./install.sh again resumes where it stopped.
Log: build/install.log

Windows uses install.cmd. The same flag names with a capital letter (-Cpu, -NoLaunch, …) are accepted.
EOF
}

die() {
    printf '\nERROR: %s\n' "$1" >&2
    exit 1
}

step() { printf '==> %s\n' "$1"; }
ok() { printf '    %s\n' "$1"; }
warn() { printf '    %s\n' "$1" >&2; }

json_escape() {
    local text=$1
    text=${text//\\/\\\\}
    text=${text//\"/\\\"}
    text=${text//$'\n'/ }
    text=${text//$'\r'/ }
    text=${text//$'\t'/ }
    printf '%s' "$text"
}

convert_compute_capability() {
    local text=$1
    text=${text#"${text%%[![:space:]]*}"}
    text=${text%"${text##*[![:space:]]}"}
    if [[ "$text" =~ ^([0-9]+)\.([0-9])$ ]]; then
        printf '%s%s' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
        return 0
    fi
    return 1
}

format_architectures() {
    if [[ $# -eq 0 ]]; then
        printf ''
        return 0
    fi
    printf '%s\n' "$@" | sed '/^$/d' | sort -n -u | paste -sd ';' -
}

parse_architecture_request() {
    PARSED_ARCH=
    PARSED_ERROR=
    local text=$1
    if [[ -z "${text// /}" ]]; then
        return 0
    fi
    local part arch
    local -a list=()
    local IFS=$';, \t'
    # shellcheck disable=SC2086
    for part in $text; do
        [[ -z "$part" ]] && continue
        if arch=$(convert_compute_capability "$part"); then
            list+=("$arch")
        elif [[ "$part" =~ ^[0-9]{2,3}$ ]]; then
            list+=("$part")
        else
            PARSED_ERROR="Invalid CUDA architecture: $part. Examples: 89 or 8.9."
            return 1
        fi
    done
    if [[ ${#list[@]} -eq 0 ]]; then
        PARSED_ERROR="Empty CUDA architecture."
        return 1
    fi
    PARSED_ARCH=$(format_architectures "${list[@]}")
}

cmake_version_of() {
    local text=$1
    if [[ "$text" =~ cmake\ version\ ([0-9]+)\.([0-9]+)\.([0-9]+) ]]; then
        printf '%s.%s.%s' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
        return 0
    fi
    return 1
}

nvcc_version_of() {
    local text=$1
    if [[ "$text" =~ release\ ([0-9]+)\.([0-9]+) ]]; then
        printf '%s.%s' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
        return 0
    fi
    return 1
}

version_lt() {
    [[ "$1" != "$2" && "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n 1)" == "$1" ]]
}

cuda_toolkit_warning() {
    local toolkit=$1 arch_text=$2
    [[ -z "$toolkit" || -z "$arch_text" ]] && return 0
    local part max=0
    local IFS=';'
    for part in $arch_text; do
        if [[ "$part" =~ ^[0-9]+$ && "$part" -gt "$max" ]]; then
            max=$part
        fi
    done
    if [[ "$max" -ge 100 ]] && version_lt "$toolkit" "12.8"; then
        printf 'sm_%s requires CUDA Toolkit 12.8 or newer (detected: %s).' "$max" "$toolkit"
    elif [[ "$max" -ge 89 ]] && version_lt "$toolkit" "11.8"; then
        printf 'sm_%s requires CUDA Toolkit 11.8 or newer (detected: %s).' "$max" "$toolkit"
    elif [[ "$max" -ge 80 ]] && version_lt "$toolkit" "11.1"; then
        printf 'sm_%s requires CUDA Toolkit 11.1 or newer (detected: %s).' "$max" "$toolkit"
    fi
}

set_plan() {
    PLAN_OK=$1
    PLAN_ERROR=$2
    PLAN_KIND=$3
    PLAN_PRESET=$4
    PLAN_ARCH=$5
    PLAN_NOTE=$6
    PLAN_WRITE=$7
    if [[ -n "$PLAN_PRESET" ]]; then
        PLAN_EXE="build/${PLAN_PRESET}/WaffleIron"
    else
        PLAN_EXE="build/cpu-release/WaffleIron"
    fi
}

select_plan() {
    local force_cpu=$1 force_cuda=$2 requested=$3 has_toolkit=$4
    shift 4
    local -a detected=("$@")
    local detected_text labels=""
    detected_text=$(format_architectures "${detected[@]+"${detected[@]}"}")
    if [[ ${#GPU_LABELS[@]} -gt 0 ]]; then
        local IFS=', '
        labels="${GPU_LABELS[*]}"
    fi

    if [[ "$OS_NAME" == "Darwin" ]]; then
        if [[ "$force_cpu" -eq 1 && "$force_cuda" -eq 1 ]]; then
            set_plan 0 "Choose only one mode: --cpu or --cuda." "" "" "" "" 0
            return 0
        fi
        if [[ "$force_cuda" -eq 1 || -n "$requested" ]]; then
            set_plan 0 "macOS builds use Metal, not CUDA. Run ./install.sh for a Metal build, or ./install.sh --cpu for the CPU." "" "" "" "" 0
            return 0
        fi
        if [[ "$force_cpu" -eq 1 ]]; then
            set_plan 1 "" cpu cpu-release "" "CPU build (--cpu)." 0
            return 0
        fi
        if [[ "$HAS_METAL" -eq 1 ]]; then
            set_plan 1 "" metal metal-release "" "Metal build for the Apple GPU." 0
            return 0
        fi
        set_plan 1 "" cpu cpu-release "" "Metal compiler (xcrun metal) was not found: CPU build. Install Xcode, then run ./install.sh again for a GPU build." 0
        return 0
    fi

    if [[ "$force_cpu" -eq 1 && "$force_cuda" -eq 1 ]]; then
        set_plan 0 "Choose only one mode: --cpu or --cuda." "" "" "" "" 0
        return 0
    fi
    if ! parse_architecture_request "$requested"; then
        set_plan 0 "$PARSED_ERROR" "" "" "" "" 0
        return 0
    fi
    if [[ "$force_cpu" -eq 1 && -n "$PARSED_ARCH" ]]; then
        set_plan 0 "--architecture cannot be used with --cpu." "" "" "" "" 0
        return 0
    fi
    if [[ ( "$force_cuda" -eq 1 || -n "$PARSED_ARCH" ) && "$has_toolkit" -eq 0 ]]; then
        local why="Install the CUDA Toolkit, then run ./install.sh again:
https://developer.nvidia.com/cuda-downloads"
        if [[ "$force_cuda" -eq 1 ]]; then
            why="The --cuda option requires the CUDA Toolkit (nvcc), which was not found.
$why"
        else
            why="A CUDA architecture was requested, but nvcc was not found.
$why"
        fi
        set_plan 0 "$why" "" "" "" "" 0
        return 0
    fi
    local arch_text=$PARSED_ARCH
    if [[ -z "$arch_text" ]]; then
        arch_text=$detected_text
    fi
    if [[ "$force_cpu" -eq 1 ]]; then
        set_plan 1 "" cpu cpu-release "" "CPU build (--cpu)." 0
        return 0
    fi
    if [[ "$has_toolkit" -eq 1 && ( "$force_cuda" -eq 1 || -n "$arch_text" ) ]]; then
        if [[ -n "$arch_text" ]]; then
            local pretty=${arch_text//;/, sm_}
            local note="CUDA build for sm_${pretty}."
            if [[ -n "$labels" ]]; then
                note+=" GPU: ${labels}."
            fi
            if [[ -n "$PARSED_ARCH" && -n "$detected_text" && "$PARSED_ARCH" != "$detected_text" ]]; then
                note+=" The --architecture option overrides detection (${detected_text})."
            fi
            set_plan 1 "" cuda cuda-release-local "$arch_text" "$note" 1
            return 0
        fi
        set_plan 1 "" cuda cuda-release "" "CUDA build using the cuda-release preset architecture (sm_89): the GPU was not identified. Use --architecture 86 to set another one." 0
        return 0
    fi
    if [[ -n "$labels" ]]; then
        set_plan 1 "" cpu cpu-release "" "NVIDIA GPU detected (${labels}), but the CUDA Toolkit is missing: CPU build. Install the CUDA Toolkit, then run ./install.sh again for a GPU build.
https://developer.nvidia.com/cuda-downloads" 0
        return 0
    fi
    if [[ "$has_toolkit" -eq 1 ]]; then
        set_plan 1 "" cpu cpu-release "" "CUDA Toolkit present, no NVIDIA GPU detected: CPU build. To force the cuda-release preset (sm_89), run again with --cuda." 0
        return 0
    fi
    set_plan 1 "" cpu cpu-release "" "No NVIDIA GPU and no CUDA Toolkit: CPU build." 0
}

write_user_preset() {
    local arch=$1 summary=$2
    local arch_json summary_json display
    arch_json=$(json_escape "$arch")
    summary_json=$(json_escape "$summary")
    display="CUDA, Release (install.sh: ${summary_json})"
    cat <<EOF
{
  "version": 6,
  "vendor": {
    "waffle.iron/install/1.0": { "generated": true }
  },
  "configurePresets": [
    {
      "name": "cuda-release-local",
      "displayName": "${display}",
      "inherits": "cuda-release",
      "cacheVariables": {
        "CMAKE_CUDA_ARCHITECTURES": "${arch_json}"
      }
    }
  ],
  "buildPresets": [
    { "name": "cuda-release-local", "configurePreset": "cuda-release-local" }
  ]
}
EOF
}

assert_eq() {
    local name=$1 actual=$2 expected=$3
    if [[ "$actual" != "$expected" ]]; then
        printf 'FAIL %s: got [%s] expected [%s]\n' "$name" "$actual" "$expected" >&2
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi
}

assert_like() {
    local name=$1 actual=$2 pattern=$3
    # shellcheck disable=SC2254
    if [[ "$actual" != $pattern ]]; then
        printf 'FAIL %s: [%s] does not match [%s]\n' "$name" "$actual" "$pattern" >&2
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi
}

run_self_test() {
    TEST_FAILURES=0
    local saved_os=$OS_NAME saved_metal=$HAS_METAL
    OS_NAME=Linux
    HAS_METAL=0
    GPU_LABELS=()

    assert_eq 'cap 8.9' "$(convert_compute_capability '8.9')" '89'
    assert_eq 'cap 12.0' "$(convert_compute_capability '12.0')" '120'
    assert_eq 'cap 10.0' "$(convert_compute_capability '10.0')" '100'
    assert_eq 'cap 7.5' "$(convert_compute_capability '7.5')" '75'
    if convert_compute_capability '8.90' >/dev/null 2>&1; then
        printf 'FAIL cap 8.90 should be rejected\n' >&2
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi

    GPU_LABELS=()
    select_plan 0 0 '' 0
    assert_eq 'nothing ok' "$PLAN_OK" 1
    assert_eq 'nothing kind' "$PLAN_KIND" cpu
    assert_eq 'nothing preset' "$PLAN_PRESET" cpu-release
    assert_like 'nothing note' "$PLAN_NOTE" '*No NVIDIA GPU*'

    GPU_LABELS=('Ada sample (sm_89)')
    select_plan 0 0 '' 0 89
    assert_eq 'gpu without toolkit preset' "$PLAN_PRESET" cpu-release
    assert_like 'gpu without toolkit note' "$PLAN_NOTE" '*CUDA Toolkit*'

    GPU_LABELS=('Ada sample (sm_89)')
    select_plan 0 0 '' 1 89
    assert_eq 'cuda preset' "$PLAN_PRESET" cuda-release-local
    assert_eq 'cuda arch' "$PLAN_ARCH" 89
    assert_eq 'cuda write' "$PLAN_WRITE" 1
    assert_eq 'cuda exe' "$PLAN_EXE" 'build/cuda-release-local/WaffleIron'
    assert_like 'cuda note' "$PLAN_NOTE" '*sm_89*'

    GPU_LABELS=('A' 'B')
    select_plan 0 0 '' 1 89 86
    assert_eq 'multi arch' "$PLAN_ARCH" '86;89'
    assert_like 'multi note' "$PLAN_NOTE" '*sm_86, sm_89*'

    GPU_LABELS=('RTX')
    select_plan 1 0 '' 1 89
    assert_eq 'force cpu preset' "$PLAN_PRESET" cpu-release
    assert_like 'force cpu note' "$PLAN_NOTE" '*--cpu*'

    GPU_LABELS=()
    select_plan 0 1 '' 0
    assert_eq 'force cuda without toolkit' "$PLAN_OK" 0
    assert_like 'force cuda message' "$PLAN_ERROR" '*nvcc*'

    select_plan 0 1 '' 1
    assert_eq 'force cuda blind preset' "$PLAN_PRESET" cuda-release
    assert_eq 'force cuda blind write' "$PLAN_WRITE" 0
    assert_like 'force cuda blind note' "$PLAN_NOTE" '*sm_89*'

    select_plan 0 0 '8.9' 1
    assert_eq 'arch 8.9' "$PLAN_ARCH" 89
    assert_eq 'arch 8.9 preset' "$PLAN_PRESET" cuda-release-local

    GPU_LABELS=('older')
    select_plan 0 0 '120,89' 1 86
    assert_eq 'arch list' "$PLAN_ARCH" '89;120'
    assert_like 'arch list note' "$PLAN_NOTE" '*overrides*'

    GPU_LABELS=()
    select_plan 0 0 '89,89' 1
    assert_eq 'duplicate arch' "$PLAN_ARCH" 89

    select_plan 0 0 'sm_89' 1
    assert_eq 'invalid arch' "$PLAN_OK" 0

    select_plan 1 1 '' 1
    assert_eq 'cpu and cuda' "$PLAN_OK" 0

    select_plan 1 0 '86' 1
    assert_eq 'arch and cpu' "$PLAN_OK" 0

    select_plan 0 0 '' 1
    assert_eq 'toolkit only preset' "$PLAN_PRESET" cpu-release
    assert_like 'toolkit only note' "$PLAN_NOTE" '*sm_89*'

    local warning
    warning=$(cuda_toolkit_warning '12.6' '120')
    assert_like 'warn 120' "$warning" '*12.8*'
    warning=$(cuda_toolkit_warning '12.6' '89')
    assert_eq 'warn 89 recent' "$warning" ''
    warning=$(cuda_toolkit_warning '11.7' '89')
    assert_like 'warn 89 old' "$warning" '*11.8*'
    warning=$(cuda_toolkit_warning '11.0' '86')
    assert_like 'warn 86' "$warning" '*11.1*'
    warning=$(cuda_toolkit_warning '11.0' '75')
    assert_eq 'warn 75' "$warning" ''

    local json_file preset_name preset_arch
    json_file=$(mktemp)
    write_user_preset '86;89' 'GPU "Ada"' >"$json_file"
    preset_name=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["configurePresets"][0]["name"])' "$json_file")
    preset_arch=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["configurePresets"][0]["cacheVariables"]["CMAKE_CUDA_ARCHITECTURES"])' "$json_file")
    assert_eq 'json name' "$preset_name" cuda-release-local
    assert_eq 'json arch' "$preset_arch" '86;89'
    if ! grep -q 'waffle.iron/install/1.0' "$json_file"; then
        printf 'FAIL json marker\n' >&2
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi
    rm -f "$json_file"

    OS_NAME=Darwin
    HAS_METAL=1
    GPU_LABELS=()
    select_plan 0 0 '' 0
    assert_eq 'metal preset' "$PLAN_PRESET" metal-release
    assert_eq 'metal kind' "$PLAN_KIND" metal
    select_plan 1 0 '' 0
    assert_eq 'darwin cpu' "$PLAN_PRESET" cpu-release
    select_plan 0 1 '' 1
    assert_eq 'darwin cuda rejected' "$PLAN_OK" 0
    assert_like 'darwin cuda message' "$PLAN_ERROR" '*Metal*'
    HAS_METAL=0
    select_plan 0 0 '' 0
    assert_eq 'darwin no metal' "$PLAN_PRESET" cpu-release
    assert_like 'darwin no metal note' "$PLAN_NOTE" '*xcrun metal*'

    OS_NAME=$saved_os
    HAS_METAL=$saved_metal
    if [[ "$TEST_FAILURES" -gt 0 ]]; then
        die "Self-test: ${TEST_FAILURES} failure(s)."
    fi
    printf 'Self-test passed.\n'
}

find_cuda_root() {
    if [[ -n "${CUDA_PATH:-}" && -x "${CUDA_PATH}/bin/nvcc" ]]; then
        printf '%s' "$CUDA_PATH"
        return 0
    fi
    local dir latest=""
    if [[ -x /usr/local/cuda/bin/nvcc ]]; then
        printf '%s' /usr/local/cuda
        return 0
    fi
    for dir in /usr/local/cuda-*; do
        [[ -x "$dir/bin/nvcc" ]] || continue
        latest=$dir
    done
    if [[ -n "$latest" ]]; then
        printf '%s' "$latest"
        return 0
    fi
    if [[ -x /usr/lib/nvidia-cuda-toolkit/bin/nvcc ]]; then
        printf '%s' /usr/lib/nvidia-cuda-toolkit
        return 0
    fi
    if command -v nvcc >/dev/null 2>&1; then
        local bin
        bin=$(dirname "$(command -v nvcc)")
        dirname "$bin"
        return 0
    fi
    return 1
}

collect_gpus() {
    DETECTED_ARCHS=()
    GPU_LABELS=()
    local smi=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        smi=$(command -v nvidia-smi)
    elif [[ -x /usr/bin/nvidia-smi ]]; then
        smi=/usr/bin/nvidia-smi
    fi
    [[ -z "$smi" ]] && return 0
    local line name cap mem arch
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*(.*)[[:space:]]*,[[:space:]]*([0-9]+\.[0-9])[[:space:]]*,[[:space:]]*(.*)$ ]] || continue
        name=${BASH_REMATCH[1]}
        cap=${BASH_REMATCH[2]}
        mem=${BASH_REMATCH[3]}
        mem=${mem%"${mem##*[![:space:]]}"}
        if arch=$(convert_compute_capability "$cap"); then
            DETECTED_ARCHS+=("$arch")
            GPU_LABELS+=("${name} (sm_${arch}, ${mem})")
            ok "GPU : ${name}  compute ${cap}  ${mem}"
        fi
    done < <("$smi" --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null || true)
}

launch_executable() {
    local marker="$ROOT/build/waffle-launch.txt"
    local relative full best="" best_time=0 time
    if [[ -f "$marker" ]]; then
        relative=$(head -n 1 "$marker" | tr -d '\r')
        relative=${relative//\\/\/}
        full="$ROOT/$relative"
        if [[ -f "$full" ]]; then
            printf '%s' "$full"
            return 0
        fi
    fi
    local candidate
    for candidate in \
        "$ROOT/build/cuda-release-local/WaffleIron" \
        "$ROOT/build/cuda-release/WaffleIron" \
        "$ROOT/build/metal-release/WaffleIron" \
        "$ROOT/build/cpu-release/WaffleIron"
    do
        [[ -f "$candidate" ]] || continue
        time=$(stat -c %Y "$candidate" 2>/dev/null || stat -f %m "$candidate")
        if [[ -z "$best" || "$time" -gt "$best_time" ]]; then
            best=$candidate
            best_time=$time
        fi
    done
    [[ -n "$best" ]] && printf '%s' "$best"
}

start_gui() {
    local exe=$1
    if [[ -z "$exe" || ! -f "$exe" ]]; then
        die "WaffleIron was not found. Run ./install.sh from the repository root first."
    fi
    local dir
    dir=$(dirname "$exe")
    (cd "$dir" && nohup "$exe" >/dev/null 2>&1 &)
    ok "Interface started: $exe"
}

vcpkg_baseline() {
    sed -n 's/^[[:space:]]*"builtin-baseline"[[:space:]]*:[[:space:]]*"\([0-9a-fA-F][0-9a-fA-F]*\)".*/\1/p' \
        "$ROOT/vcpkg.json" | head -n 1
}

ensure_vcpkg_baseline() {
    local root=$1
    local baseline
    baseline=$(vcpkg_baseline)
    [[ -n "$baseline" ]] || die "vcpkg.json has no builtin-baseline."
    if [[ ! -d "$root/.git" ]]; then
        warn "vcpkg at $root has no git metadata. A copied or shallow tree can miss baseline $baseline."
        return 0
    fi
    if git -C "$root" cat-file -e "${baseline}^{commit}" 2>/dev/null; then
        ok "vcpkg baseline $baseline"
        return 0
    fi
    step "Fetching vcpkg baseline $baseline"
    warn "A shallow vcpkg clone does not contain builtin-baseline. Fetching that commit."
    if ! git -C "$root" fetch --depth=1 origin "$baseline"; then
        git -C "$root" fetch origin "$baseline" || die "Could not fetch vcpkg baseline $baseline.
Clone vcpkg without --depth:
  git clone https://github.com/microsoft/vcpkg.git \"\$HOME/vcpkg\""
    fi
    git -C "$root" cat-file -e "${baseline}^{commit}" \
        || die "vcpkg baseline $baseline is still missing after fetch."
    ok "vcpkg baseline $baseline"
}

require_tools() {
    local -a problems=()
    if ! command -v git >/dev/null 2>&1; then
        problems+=("Git was not found.
It is used to fetch llama.cpp (the submodule) and by vcpkg.
Install Git, then run ./install.sh again.")
    fi
    local compiler=""
    if command -v c++ >/dev/null 2>&1; then
        compiler=$(command -v c++)
    elif command -v clang++ >/dev/null 2>&1; then
        compiler=$(command -v clang++)
    elif command -v g++ >/dev/null 2>&1; then
        compiler=$(command -v g++)
    fi
    if [[ -z "$compiler" ]]; then
        if [[ "$OS_NAME" == "Darwin" ]]; then
            problems+=("A C++ compiler was not found.
Install Xcode or the Command Line Tools: xcode-select --install")
        else
            problems+=("A C++20 compiler was not found.
Install GCC 11+ or Clang 14+ (Debian/Ubuntu: build-essential).")
        fi
    else
        ok "C++ compiler : $compiler"
    fi
    if ! command -v cmake >/dev/null 2>&1; then
        problems+=("CMake 3.25 or newer was not found.
https://cmake.org/download/")
    else
        local cmake_text cmake_ver
        cmake_text=$(cmake --version | head -n 1)
        if ! cmake_ver=$(cmake_version_of "$cmake_text"); then
            problems+=("CMake did not report a version. Version 3.25 or newer is required.")
        elif version_lt "$cmake_ver" "3.25.0"; then
            problems+=("CMake ${cmake_ver} is too old. Version 3.25 or newer is required (presets).
https://cmake.org/download/")
        else
            ok "CMake $cmake_ver"
        fi
    fi
    if ! command -v ninja >/dev/null 2>&1; then
        problems+=("Ninja was not found.
https://ninja-build.org/  (Debian/Ubuntu: ninja-build, macOS: brew install ninja)")
    else
        ok "Ninja : $(command -v ninja)"
    fi
    local vcpkg_root=""
    if [[ -n "${VCPKG_ROOT:-}" && -x "${VCPKG_ROOT}/vcpkg" ]]; then
        vcpkg_root=$VCPKG_ROOT
    elif [[ -x "$HOME/vcpkg/vcpkg" ]]; then
        vcpkg_root=$HOME/vcpkg
    elif [[ -x /opt/vcpkg/vcpkg ]]; then
        vcpkg_root=/opt/vcpkg
    elif [[ -x /usr/local/vcpkg/vcpkg ]]; then
        vcpkg_root=/usr/local/vcpkg
    elif command -v vcpkg >/dev/null 2>&1; then
        vcpkg_root=$(dirname "$(command -v vcpkg)")
    fi
    if [[ -z "$vcpkg_root" || ! -f "$vcpkg_root/scripts/buildsystems/vcpkg.cmake" ]]; then
        if [[ -e "$HOME/vcpkg" ]]; then
            problems+=("vcpkg was not found, and $HOME/vcpkg is not a usable checkout.
Clone it without --depth (a shallow clone omits the builtin-baseline commit):
  git clone https://github.com/microsoft/vcpkg.git \"\$HOME/vcpkg\"
  \"\$HOME/vcpkg/bootstrap-vcpkg.sh\" -disableMetrics
  export VCPKG_ROOT=\"\$HOME/vcpkg\"")
        elif ! command -v git >/dev/null 2>&1; then
            problems+=("vcpkg was not found, and Git is required to clone it.")
        else
            step "Cloning vcpkg into $HOME/vcpkg (full history, not a shallow clone)"
            if git clone --filter=blob:none https://github.com/microsoft/vcpkg.git "$HOME/vcpkg"; then
                vcpkg_root=$HOME/vcpkg
            else
                problems+=("Could not clone vcpkg. Clone it yourself without --depth:
  git clone https://github.com/microsoft/vcpkg.git \"\$HOME/vcpkg\"")
            fi
        fi
    fi
    if [[ -n "$vcpkg_root" && -f "$vcpkg_root/scripts/buildsystems/vcpkg.cmake" ]]; then
        export VCPKG_ROOT=$vcpkg_root
        ok "vcpkg : $vcpkg_root"
    fi
    if [[ "$OS_NAME" == "Linux" && ! -f /usr/include/GL/gl.h && ! -f /usr/include/GL/glcorearb.h ]]; then
        problems+=("OpenGL headers were not found. The interface links OpenGL.
Debian/Ubuntu: sudo apt-get install libgl1-mesa-dev
Fedora: sudo dnf install mesa-libGL-devel
The command-line tool can be built without the interface: cmake --preset cpu-release -DWAFFLE_BUILD_UI=OFF")
    fi
    if [[ "$OS_NAME" == "Linux" ]]; then
        local missing_auto=()
        local tool
        for tool in autoconf automake libtoolize pkg-config; do
            if ! command -v "$tool" >/dev/null 2>&1; then
                missing_auto+=("$tool")
            fi
        done
        if [[ ${#missing_auto[@]} -gt 0 ]]; then
            problems+=("Autotools needed to build GLFW's X11 dependencies are missing: ${missing_auto[*]}.
Debian/Ubuntu: sudo apt-get install autoconf autoconf-archive automake libtool pkg-config xutils-dev
Fedora: sudo dnf install autoconf autoconf-archive automake libtool pkgconf xorg-x11-util-macros
Arch: sudo pacman -S base-devel pkgconf")
        fi
        if ! command -v zip >/dev/null 2>&1; then
            problems+=("zip was not found. vcpkg's bootstrap script uses it.
Debian/Ubuntu: sudo apt-get install zip
Fedora: sudo dnf install zip
Arch: sudo pacman -S zip")
        fi
        if command -v pkg-config >/dev/null 2>&1; then
            local missing_x=()
            local module
            for module in xrandr xinerama xcursor xi xxf86vm glu; do
                if ! pkg-config --exists "$module"; then
                    missing_x+=("$module")
                fi
            done
            if [[ ${#missing_x[@]} -gt 0 ]]; then
                problems+=("GLFW's X11 dependencies are missing (${missing_x[*]}).
Debian/Ubuntu: sudo apt-get install libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libxxf86vm-dev libglu1-mesa-dev pkg-config
Fedora: sudo dnf install libXrandr-devel libXinerama-devel libXcursor-devel libXi-devel libXxf86vm-devel mesa-libGLU-devel pkgconf
Arch: sudo pacman -S libxrandr libxinerama libxcursor libxi libxxf86vm glu pkgconf")
            fi
        fi
    fi
    if [[ ${#problems[@]} -gt 0 ]]; then
        printf '\nInstallation cannot continue: prerequisites are missing.\n\n' >&2
        local problem
        for problem in "${problems[@]}"; do
            printf '%s\n\n' "$problem" >&2
        done
        die "Prerequisites are incomplete."
    fi
    if [[ -n "${VCPKG_ROOT:-}" && -x "$VCPKG_ROOT/vcpkg" ]]; then
        :
    elif [[ -n "${VCPKG_ROOT:-}" && -f "$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" ]]; then
        step "Bootstrapping vcpkg"
        "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
    fi
    if [[ -n "${VCPKG_ROOT:-}" ]]; then
        ensure_vcpkg_baseline "$VCPKG_ROOT"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help | -h | -Help) SHOW_HELP=1 ;;
        --launch | -Launch) LAUNCH=1 ;;
        --no-launch | -NoLaunch) NO_LAUNCH=1 ;;
        --cpu | -Cpu) FORCE_CPU=1 ;;
        --cuda | -Cuda) FORCE_CUDA=1 ;;
        --tests | -Tests) RUN_TESTS=1 ;;
        --auto-test | -AutoTest) AUTO_TEST=1 ;;
        --architecture | -Architecture)
            shift
            REQUESTED_ARCH=${1:-}
            [[ -n "$REQUESTED_ARCH" ]] || die "--architecture needs a value, for example 89 or 8.9."
            ;;
        *)
            die "Unknown option: $1 (try ./install.sh --help)"
            ;;
    esac
    shift
done

case "$(uname -s)" in
    Linux) OS_NAME=Linux ;;
    Darwin)
        OS_NAME=Darwin
        if xcrun --find metal >/dev/null 2>&1; then
            HAS_METAL=1
        fi
        ;;
    MINGW* | MSYS* | CYGWIN*)
        die "This script is for Linux and macOS. On Windows, run install.cmd."
        ;;
    *)
        OS_NAME=Linux
        warn "Unsupported kernel $(uname -s): using the Linux CPU/CUDA path."
        ;;
esac

if [[ "$AUTO_TEST" -eq 1 ]]; then
    run_self_test
    exit 0
fi
if [[ "$SHOW_HELP" -eq 1 ]]; then
    usage
    exit 0
fi

mkdir -p "$ROOT/build"
LOG="$ROOT/build/install.log"
if [[ "$LAUNCH" -eq 0 ]]; then
    : >"$LOG"
fi
log() {
    if [[ "$LAUNCH" -eq 0 ]]; then
        printf '%s\n' "$*" | tee -a "$LOG"
    else
        printf '%s\n' "$*"
    fi
}

step "Waffle Iron"
log "    Folder: $ROOT"

if [[ "$LAUNCH" -eq 1 ]]; then
    start_gui "$(launch_executable)"
    exit 0
fi

require_tools

if [[ ! -f "$ROOT/external/llama.cpp/CMakeLists.txt" ]]; then
    if [[ ! -d "$ROOT/.git" ]]; then
        die "This folder is not a Git clone. Use git clone --recurse-submodules, not a zip archive: the external/llama.cpp submodule would be missing."
    fi
    step "Fetching the llama.cpp submodule (internet access)"
    GIT_TERMINAL_PROMPT=0 git -C "$ROOT" submodule update --init --recursive
    [[ -f "$ROOT/external/llama.cpp/CMakeLists.txt" ]] || die "The external/llama.cpp submodule is still missing after git submodule update."
else
    ok "llama.cpp submodule is present."
fi

CUDA_ROOT=
if [[ "$OS_NAME" != "Darwin" ]]; then
    if CUDA_ROOT=$(find_cuda_root); then
        export CUDA_PATH=$CUDA_ROOT
        export PATH="${CUDA_ROOT}/bin:${PATH}"
        ok "CUDA Toolkit : $CUDA_ROOT"
    else
        CUDA_ROOT=
    fi
    collect_gpus
    if [[ ${#DETECTED_ARCHS[@]} -eq 0 ]]; then
        warn "No NVIDIA GPU detected (nvidia-smi)."
    fi
else
    if [[ "$HAS_METAL" -eq 1 ]]; then
        ok "Metal compiler : $(xcrun --find metal)"
    else
        warn "xcrun metal was not found. A CPU build will be used unless Xcode is installed."
    fi
fi

HAS_TOOLKIT=0
[[ -n "$CUDA_ROOT" ]] && HAS_TOOLKIT=1
select_plan "$FORCE_CPU" "$FORCE_CUDA" "$REQUESTED_ARCH" "$HAS_TOOLKIT" ${DETECTED_ARCHS[@]+"${DETECTED_ARCHS[@]}"}
[[ "$PLAN_OK" -eq 1 ]] || die "$PLAN_ERROR"
step "Selected configuration"
log "    $PLAN_NOTE"

if [[ "$PLAN_KIND" == "cuda" ]]; then
    nvcc_text=$(nvcc --version 2>/dev/null || true)
    nvcc_ver=""
    nvcc_ver=$(nvcc_version_of "$nvcc_text" || true)
    arch_for_warning=$PLAN_ARCH
    [[ -z "$arch_for_warning" ]] && arch_for_warning=89
    warning=$(cuda_toolkit_warning "$nvcc_ver" "$arch_for_warning" || true)
    [[ -n "$warning" ]] && warn "$warning"
fi

configure_preset=$PLAN_PRESET
cmake_extra=()
if [[ "$PLAN_WRITE" -eq 1 ]]; then
    user_preset="$ROOT/CMakeUserPresets.json"
    foreign=0
    if [[ -f "$user_preset" ]] && ! grep -q 'waffle.iron/install/1.0' "$user_preset"; then
        foreign=1
    fi
    if [[ "$foreign" -eq 1 ]]; then
        warn "CMakeUserPresets.json exists and was not created by install.sh: it is left unchanged."
        arch_file="$ROOT/build/waffle-cuda-arch.cmake"
        printf 'set(CMAKE_CUDA_ARCHITECTURES "%s" CACHE STRING "Chosen by install.sh" FORCE)\n' "$PLAN_ARCH" >"$arch_file"
        include=$(cd "$(dirname "$arch_file")" && pwd)/$(basename "$arch_file")
        if [[ "$include" == *";"* ]]; then
            die "The repository path contains a semicolon, which cannot be used in a CMake list: $include"
        fi
        configure_preset=cuda-release
        PLAN_EXE="build/cuda-release/WaffleIron"
        cmake_extra+=("-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=${include}")
    else
        summary=$(printf '%s' "$PLAN_NOTE" | tr '\n' ' ' | tr -s ' ')
        if [[ ${#summary} -gt 140 ]]; then
            summary=${summary:0:140}
        fi
        write_user_preset "$PLAN_ARCH" "$summary" >"$user_preset"
        pretty=${PLAN_ARCH//;/, sm_}
        ok "Local preset: CMakeUserPresets.json (sm_${pretty})."
    fi
fi

log "    Preset : $configure_preset"
log "    Executable: $PLAN_EXE"
step "CMake configure (the first run downloads vcpkg dependencies; internet required)"
cmake --preset "$configure_preset" "${cmake_extra[@]+"${cmake_extra[@]}"}" 2>&1 | tee -a "$LOG"
if [[ "$PLAN_KIND" == "cuda" || "$PLAN_KIND" == "metal" ]]; then
    step "GPU build. It can take a long time. If it stops, run ./install.sh again: it resumes."
else
    step "Build"
fi
cmake --build --preset "$configure_preset" 2>&1 | tee -a "$LOG"

exe="$ROOT/$PLAN_EXE"
[[ -f "$exe" ]] || die "The build finished without producing $exe"
printf '%s\n' "$PLAN_EXE" >"$ROOT/build/waffle-launch.txt"

if [[ "$RUN_TESTS" -eq 1 ]]; then
    step "Tests"
    ctest --test-dir "$(dirname "$exe")" --output-on-failure 2>&1 | tee -a "$LOG"
fi

if [[ "$NO_LAUNCH" -eq 1 ]]; then
    ok "Build finished. Open the interface with ./launch.sh."
else
    step "Opening the interface"
    start_gui "$exe"
    log "    On first display, the library scans your model folders."
    log "    To open it again later: ./launch.sh"
fi
ok "Log: build/install.log"
