cmake_minimum_required(VERSION 3.25)
project(banc_local VERSION 0.1.0 LANGUAGES C CXX)

option(BANC_CUDA        "Compiler ggml avec le backend CUDA" OFF)
option(BANC_BUILD_UI    "Construire l'application graphique (Windows)" ${WIN32})
option(BANC_BUILD_TESTS "Construire les tests Catch2" ON)
option(BANC_ASAN        "Activer AddressSanitizer" OFF)

if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
  if(BANC_ASAN)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/fsanitize=address> $<$<COMPILE_LANGUAGE:C,CXX>:/Zi>)
    add_compile_definitions(_DISABLE_VECTOR_ANNOTATION _DISABLE_STRING_ANNOTATION)
  endif()
endif()

# --- llama.cpp (sous-module épinglé) -------------------------------------
set(GGML_CUDA ${BANC_CUDA} CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
add_subdirectory(external/llama.cpp EXCLUDE_FROM_ALL)

execute_process(
  COMMAND git -C ${CMAKE_SOURCE_DIR}/external/llama.cpp describe --tags --always
  OUTPUT_VARIABLE BANC_LLAMA_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

# C++20 pour nos cibles seulement : llama.cpp compile en C++17 (littéraux u8).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(unofficial-sqlite3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

# --- Cœur : aucune dépendance d'interface ----------------------------------
add_library(banc_core STATIC
  src/core/gguf_reader.cpp
  src/core/vram_planner.cpp
  src/core/telemetry.cpp
  src/core/engine.cpp
  src/core/bench.cpp
  src/core/comparator.cpp
  src/core/sampler_lab.cpp
  src/store/store.cpp
)
target_include_directories(banc_core PUBLIC src)
target_link_libraries(banc_core PUBLIC llama ggml unofficial::sqlite3::sqlite3 spdlog::spdlog)
target_compile_definitions(banc_core PUBLIC
  BANC_VERSION="${PROJECT_VERSION}"
  BANC_LLAMA_VERSION="${BANC_LLAMA_VERSION}"
  $<$<BOOL:${BANC_CUDA}>:BANC_HAS_CUDA=1>)

# --- Outil en ligne de commande ---------------------------------------------
add_executable(banc src/cli/main.cpp)
target_link_libraries(banc PRIVATE banc_core)

# --- Application graphique --------------------------------------------------
if(BANC_BUILD_UI)
  find_package(imgui CONFIG REQUIRED)
  find_package(implot CONFIG REQUIRED)
  add_executable(banc_gui WIN32
    src/ui/main_win32.cpp
    src/ui/app.cpp
    src/ui/theme.cpp
    src/ui/widgets.cpp
    src/ui/views/library_view.cpp
    src/ui/views/vram_view.cpp
    src/ui/views/bench_view.cpp
    src/ui/views/sampler_view.cpp
    src/ui/views/compare_view.cpp
    src/ui/views/monitor_view.cpp
  )
  target_link_libraries(banc_gui PRIVATE banc_core imgui::imgui implot::implot d3d11 dxgi)
  set_target_properties(banc_gui PROPERTIES OUTPUT_NAME "BancLocal")
  add_custom_command(TARGET banc_gui POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:banc_gui>/assets)
endif()

# --- Tests ------------------------------------------------------------------
if(BANC_BUILD_TESTS)
  enable_testing()
  find_package(Catch2 3 CONFIG REQUIRED)
  add_executable(banc_tests
    tests/test_gguf_reader.cpp
    tests/test_vram_planner.cpp
    tests/test_engine.cpp
    tests/test_bench.cpp
    tests/test_sampler_lab.cpp
    tests/test_comparator.cpp
    tests/test_store.cpp
    tests/test_telemetry.cpp
  )
  target_link_libraries(banc_tests PRIVATE banc_core Catch2::Catch2WithMain)
  target_compile_definitions(banc_tests PRIVATE BANC_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  include(Catch)
  catch_discover_tests(banc_tests DISCOVERY_MODE PRE_TEST)
endif()
