cmake_minimum_required(VERSION 3.25)
project(waffle_iron VERSION 0.1.0 LANGUAGES C CXX)

option(WAFFLE_CUDA        "Build ggml with the CUDA backend" OFF)
option(WAFFLE_BUILD_UI    "Build the graphical application (Windows)" ${WIN32})
option(WAFFLE_BUILD_TESTS "Build the Catch2 tests" ON)
option(WAFFLE_ASAN        "Enable AddressSanitizer" OFF)

if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
  if(WAFFLE_ASAN)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/fsanitize=address> $<$<COMPILE_LANGUAGE:C,CXX>:/Zi>)
    add_compile_definitions(_DISABLE_VECTOR_ANNOTATION _DISABLE_STRING_ANNOTATION)
  endif()
endif()

# --- llama.cpp (pinned submodule) ----------------------------------------
set(GGML_CUDA ${WAFFLE_CUDA} CACHE BOOL "" FORCE)
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
  OUTPUT_VARIABLE WAFFLE_LLAMA_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

# C++20 for our targets only: llama.cpp builds as C++17 (u8 literals).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(unofficial-sqlite3 CONFIG REQUIRED)

# --- Core: no UI dependency ------------------------------------------------
add_library(waffle_core STATIC
  src/core/gguf_reader.cpp
  src/core/model_roots.cpp
  src/core/vram_planner.cpp
  src/core/telemetry.cpp
  src/core/engine.cpp
  src/core/bench.cpp
  src/core/comparator.cpp
  src/core/sampler_lab.cpp
  src/store/store.cpp
)
target_include_directories(waffle_core PUBLIC src)
target_link_libraries(waffle_core PUBLIC llama ggml unofficial::sqlite3::sqlite3)
target_compile_definitions(waffle_core PUBLIC
  WAFFLE_VERSION="${PROJECT_VERSION}"
  WAFFLE_LLAMA_VERSION="${WAFFLE_LLAMA_VERSION}"
  $<$<BOOL:${WAFFLE_CUDA}>:WAFFLE_HAS_CUDA=1>)

# --- Command-line tool ------------------------------------------------------
add_executable(waffle src/cli/main.cpp)
target_link_libraries(waffle PRIVATE waffle_core)

# --- Graphical application --------------------------------------------------
if(WAFFLE_BUILD_UI)
  find_package(imgui CONFIG REQUIRED)
  find_package(implot CONFIG REQUIRED)
  add_executable(waffle_gui WIN32
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
  target_link_libraries(waffle_gui PRIVATE waffle_core imgui::imgui implot::implot d3d11 dxgi)
  set_target_properties(waffle_gui PROPERTIES OUTPUT_NAME "WaffleIron")
  add_custom_command(TARGET waffle_gui POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:waffle_gui>/assets)
endif()

# --- Tests ------------------------------------------------------------------
if(WAFFLE_BUILD_TESTS)
  enable_testing()
  find_package(Catch2 3 CONFIG REQUIRED)
  add_executable(waffle_tests
    tests/test_gguf_reader.cpp
    tests/test_vram_planner.cpp
    tests/test_engine.cpp
    tests/test_bench.cpp
    tests/test_sampler_lab.cpp
    tests/test_comparator.cpp
    tests/test_store.cpp
    tests/test_telemetry.cpp
  )
  target_link_libraries(waffle_tests PRIVATE waffle_core Catch2::Catch2WithMain)
  target_compile_definitions(waffle_tests PRIVATE WAFFLE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  include(Catch)
  catch_discover_tests(waffle_tests DISCOVERY_MODE PRE_TEST)
endif()
