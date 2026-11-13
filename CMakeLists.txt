cmake_minimum_required(VERSION 3.25)
project(waffle_iron VERSION 0.1.0 LANGUAGES C CXX)

option(WAFFLE_CUDA        "Build ggml with the CUDA backend" OFF)
# Metal is the macOS GPU backend. Presets that want a CPU build set this OFF,
# which also overrides llama.cpp's Apple default of ON.
if(APPLE)
  option(WAFFLE_METAL "Build ggml with the Metal backend" ON)
else()
  option(WAFFLE_METAL "Build ggml with the Metal backend" OFF)
endif()
if(WIN32 OR UNIX)
  set(_WAFFLE_UI_DEFAULT ON)
else()
  set(_WAFFLE_UI_DEFAULT OFF)
endif()
option(WAFFLE_BUILD_UI    "Build the graphical application" ${_WAFFLE_UI_DEFAULT})
option(WAFFLE_BUILD_TESTS "Build the Catch2 tests" ON)
option(WAFFLE_ASAN        "Enable AddressSanitizer" OFF)

if(WAFFLE_CUDA AND WAFFLE_METAL)
  message(FATAL_ERROR "WAFFLE_CUDA and WAFFLE_METAL cannot both be ON")
endif()
if(WAFFLE_METAL AND NOT APPLE)
  message(FATAL_ERROR "WAFFLE_METAL requires macOS")
endif()

if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
  if(WAFFLE_ASAN)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/fsanitize=address> $<$<COMPILE_LANGUAGE:C,CXX>:/Zi>)
    add_compile_definitions(_DISABLE_VECTOR_ANNOTATION _DISABLE_STRING_ANNOTATION)
  endif()
elseif(WAFFLE_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address)
  # A manifest install is often release-only. Debug then looks for missing
  # debug/lib archives. Prefer the release vcpkg libraries; our objects stay
  # instrumented. The empty entry still accepts config-less imported targets
  # such as OpenGL::GL from FindOpenGL.
  set(CMAKE_MAP_IMPORTED_CONFIG_DEBUG "Release;" CACHE STRING
      "Imported configs tried by the AddressSanitizer Debug build" FORCE)
endif()

# --- llama.cpp (pinned submodule) ----------------------------------------
set(GGML_CUDA ${WAFFLE_CUDA} CACHE BOOL "" FORCE)
set(GGML_METAL ${WAFFLE_METAL} CACHE BOOL "" FORCE)
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
  src/core/util.cpp
  src/core/json.cpp
  src/core/text_diff.cpp
  src/core/gguf_reader.cpp
  src/core/model_roots.cpp
  src/core/vram_planner.cpp
  src/core/vram_verify.cpp
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
  $<$<BOOL:${WAFFLE_CUDA}>:WAFFLE_HAS_CUDA=1>
  $<$<BOOL:${WAFFLE_METAL}>:WAFFLE_HAS_METAL=1>)
if(NOT WIN32)
  find_package(Threads REQUIRED)
  target_link_libraries(waffle_core PUBLIC Threads::Threads)
  target_link_libraries(waffle_core PRIVATE ${CMAKE_DL_LIBS})
endif()
if(APPLE)
  target_sources(waffle_core PRIVATE src/core/telemetry_metal.mm)
  target_link_libraries(waffle_core PRIVATE "-framework Metal" "-framework Foundation")
  set_source_files_properties(src/core/telemetry_metal.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
endif()

# --- Command-line tool ------------------------------------------------------
add_library(waffle_cli STATIC src/cli/cli.cpp)
target_link_libraries(waffle_cli PUBLIC waffle_core)
target_include_directories(waffle_cli PUBLIC src)
add_executable(waffle src/cli/main.cpp)
target_link_libraries(waffle PRIVATE waffle_cli)
include(GNUInstallDirs)
install(TARGETS waffle RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# --- Graphical application --------------------------------------------------
if(WAFFLE_BUILD_UI)
  find_package(imgui CONFIG REQUIRED)
  find_package(implot CONFIG REQUIRED)
  set(WAFFLE_UI_SOURCES
    src/ui/app.cpp
    src/ui/theme.cpp
    src/ui/widgets.cpp
    src/ui/platform.cpp
    src/ui/views/library_view.cpp
    src/ui/views/vram_view.cpp
    src/ui/views/bench_view.cpp
    src/ui/views/sampler_view.cpp
    src/ui/views/compare_view.cpp
    src/ui/views/monitor_view.cpp
  )
  if(WIN32)
    list(APPEND WAFFLE_UI_SOURCES src/ui/main_win32.cpp)
  else()
    list(APPEND WAFFLE_UI_SOURCES src/ui/main_glfw.cpp)
  endif()
  add_executable(waffle_gui WIN32 ${WAFFLE_UI_SOURCES})
  target_link_libraries(waffle_gui PRIVATE waffle_core imgui::imgui implot::implot)
  if(WIN32)
    target_link_libraries(waffle_gui PRIVATE d3d11 dxgi ole32 shell32)
  else()
    find_package(glfw3 CONFIG REQUIRED)
    find_package(OpenGL REQUIRED)
    if(TARGET glfw)
      set(_WAFFLE_GLFW_TARGET glfw)
    else()
      set(_WAFFLE_GLFW_TARGET glfw3)
    endif()
    target_link_libraries(waffle_gui PRIVATE ${_WAFFLE_GLFW_TARGET} OpenGL::GL)
    if(APPLE)
      target_compile_definitions(waffle_gui PRIVATE GL_SILENCE_DEPRECATION)
    endif()
  endif()
  set_target_properties(waffle_gui PROPERTIES OUTPUT_NAME "WaffleIron")
  # Every preset that builds the interface copies assets (including lucide.ttf) and NOTICE
  # next to the binary, including when only the font changed and the executable is up to date.
  add_custom_target(waffle_assets ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/assets" "$<TARGET_FILE_DIR:waffle_gui>/assets"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/NOTICE" "$<TARGET_FILE_DIR:waffle_gui>/NOTICE"
    VERBATIM
    COMMENT "Copy assets and NOTICE next to WaffleIron")
  add_dependencies(waffle_assets waffle_gui)
  install(TARGETS waffle_gui RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/assets" DESTINATION "${CMAKE_INSTALL_BINDIR}")
  install(FILES
    "${CMAKE_SOURCE_DIR}/NOTICE"
    "${CMAKE_SOURCE_DIR}/LICENSE"
    "${CMAKE_SOURCE_DIR}/THIRD_PARTY.md"
    DESTINATION "${CMAKE_INSTALL_BINDIR}")
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
    tests/test_cli.cpp
    tests/test_json.cpp
  )
  target_link_libraries(waffle_tests PRIVATE waffle_core waffle_cli Catch2::Catch2WithMain)
  target_compile_definitions(waffle_tests PRIVATE WAFFLE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  include(Catch)
  catch_discover_tests(waffle_tests DISCOVERY_MODE PRE_TEST)
endif()

function(waffle_warnings target)
  if(TARGET ${target})
    if(MSVC)
      target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
      target_compile_options(${target} PRIVATE -Wall -Wextra)
    endif()
  endif()
endfunction()
waffle_warnings(waffle_core)
waffle_warnings(waffle_cli)
waffle_warnings(waffle)
if(TARGET waffle_gui)
  waffle_warnings(waffle_gui)
endif()
if(TARGET waffle_tests)
  waffle_warnings(waffle_tests)
endif()
