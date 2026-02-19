cmake_minimum_required(VERSION 3.18)
project(pivision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- Build type: Release by default ----------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ---------- Interprocedural Optimization (IPO/LTO) ----------
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
if(IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "IPO/LTO: Enabled")
else()
    message(STATUS "IPO/LTO: Not supported (${IPO_ERROR})")
endif()

# ---------- Architecture-specific optimisation flags ----------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    # ARM64: Use generic ARMv8 flags (works on Pi 5, Orange Pi, etc.)
    # For Pi 5 specifically, you can use: -mcpu=cortex-a76
    add_compile_options(-march=armv8-a -O3)
    message(STATUS "Target: ARM64 with ARMv8-A optimizations")
else()
    # x86_64 or other: Just use -O3
    add_compile_options(-O3)
    message(STATUS "Target: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# ---------- llama.cpp location ----------
# Pass -DLLAMA_DIR=/path/to/llama.cpp at configure time.
if(NOT DEFINED LLAMA_DIR)
    message(FATAL_ERROR "Set -DLLAMA_DIR=<path-to-llama.cpp-root>")
endif()

set(LLAMA_INCLUDE_DIR  "${LLAMA_DIR}/include")
set(GGML_INCLUDE_DIR   "${LLAMA_DIR}/ggml/include")
set(LLAMA_MTMD_DIR     "${LLAMA_DIR}/tools/mtmd")
set(LLAMA_LIB_DIR      "${LLAMA_DIR}/build/bin")
set(LLAMA_COMMON_SRC   "${LLAMA_DIR}/common")
set(LLAMA_COMMON_DIR   "${LLAMA_DIR}/build/common")

# ---------- stb (header-only, used internally by the library) ----------
set(STB_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/stb")

# ---------- Import pre-built llama.cpp shared libraries ----------
add_library(llama     SHARED IMPORTED)
add_library(ggml      SHARED IMPORTED)
add_library(ggml_base SHARED IMPORTED)
add_library(mtmd      SHARED IMPORTED)

set_target_properties(llama     PROPERTIES IMPORTED_LOCATION "${LLAMA_LIB_DIR}/libllama.so")
set_target_properties(ggml      PROPERTIES IMPORTED_LOCATION "${LLAMA_LIB_DIR}/libggml.so")
set_target_properties(ggml_base PROPERTIES IMPORTED_LOCATION "${LLAMA_LIB_DIR}/libggml-base.so")
set_target_properties(mtmd      PROPERTIES IMPORTED_LOCATION "${LLAMA_LIB_DIR}/libmtmd.so")

# ---------- Import pre-built common static library ----------
add_library(llama_common STATIC IMPORTED)
set_target_properties(llama_common PROPERTIES
    IMPORTED_LOCATION "${LLAMA_COMMON_DIR}/libcommon.a")

# ---------- pivision static library ----------
add_library(pivision STATIC
    src/core.cpp
)

target_include_directories(pivision
    PUBLIC  ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_SOURCE_DIR}/src
    PRIVATE ${LLAMA_INCLUDE_DIR}
    PRIVATE ${GGML_INCLUDE_DIR}
    PRIVATE ${LLAMA_COMMON_SRC}
    PRIVATE ${LLAMA_MTMD_DIR}
    PRIVATE ${STB_INCLUDE_DIR}
)

target_link_libraries(pivision
    PRIVATE llama_common
    PRIVATE mtmd
    PRIVATE llama
    PRIVATE ggml
    PRIVATE ggml_base
)

# ---------- CLI executable ----------
add_executable(pivision_cli cmd/main.cpp)

# CLI only sees the public pivision header — never llama.cpp internals.
target_include_directories(pivision_cli PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(pivision_cli PRIVATE pivision)

# ---------- Strip binary in Release mode ----------
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_custom_command(TARGET pivision_cli POST_BUILD
        COMMAND ${CMAKE_STRIP} $<TARGET_FILE:pivision_cli>
        COMMENT "Stripping pivision_cli for production deployment"
    )
endif()

# ---------- Log-to-CSV scraper (standalone, no llama/pivision) ----------
add_executable(log_to_csv cmd/log_to_csv.cpp)
# Uses only C++17 stdlib and filesystem; no pivision/llama dependencies

# ---------- Installation ----------
install(TARGETS pivision_cli log_to_csv DESTINATION bin)
install(FILES include/pivision.h DESTINATION include)
