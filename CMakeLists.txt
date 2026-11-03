cmake_minimum_required(VERSION 3.20)
project(aegis_cfr VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(AEGIS_NATIVE "Tune for the host CPU" ON)
option(AEGIS_NO_VECTORIZE "Disable auto-vectorization (SIMD ablation build)" OFF)
option(AEGIS_BUILD_PYTHON "Build the pybind11 module" ON)

include(CheckCXXCompilerFlag)
add_compile_options(-Wall -Wextra -Wno-unused-parameter)
if(AEGIS_NATIVE)
  check_cxx_compiler_flag("-mcpu=native" HAS_MCPU_NATIVE)
  check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
  if(HAS_MCPU_NATIVE)
    add_compile_options(-mcpu=native)
  elseif(HAS_MARCH_NATIVE)
    add_compile_options(-march=native)
  endif()
endif()
if(AEGIS_NO_VECTORIZE)
  add_compile_options(-fno-vectorize -fno-slp-vectorize)
  add_compile_definitions(AEGIS_NO_VECTORIZE=1)
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
file(GLOB_RECURSE AEGIS_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_SOURCE_DIR}/cpp/util/*.cpp
  ${CMAKE_SOURCE_DIR}/cpp/core/*.cpp
  ${CMAKE_SOURCE_DIR}/cpp/games/*.cpp
  ${CMAKE_SOURCE_DIR}/cpp/poker/*.cpp
  ${CMAKE_SOURCE_DIR}/cpp/solvers/*.cpp
  ${CMAKE_SOURCE_DIR}/cpp/adaptive/*.cpp
  ${CMAKE_SOURCE_DIR}/cpp/value_net/*.cpp)

add_library(aegis STATIC ${AEGIS_SOURCES})
target_include_directories(aegis PUBLIC ${CMAKE_SOURCE_DIR}/cpp)
target_link_libraries(aegis PUBLIC Threads::Threads)
if(APPLE)
  # Accelerate provides BLAS (cblas_sgemm) for batched value-network inference.
  target_link_libraries(aegis PUBLIC "-framework Accelerate")
  target_compile_definitions(aegis PUBLIC AEGIS_HAVE_ACCELERATE=1 ACCELERATE_NEW_LAPACK=1)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
file(GLOB AEGIS_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/cpp/*.cpp)
add_executable(aegis_tests ${AEGIS_TEST_SOURCES})
target_link_libraries(aegis_tests PRIVATE aegis)
add_test(NAME aegis_unit COMMAND aegis_tests --exclude-slow)
add_test(NAME aegis_slow COMMAND aegis_tests --only-slow)

# ---------------------------------------------------------------------------
# Applications and benchmarks
# ---------------------------------------------------------------------------
add_executable(aegis_cli ${CMAKE_SOURCE_DIR}/cpp/apps/aegis_cli.cpp)
target_link_libraries(aegis_cli PRIVATE aegis)

add_executable(aegis_bench ${CMAKE_SOURCE_DIR}/benchmarks/aegis_bench.cpp)
target_link_libraries(aegis_bench PRIVATE aegis)

# ---------------------------------------------------------------------------
# Python module
# ---------------------------------------------------------------------------
if(AEGIS_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
                  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_package(pybind11 CONFIG REQUIRED HINTS ${PYBIND11_CMAKE_DIR})
  file(GLOB AEGIS_BINDING_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/cpp/bindings/*.cpp)
  pybind11_add_module(aegis_core ${AEGIS_BINDING_SOURCES})
  target_link_libraries(aegis_core PRIVATE aegis)
  set_target_properties(aegis_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/aegis)
endif()
