cmake_minimum_required(VERSION 3.20)
project(gclab LANGUAGES CXX VERSION 0.1.0)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(GCLAB_ENABLE_CUDA "Build CUDA/NCCL components (needs nvcc; Linux only)" OFF)
option(GCLAB_BUILD_TESTS "Build unit tests" ON)
option(GCLAB_BUILD_BENCH "Build benchmarks" ON)
set(GCLAB_SANITIZER "" CACHE STRING "Sanitizer: '', 'address', or 'thread'")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

if(GCLAB_SANITIZER STREQUAL "address")
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
elseif(GCLAB_SANITIZER STREQUAL "thread")
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Core library: hardware-independent code (CPU_ONLY)
# ---------------------------------------------------------------------------
add_library(gclab_core
  src/common/log.cc
  src/sim/cost_model.cc
  src/sim/channel.cc
  src/sim/world.cc
  src/sim/inproc_transport.cc
  src/sim/collectives.cc
)
target_include_directories(gclab_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gclab_core PUBLIC Threads::Threads)

# ---------------------------------------------------------------------------
# CUDA components arrive in Module 3 (GPU_COMPILE). Kept behind a flag so the
# CPU build never depends on a CUDA toolkit.
# ---------------------------------------------------------------------------
if(GCLAB_ENABLE_CUDA)
  enable_language(CUDA)
  message(STATUS "gclab: CUDA enabled (no CUDA targets yet; added in Module 3)")
endif()

if(GCLAB_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(GCLAB_BUILD_BENCH)
  add_subdirectory(bench)
endif()
