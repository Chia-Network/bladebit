cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CONFIGURATION_TYPES Release Debug)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release"
        CACHE STRING "Possible values are: Release, Debug"
        FORCE
    )
endif()

# Allows for CMAKE_MSVC_RUNTIME_LIBRARY
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW) 
endif()

set(CMAKE_OSX_DEPLOYMENT_TARGET "10.16" CACHE STRING "macOS minimum supported version.")
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "MSVC Runtime Library")

project(bladebit LANGUAGES C CXX ASM)

# Ensure supported OS and Architecture
if(NOT( (${CMAKE_SYSTEM_NAME} MATCHES "Linux") OR (${CMAKE_SYSTEM_NAME} MATCHES "Darwin") OR (${CMAKE_SYSTEM_NAME} MATCHES "Windows") ))
    message( FATAL_ERROR "Unsupported operating system '${CMAKE_SYSTEM_NAME}'" )
endif()

if(NOT (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "arm64" OR ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64" OR ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64" OR ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64"))
    message( FATAL_ERROR "Unsupported architecture '${CMAKE_HOST_SYSTEM_PROCESSOR}'" )
endif()

if(NOT CMAKE_CUDA_COMPILER)
    include(FindCUDAToolkit)

    if(CUDAToolkit_FOUND)
        message("Found CUDA: true")
        message("NVCC      : ${CUDAToolkit_NVCC_EXECUTABLE}")
        set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
    endif()
endif()

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()


message("Config   : ${CMAKE_BUILD_TYPE}")
message("Compiler : ${CMAKE_CXX_COMPILER_ID}")
if(DEFINED ENV{CI})
    message("CI build : true")
else()
    message("CI build : false")
endif()

set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules
)

# Is this project included as a dependency/FetchContent?
if(NOT(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR))
    set(BB_IS_DEPENDENCY ON)
    set(BB_ENABLE_TESTS OFF)
    set(BB_ENABLE_EXE OFF)
endif()


#
# Options
#
option(BENCHMARK_MODE "Enable benchmark mode for memplot. No final plot is written." OFF)
if(BENCHMARK_MODE)
    add_compile_definitions("BB_BENCHMARK_MODE=1")
endif()

option(ENABLE_DISK_METRICS "Enable I/O metrics for diskplot." OFF)
if(ENABLE_DISK_METRICS)
    add_compile_definitions("BB_IO_METRICS_ON=1")
endif()

# NOTE: These are mostly sandbox test environment, not proper tests
option(BB_ENABLE_TESTS "Enable tests." OFF)
option(NO_CUDA_HARVESTER "Explicitly disable CUDA in the bladebit_harvester target." OFF)
option(BB_NO_EMBED_VERSION "Disable embedding the version when building locally (non-CI)." OFF)
option(BB_HARVESTER_ONLY "Enable only the harvester target." OFF)
option(BB_HARVESTER_STATIC "Build the harvester target as a static library." OFF)
option(BB_CUDA_USE_NATIVE "Only build the native CUDA architecture when in release mode." OFF)

#
# Dependencies
#
include(FetchContent)

# Threads
find_package(Threads REQUIRED)

if(NOT ${BB_HARVESTER_ONLY})
    # BLS
    FetchContent_Declare(
    bls
    GIT_REPOSITORY https://github.com/Chia-Network/bls-signatures.git
    GIT_TAG        2.0.2
    EXCLUDE_FROM_ALL ${BB_IS_DEPENDENCY}
)

    set(BUILD_BLS_PYTHON_BINDINGS "0" CACHE STRING "0")
    set(BUILD_BLS_TESTS "0" CACHE STRING "")
    set(BUILD_BLS_BENCHMARKS "0" CACHE STRING "")
    FetchContent_MakeAvailable(bls)

    # NUMA
    if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        find_package(NUMA REQUIRED)
        set(platform_libs ${NUMA_LIBRARY})
    endif()
endif() # BB_HARVESTER_ONLY

#
# Internal Config
#
set(is_release $<CONFIG:Release>)
set(is_debug $<CONFIG:Debug>)
set(is_c_cpp $<COMPILE_LANGUAGE:CXX,C>)
set(is_cuda $<COMPILE_LANGUAGE:CUDA>)
set(is_cuda_release $<AND:${is_cuda},${is_release}>)
set(is_cuda_debug $<AND:${is_cuda},${is_debug}>)
set(is_x86 $<OR:$<STREQUAL:${CMAKE_HOST_SYSTEM_PROCESSOR},AMD64>,$<STREQUAL:${CMAKE_HOST_SYSTEM_PROCESSOR},x86_64>>)
set(is_arm $<OR:$<STREQUAL:${CMAKE_HOST_SYSTEM_PROCESSOR},arm64>,$<STREQUAL:${CMAKE_HOST_SYSTEM_PROCESSOR},aarch64>>)
set(is_msvc_c_cpp $<AND:${is_c_cpp},$<CXX_COMPILER_ID:MSVC>>)


if(CUDAToolkit_FOUND AND NOT ${NO_CUDA_HARVESTER})
    set(have_cuda $<BOOL:1>)
else()
    set(have_cuda $<BOOL:0>)
endif()


#
# Targets
#
include(Config.cmake)

if(NOT ${BB_HARVESTER_ONLY})
    if((NOT BB_IS_DEPENDENCY) AND (NOT BB_NO_EMBED_VERSION))
        include(cmake_modules/EmbedVersion.cmake)
    endif()

    include(Bladebit.cmake)
    set_target_properties(bladebit_core bladebit PROPERTIES EXCLUDE_FROM_ALL $<BOOL:${BB_IS_DEPENDENCY}>)

    if(CUDAToolkit_FOUND)
        include(BladebitCUDA.cmake)
        set_target_properties(bladebit_cuda PROPERTIES EXCLUDE_FROM_ALL $<BOOL:${BB_IS_DEPENDENCY}>)
    endif()
endif()

include(Harvester.cmake)

if(${BB_ENABLE_TESTS} AND NOT ${BB_HARVESTER_ONLY})
    include(Tests.cmake)
endif()

