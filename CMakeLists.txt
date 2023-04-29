cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
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

set(CMAKE_OSX_DEPLOYMENT_TARGET "10.14" CACHE STRING "macOS minimum supported version.")
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "MSVC Runtime Library")

project(bladebit LANGUAGES C CXX ASM)

# Ensure supported OS and Architecture
if(NOT( (${CMAKE_SYSTEM_NAME} MATCHES "Linux") OR (${CMAKE_SYSTEM_NAME} MATCHES "Darwin") OR (${CMAKE_SYSTEM_NAME} MATCHES "Windows") ))
    message( FATAL_ERROR "Unsupported operating system '${CMAKE_SYSTEM_NAME}'" )
endif()

if(NOT(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "arm64" OR ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64"))
else()
    message( FATAL_ERROR "Unsupported architecture '${CMAKE_HOST_SYSTEM_PROCESSOR}'" )
endif()



# Is this project included as a dependency/FetchContent?
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(BB_IS_DEPENDENCY ON)
    set(BB_ENABLE_TESTS OFF)
    set(BB_ENABLE_EXE OFF)
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


#
# Grab Dependencies
#
set(platform_libs)

include(FetchContent)

# BLS
FetchContent_Declare(
  bls
  GIT_REPOSITORY https://github.com/Chia-Network/bls-signatures.git
  GIT_TAG        1.0.10
)

set(BUILD_BLS_PYTHON_BINDINGS "0" CACHE STRING "0")
set(BUILD_BLS_TESTS "0" CACHE STRING "")
set(BUILD_BLS_BENCHMARKS "0" CACHE STRING "")
FetchContent_MakeAvailable(bls)

# Threads
find_package(Threads REQUIRED)

# NUMA
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    find_package(NUMA REQUIRED)
    set(platform_libs ${NUMA_LIBRARY})
endif()

# Catch2
if(${BB_ENABLE_TESTS})
    include(cmake_modules/FindCatch2.cmake)
endif()

# Internal Config
set(is_release $<CONFIG:Release>)
set(is_debug $<CONFIG:Debug>)
set(is_c_cpp $<COMPILE_LANGUAGE:CXX,C>)
set(is_cuda $<COMPILE_LANGUAGE:CUDA>)
set(is_cuda_release $<AND:${is_cuda},${is_release}>)
set(is_cuda_debug $<AND:${is_cuda},${is_debug}>)
set(is_x86 $<OR:$<STREQUAL:${CMAKE_HOST_SYSTEM_PROCESSOR},AMD64>,$<STREQUAL:${CMAKE_HOST_SYSTEM_PROCESSOR},x86_64>>)
set(is_arm $<OR:$<STREQUAL:${CMAKE_HOST_SYSTEM_PROCESSOR},arm64>,$<STREQUAL:${CMAKE_HOST_SYSTEM_PROCESSOR},aarch64>>)

# Options
option(BENCHMARK_MODE "Enable benchmark mode for memplot. No final plot is written." OFF)
if(BENCHMARK_MODE)
    add_compile_definitions("BB_BENCHMARK_MODE=1")
endif()

option(ENABLE_DISK_METRICS "Enable I/O metrics for diskplot." OFF)
if(ENABLE_DISK_METRICS)
    add_compile_definitions("BB_IO_METRICS_ON=1")
endif()

# NOTE: These are mostly sandbox test environment, not proper tests
option(ENABLE_TESTS "Enable tests.." OFF)


# Targets

# macro(config_bladebit_proj tgt)
#     # message("Configuring target ${tgt}:${CMAKE_BUILD_TYPE}.")
#     target_link_libraries(${tgt} PUBLIC Threads::Threads bls ${platform_libs})
#     target_include_directories(${tgt} PUBLIC ${bb_include_dirs})

#     target_compile_definitions(${tgt} PUBLIC 
#         $<$<CONFIG:Release>:${c_defs} ${release_c_defs}>
#         $<$<CONFIG:Debug>:${c_defs} ${debug_c_defs}>
#     )

#     target_compile_options(${tgt} PRIVATE
#         ${c_opts}
#         $<$<CONFIG:Release>:${release_c_opts}>
#         $<$<CONFIG:Debug>:${debug_c_opts}>
#         $<${is_cuda}:
#             --pre-include pch.h
#         >
#         $<${is_c_cpp}:
#             $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:--include=pch.h>
#             $<$<CXX_COMPILER_ID:MSVC>:/FIpch.h>
#         >
#     )

#     target_link_options(${tgt} PRIVATE
#         $<$<CONFIG:Release>:${link_opts} ${release_link_opts}>
#         $<$<CONFIG:Debug>:${link_opts} ${debug_link_opts}>
#     )

#     set_target_properties(${tgt} PROPERTIES 
#         MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>
#         CUDA_RUNTIME_LIBRARY Static
#         CUDA_SEPARABLE_COMPILATION ON
#         CUDA_RESOLVE_DEVICE_SYMBOLS ON
#         CUDA_ARCHITECTURES OFF
#     )
#  endmacro()

include(Config.cmake)
include(Bladebit.cmake)

if(CUDAToolkit_FOUND)
    include(BladebitCUDA.cmake)
    include(Harvester.cmake)
endif()

if(ENABLE_TESTS)
    include(Tests.cmake)
endif()

# Pretty source view for IDE projects
# source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src 
#     FILES ${src_full} ${bb_headers}
# )