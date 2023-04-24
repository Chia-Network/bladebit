cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

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

# BLS
include(FetchContent)

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



# Catch
# TODO: Add configuration var to disable this
include(cmake_modules/FindCatch2.cmake)
# set_target_properties(Catch2 PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
# set_target_properties(Catch2WithMain PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")


# Config
set(c_opts)
set(link_opts)

set(release_c_opts)
set(debug_c_opts)
set(dev_c_opts)

set(c_defs)
set(release_c_defs
    NDEBUG=1
    _NDEBUG=1
)
set(debug_c_defs
    DEBUG=1
    _DEBUG=1
)

set(release_link_opts)
set(debug_link_opts)

option(BENCHMARK_MODE "Enable benchmark mode for memplot. No final plot is written." OFF)
if(BENCHMARK_MODE)
    add_compile_definitions("BB_BENCHMARK_MODE=1")
endif()

option(ENABLE_DISK_METRICS "Enable I/O metrics for diskplot." OFF)
if(ENABLE_DISK_METRICS)
    add_compile_definitions("BB_IO_METRICS_ON=1")
endif()

# Embed version inline when in dev mode
if((NOT DEFINED ENV{CI}) AND (NOT DEFINED CACHE{bb_version_embedded}))
    message("Embedding local build version")

    set(bb_version_embedded on CACHE BOOL "Version embedding has already happened.")

    set(cmd_ver bash)
    if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        set(cmd_ver bash.exe)
    endif()

    execute_process(COMMAND ${cmd_ver} extract-version.sh major    OUTPUT_VARIABLE bb_ver_maj    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND_ERROR_IS_FATAL ANY)
    execute_process(COMMAND ${cmd_ver} extract-version.sh minor    OUTPUT_VARIABLE bb_ver_min    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND_ERROR_IS_FATAL ANY)
    execute_process(COMMAND ${cmd_ver} extract-version.sh revision OUTPUT_VARIABLE bb_ver_rev    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND_ERROR_IS_FATAL ANY)
    execute_process(COMMAND ${cmd_ver} extract-version.sh suffix   OUTPUT_VARIABLE bb_ver_suffix WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND_ERROR_IS_FATAL ANY)
    execute_process(COMMAND ${cmd_ver} extract-version.sh commit   OUTPUT_VARIABLE bb_ver_commit WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND_ERROR_IS_FATAL ANY)

    # Remove trailing whitespace incurred in windows gitbash
    string(STRIP "${bb_ver_maj}"    bb_ver_maj)
    string(STRIP "${bb_ver_min}"    bb_ver_min)
    string(STRIP "${bb_ver_rev}"    bb_ver_rev)
    string(STRIP "${bb_ver_suffix}" bb_ver_suffix)
    string(STRIP "${bb_ver_commit}" bb_ver_commit)

    set(bb_ver_suffix ${bb_ver_suffix}-dev)

    # This is slow on windows, so let's cache them
    set(bb_ver_maj    ${bb_ver_maj}    CACHE STRING "")
    set(bb_ver_min    ${bb_ver_min}    CACHE STRING "")
    set(bb_ver_rev    ${bb_ver_rev}    CACHE STRING "")
    set(bb_ver_suffix ${bb_ver_suffix} CACHE STRING "")
    set(bb_ver_commit ${bb_ver_commit} CACHE STRING "")
endif()

if(NOT DEFINED ENV{CI})
    add_compile_definitions(BLADEBIT_VERSION_MAJ=${bb_ver_maj})
    add_compile_definitions(BLADEBIT_VERSION_MIN=${bb_ver_min})
    add_compile_definitions(BLADEBIT_VERSION_REV=${bb_ver_rev})
    add_compile_definitions(BLADEBIT_VERSION_SUFFIX="${bb_ver_suffix}")
    add_compile_definitions(BLADEBIT_GIT_COMMIT="${bb_ver_commit}")
endif()

if(MSVC)

    # MSVC
    set(c_opts ${c_opts}
        /std:c++17
        /Zc:__cplusplus
        /MP
        /Zi
        # /EHsc-
        # /Wall
        /W3
        /WX
        #/FIpch.h 
        /wd4068
        /wd4464
        /wd4668
        /wd4820
        /wd4514
        /wd4626
        /wd5027
        )
    
    set(tests_c_opts /DBB_TEST_MODE=1 ${tests_c_opts})
    
    set(link_opts 
        /SUBSYSTEM:CONSOLE
        /STACK:33554432,1048576
        ${link_opts})

    set(release_c_opts 
        /Oi
        /O2
        /Gy
        /GL
        ${release_c_opts})

    set(debug_c_opts ${debug_c_opts}
        /Od
        )

    set(dev_c_opts 
        ${dev_c_opts})

    set(c_defs 
        UNICODE=1
        WIN32_LEAN_AND_MEAN=1
        NOMINMAX=1
        _CRT_SECURE_NO_WARNINGS=1
        _HAS_EXCEPTIONS=0
        ${c_defs})
    
    set(release_link_opts 
        /DEBUG:FULL
        /LTCG
        /OPT:REF,ICF,LBR
        ${release_link_opts})

    set(debug_link_opts
#        /DEBUG:FASTLINK
        /OPT:NOREF,NOICF,NOLBR
#        /INCREMENTAL
        ${debug_link_opts})

    # Dependency config
    target_compile_options(bls     PRIVATE /MP)
    target_compile_options(relic_s PRIVATE /MP)
    target_compile_options(sodium  PRIVATE /MP)
else()

    # *Nix
    set(c_opts --include=pch.h -Wall -Wno-comment -Wno-unknown-pragmas -g ${c_opts})

    set(tests_c_opts -DBB_TEST_MODE=1 ${tests_c_opts})

    set(release_c_opts 
        -O3 #-flto
        -D_NDEBUG=1
        -DNDEBUG=1
        ${release_c_opts})

    set(debug_c_opts 
        -O0
        -DDEBUG=1
        -D_DEBUG=1
        ${debug_c_opts})

    set(dev_c_opts 
        ${dev_c_opts})

    set(link_opts -g -rdynamic #-flto
        ${link_opts})

    # GCC
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

        set(c_opts -fmax-errors=5 ${c_opts})

        # Avoid ranlib error: plugin needed to handle lto objectR "gcc-ar")
        # set(c_opts -ffat-lto-objects ${c_opts})
        
        # Build with native architecture when not building release packages
        if(NOT DEFINED ENV{CI})
            set(c_opts -march=native ${c_opts})
        endif()

    # Clang
    elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")

        set(c_opts -ferror-limit=5 -fdeclspec -Wno-empty-body ${c_opts})

    endif()

endif()

#
# Sources
#
file(GLOB_RECURSE bb_sources
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES false
    src/*.cpp
    src/*.c
)
set(src_full ${bb_sources})

# Headers
file(GLOB_RECURSE bb_headers
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES false
    src/*.h
    src/*.hpp
    src/*.inl
)

# Ignore some sources
list(FILTER bb_sources EXCLUDE REGEX "src/main\\.cpp")
list(FILTER bb_sources EXCLUDE REGEX "src/tools/FSETableGenerator.cpp")
list(FILTER bb_sources EXCLUDE REGEX "src/sandbox/.+")
list(FILTER bb_sources EXCLUDE REGEX "src/platform/.+")
list(FILTER bb_sources EXCLUDE REGEX "src/b3/blake3_(avx|sse).+")
list(FILTER bb_sources EXCLUDE REGEX "src/uint128_t/.+")


# Project-specific sources
file(GLOB_RECURSE src_tests RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS LIST_DIRECTORIES false
    tests/*.cpp
)

file(GLOB_RECURSE src_dev RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
    CONFIGURE_DEPENDS LIST_DIRECTORIES false
    src/sandbox/*.cpp
    src/sandbox/*.h
)


# Configure dependent on config/platform/architecture
# Architecture
if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "x86_64" OR ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64")

    if(NOT MSVC)
        list(APPEND bb_sources
            src/b3/blake3_avx2_x86-64_unix.S
            src/b3/blake3_avx512_x86-64_unix.S
            src/b3/blake3_sse41_x86-64_unix.S
        )
    else()
        list(APPEND bb_sources
            src/b3/blake3_avx2.c
            src/b3/blake3_avx512.c
            src/b3/blake3_sse41.c
            src/uint128_t/uint128_t.cpp
        )
        # Disable blake3 conversion loss of data warnings
        set_source_files_properties( 
            src/b3/blake3_avx2.c
            src/b3/blake3_avx512.c
            src/b3/blake3_sse41.c
            PROPERTIES COMPILE_FLAGS
            /wd4244
        )
    endif()

elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "arm64" OR ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
else()
    message( FATAL_ERROR "Unsupported architecture '${CMAKE_HOST_SYSTEM_PROCESSOR}'" )
endif()

# OS
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

    file(GLOB_RECURSE src_linux RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
        CONFIGURE_DEPENDS LIST_DIRECTORIES false
        src/platform/unix/*.cpp
        src/platform/linux/*.cpp
    )
    list(APPEND bb_sources ${src_linux})

elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    
    file(GLOB_RECURSE src_win RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
        CONFIGURE_DEPENDS LIST_DIRECTORIES false
        src/platform/win32/*.cpp
    )
    list(APPEND bb_sources ${src_win})

elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

    file(GLOB_RECURSE src_mac RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
        CONFIGURE_DEPENDS LIST_DIRECTORIES false
        src/platform/unix/*.cpp
        src/platform/macos/*.cpp
    )
    list(APPEND bb_sources ${src_mac})

else()
    message( FATAL_ERROR "Unsupported operating system '${CMAKE_SYSTEM_NAME}'" )
endif()

#
# Targets
#
set(bb_include_dirs 
    ${INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set(bb_c_opts)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(bb_c_opts ${bb_c_opts} /FIpch.h )
endif()


 macro(config_bladebit_proj tgt)
     message("Configuring target ${tgt}:${CMAKE_BUILD_TYPE}.")
     target_link_libraries(${tgt} PUBLIC Threads::Threads bls ${platform_libs})
     target_include_directories(${tgt} PUBLIC ${bb_include_dirs})

     target_compile_definitions(${tgt} PUBLIC $<$<CONFIG:Release>:${c_defs} ${release_c_defs}>)
     target_compile_definitions(${tgt} PUBLIC $<$<CONFIG:Debug>:${c_defs} ${debug_c_defs}>)

     target_compile_options(${tgt} PRIVATE $<$<CONFIG:Release>:${c_opts} ${release_c_opts}>)
     target_compile_options(${tgt} PRIVATE $<$<CONFIG:Debug>:${c_opts} ${debug_c_opts}>)
     target_compile_options(${tgt} PRIVATE ${bb_c_opts})

     target_link_options(${tgt} PRIVATE $<$<CONFIG:Release>:${link_opts} ${release_link_opts}>)
     target_link_options(${tgt} PRIVATE $<$<CONFIG:Debug>:${link_opts} ${debug_link_opts}>)
 endmacro()

# BladeBit
add_library(lib_bladebit ${bb_sources} ${bb_headers}
    cuda/harvesting/CudaThresherDummy.cpp # Dummy CudaThresher factory
)
set_target_properties(lib_bladebit PROPERTIES OUTPUT_NAME bladebit )
config_bladebit_proj(lib_bladebit)
#target_link_libraries(lib_bladebit PUBLIC Threads::Threads bls ${platform_libs})
#target_include_directories(lib_bladebit PUBLIC ${bb_include_dirs})
#
#target_compile_definitions(lib_bladebit PUBLIC $<$<CONFIG:Release>:${c_defs} ${release_c_defs}>)
#target_compile_definitions(lib_bladebit PUBLIC $<$<CONFIG:Debug>:${c_defs} ${debug_c_defs}>)
#
#target_compile_options(lib_bladebit PRIVATE $<$<CONFIG:Release>:${c_opts} ${release_c_opts}>)
#target_compile_options(lib_bladebit PRIVATE $<$<CONFIG:Debug>:${c_opts} ${debug_c_opts}>)
#target_compile_options(lib_bladebit PRIVATE ${bb_c_opts})
#
#target_link_options(lib_bladebit PRIVATE $<$<CONFIG:Release>:${link_opts} ${release_link_opts}>)
#target_link_options(lib_bladebit PRIVATE $<$<CONFIG:Debug>:${link_opts} ${debug_link_opts}>)


add_executable(bladebit ${bb_headers} src/main.cpp)
target_compile_options(bladebit PRIVATE $<$<CONFIG:Release>:${c_opts} ${release_c_opts}>)
target_compile_options(bladebit PRIVATE $<$<CONFIG:Debug>:${c_opts} ${debug_c_opts}>)
target_compile_options(bladebit PRIVATE ${bb_c_opts})
target_link_libraries(bladebit PRIVATE lib_bladebit)

# Tests
add_executable(tests ${src_tests} ${bb_headers})
target_compile_options(tests PRIVATE $<$<CONFIG:Release>:${c_opts} ${release_c_opts} ${tests_c_opts}>)
target_compile_options(tests PRIVATE $<$<CONFIG:Debug>:${c_opts} ${debug_c_opts} ${tests_c_opts}>)
target_compile_options(tests PRIVATE ${bb_c_opts})
target_link_libraries(tests PRIVATE lib_bladebit Catch2::Catch2WithMain)


#
# CUDA Plotter
#
if(CUDAToolkit_FOUND)

set(is_release "$<CONFIG:Release>")
set(is_debug "$<CONFIG:Debug>")
set(is_cuda "$<COMPILE_LANGUAGE:CUDA>")
set(is_cuda_release "$<AND:${is_cuda},${is_release}>")
set(is_cuda_debug "$<AND:${is_cuda},${is_debug}>")
set(cuda_release_c_opts ${c_opts} ${debug_c_opts})
set(cuda_debug_c_opts ${c_opts} ${debug_c_opts})

add_library(lib_bladebit_cuda_core ${bb_sources} ${bb_headers}
#        src/plotdisk/k32/FpMatchBounded.inl src/plotdisk/k32/F1Bounded.inl
        )
set_target_properties(lib_bladebit_cuda_core PROPERTIES OUTPUT_NAME bladebit_cuda_core )
config_bladebit_proj(lib_bladebit_cuda_core)
target_compile_definitions(lib_bladebit_cuda_core PUBLIC
    BB_CUDA_ENABLED=1
    THRUST_IGNORE_CUB_VERSION_CHECK=1)
target_include_directories(lib_bladebit_cuda_core PRIVATE SYSTEM cuda)
target_link_libraries(lib_bladebit_cuda_core PUBLIC CUDA::cudart_static CUDA::cuda_driver)

add_library(lib_bladebit_cuda
    cuda/CudaPlotter.cu
    cuda/CudaPlotter.h
    cuda/CudaPlotContext.h
    cuda/CudaPlotPhase2.cu
    cuda/CudaPlotPhase3.cu
    cuda/CudaPlotPhase3Step2.cu
    cuda/CudaPlotPhase3Step3.cu
    cuda/CudaPlotPhase3Internal.h
    cuda/CudaParkSerializer.h
    cuda/CudaParkSerializer.cu
    cuda/chacha8.cu
    cuda/CudaF1.h
    cuda/CudaF1.cu
    cuda/CudaMatch.h
    cuda/CudaMatch.cu
    cuda/CudaFx.h
    cuda/FxCuda.cu
    cuda/CudaUtil.h
    cuda/CudaPlotUtil.cu
    cuda/GpuStreams.h
    cuda/GpuStreams.cpp

    # Harvester
    cuda/harvesting/CudaThresher.cu
)

target_include_directories(lib_bladebit_cuda PUBLIC ${bb_include_dirs} src cuda)
target_include_directories(lib_bladebit_cuda PRIVATE SYSTEM cuda)
#target_compile_definitions(lib_bladebit_cuda PRIVATE THRUST_IGNORE_CUB_VERSION_CHECK=1)
target_compile_definitions(lib_bladebit_cuda PRIVATE $<$<CONFIG:Release>:${c_defs} ${release_c_defs}>)
target_compile_definitions(lib_bladebit_cuda PRIVATE $<$<CONFIG:Debug>:${c_defs} ${debug_c_defs}>)

target_compile_options(lib_bladebit_cuda PUBLIC $<${is_cuda}:--pre-include pch.h> )

target_compile_options(lib_bladebit_cuda PRIVATE 
    $<${is_cuda_release}:
        -gencode=arch=compute_52,code=sm_52 # Maxwell
        -gencode=arch=compute_61,code=sm_61 # Pascal
        -gencode=arch=compute_70,code=sm_70 # Volta 
        -gencode=arch=compute_86,code=sm_86 # Ampere
        -gencode=arch=compute_89,code=sm_89 # Ada
    >)

 target_compile_options(lib_bladebit_cuda PRIVATE 
    $<${is_cuda_debug}:
    # -arch=native

    -G
    >
 )

# target_compile_options(lib_bladebit_cuda PRIVATE "$<${is_cuda_debug}:-Xcompiler=${cuda_debug_c_opts}">)
# target_compile_options(lib_bladebit_cuda PRIVATE "$<${is_cuda_debug}:-G">)
# target_compile_options(lib_bladebit_cuda PRIVATE $<${is_cuda}:-Xcompiler=/FIpch.h>)

# target_compile_options(lib_bladebit_cuda PRIVATE $<$<CONFIG:Release>:${c_opts} ${release_c_opts}>)
# target_compile_options(lib_bladebit_cuda PRIVATE $<$<CONFIG:Debug>:${c_opts} ${debug_c_opts}>)
target_link_libraries(lib_bladebit_cuda PUBLIC lib_bladebit_cuda_core)


add_executable(bladebit_cuda ${bb_headers} src/main.cpp)

#target_compile_definitions(bladebit_cuda PUBLIC BB_CUDA_ENABLED=1)
target_compile_options(bladebit_cuda PRIVATE $<$<CONFIG:Release>:${c_opts} ${release_c_opts}>)
target_compile_options(bladebit_cuda PRIVATE $<$<CONFIG:Debug>:${c_opts} ${debug_c_opts}>)
target_compile_options(bladebit_cuda PRIVATE ${bb_c_opts})
target_link_libraries(bladebit_cuda PRIVATE lib_bladebit_cuda)

set_target_properties(lib_bladebit_cuda_core lib_bladebit_cuda bladebit_cuda PROPERTIES 
    MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>
    CUDA_RUNTIME_LIBRARY Static
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON  # Required under windows
    CUDA_ARCHITECTURES OFF
)

# Pretty source view for IDE projects
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src 
    FILES ${src_full} ${bb_headers}
)
endif()