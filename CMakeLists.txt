cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(bladebit)

set(CMAKE_CONFIGURATION_TYPES release debug)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "release"
        CACHE STRING "Possible values are: release, debug"
        FORCE
   )
endif()
message("Config: ${CMAKE_BUILD_TYPE}")


# Grab BLS
include(FetchContent)

set(BUILD_BLS_PYTHON_BINDINGS "0" CACHE STRING "0")
set(BUILD_BLS_TESTS "0" CACHE STRING "")
set(BUILD_BLS_BENCHMARKS "0" CACHE STRING "")

FetchContent_Declare(
  bls
  GIT_REPOSITORY https://github.com/Chia-Network/bls-signatures.git
  GIT_TAG        1.0.6
)
FetchContent_MakeAvailable(bls)

set(CMAKE_C_FLAGS_RELEASE_INIT)
set(CMAKE_C_FLAGS_DEBUG_INIT)
set(CMAKE_CXX_FLAGS_RELEASE_INIT)
set(CMAKE_CXX_FLAGS_DEBUG_INIT)


set(c_opts)
set(link_opts)
# set(cxx_opts)

set(release_c_opts)
set(debug_c_opts)
set(dev_c_opts)

set(release_link_opts)
set(debug_link_opts)

# Config
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")

    # MSVC
    set(c_opts
        /std:c++17
        /Zc:__cplusplus
        /MP
        /Zi
        /EHsc-
        # /Wall
        /W3
        /WX
        /FIpch.h 
        /wd4068
        /wd4464
        /wd4668
        /wd4820
        /wd4514
        /wd4626
        /wd5027
        /DUNICODE=1
        /DWIN32_LEAN_AND_MEAN=1
        /DNOMINMAX=1
        /D_CRT_SECURE_NO_WARNINGS=1
        /D_HAS_EXCEPTIONS=0
        ${c_opts})
    
    set(link_opts 
        /SUBSYSTEM:CONSOLE
        /STACK:33554432,1048576
        ${link_opts})

    set(release_c_opts 
        /MT
        /Oi
        /O2
        /Gy
        /GL
        /DNDEBUG=1
        /D_NDEBUG=1
        ${release_c_opts})

    set(debug_c_opts 
        /MTd
        /Od
        /DDEBUG=1
        /D_DEBUG=1
        ${debug_c_opts})

    set(dev_c_opts 
        ${dev_c_opts})

    
    set(release_link_opts 
        /DEBUG:FULL
        /LTCG
        /OPT:REF,ICF,LBR
        ${release_link_opts})

    set(debug_link_opts
        /DEBUG:FASTLINK
        /OPT:NOREF,NOICF,NOLBR
        /INCREMENTAL
        ${debug_link_opts})

    # Dependency config
    target_compile_options(bls     PRIVATE /MP)
    target_compile_options(relic_s PRIVATE /MP)
    target_compile_options(sodium  PRIVATE /MP)

    target_compile_options(bls     PRIVATE $<$<CONFIG:release>:/MT>)
    target_compile_options(relic_s PRIVATE $<$<CONFIG:release>:/MT>)
    target_compile_options(sodium  PRIVATE $<$<CONFIG:release>:/MT>)

    target_compile_options(bls     PRIVATE $<$<CONFIG:debug>:/MTd>)
    target_compile_options(relic_s PRIVATE $<$<CONFIG:debug>:/MTd>)
    target_compile_options(sodium  PRIVATE $<$<CONFIG:debug>:/MTd>)
    
else()

    # *Nix
    set(CMAKE_C_FLAGS --include=pch.h -Wall -Wno-comment -g ${CMAKE_C_FLAGS})
    set(CMAKE_CXX_FLAGS -std=c++17 ${CMAKE_CXX_FLAGS})

    set(release_c_opts 
        -O3 -flto
        ${release_c_opts})

    set(debug_c_opts 
        -O0
        ${debug_c_opts})

    set(dev_c_opts 
        ${dev_c_opts})

    # GCC
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    
        set(CMAKE_C_FLAGS -fmax-errors=5 ${CMAKE_C_FLAGS})

    # Clang
    elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        
        set(CMAKE_C_FLAGS -ferror-limit=5 -fdeclspec -fno-exceptions -Wunknown-pragmas ${CMAKE_C_FLAGS})

    endif()

endif()


# Main Sources
file(GLOB_RECURSE bb_sources
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES false
    src/*.cpp
    src/*.c
)

# Headers
file(GLOB_RECURSE bb_headers
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES false
    src/*.h
    src/*.hpp
    src/*.inl
)

# Configure dependent on config/platform/architecture
list(FILTER bb_sources EXCLUDE REGEX "src/main\.cpp")
list(FILTER bb_sources EXCLUDE REGEX "src/(test|platform)/.+")
list(FILTER bb_sources EXCLUDE REGEX "src/b3/blake3_(avx|sse).+")

# Architecture
if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "x86_64" OR ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64")
    
    list(APPEND bb_sources
        src/b3/blake3_avx2.c
        src/b3/blake3_avx512.c
        src/b3/blake3_sse41.c
    )

    if(NOT MSVC)
        list(APPEND bb_sources
            src/b3/blake3_avx2_x86-64_unix.S
            src/b3/blake3_avx512_x86-64_unix.S
            src/b3/blake3_sse41_x86-64_unix.S
        )
    else()
        # Disable blake3 convertion loss of data warnings 
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

# Dev config?
file(GLOB_RECURSE src_dev RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
    CONFIGURE_DEPENDS LIST_DIRECTORIES false
    src/test/*.cpp
)

file(GLOB_RECURSE headers_dev RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
    CONFIGURE_DEPENDS LIST_DIRECTORIES false
    src/test/*.h
)


# Exe
find_package(Threads REQUIRED)

set(platform_libs)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(platform_libs numa)
endif()

set(bb_include_dirs 
    ${INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${bls_SOURCE_DIR}/include
    ${relic_SOURCE_DIR}/include
)


# BladeBit
add_executable(bladebit     src/main.cpp ${bb_sources} ${bb_headers})
add_executable(bladebit_dev src/test/test_main.cpp ${bb_sources} ${src_dev} ${bb_headers} ${headers_dev})

macro(config_proj tgt)

    target_compile_options(${tgt} PRIVATE $<$<CONFIG:release>:${c_opts} ${release_c_opts}>)
    target_compile_options(${tgt} PRIVATE $<$<CONFIG:debug>:${c_opts} ${debug_c_opts}>)
    target_link_options(${tgt} PRIVATE $<$<CONFIG:release>:${link_opts} ${release_link_opts}>)
    target_link_options(${tgt} PRIVATE $<$<CONFIG:debug>:${link_opts} ${debug_link_opts}>)

    target_include_directories(${tgt} PRIVATE ${bb_include_dirs})
    target_link_libraries(${tgt} Threads::Threads bls ${platform_libs})
endmacro()

config_proj(bladebit)
config_proj(bladebit_dev)


# Pretty source view for IDE projects
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src 
    FILES ${bb_sources} ${bb_headers} ${src_dev} ${headers_dev}
)

