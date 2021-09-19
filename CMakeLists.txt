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
        /D_HAS_EXCEPTIONS=0
        /MP
        /Zi
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
        ${c_opts})
    
    set(link_opts 
        /SUBSYSTEM:CONSOLE
        ${link_opts})

    set(release_c_opts 
        /MT
        /O2
        /Oi
        /Gy
        /GL
        /DNDEBUG=1
        /DNDEBUG=1
        ${release_c_opts})

    set(debug_c_opts 
        /MTd
        /Od
        ${debug_c_opts})

    set(dev_c_opts 
        ${dev_c_opts})

    
    set(release_link_opts 
        /DEBUG:FULL
        /OPT:REF,IFC,LBR
        ${release_link_opts})

    set(debug_link_opts
        /DEBUG:FASTLINK
        /OPT:NOREF,NOICF,NOLBR
        /INCREMENTAL
        ${debug_link_opts})
    
else()

    # *Nix
    set(CMAKE_C_FLAGS "--include=pch.h -Wall -Wno-comment -g ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")

    set(release_c_opts "
        -O3 -flto
        ${release_c_opts}")

    set(debug_c_opts "
        -O0
        ${debug_c_opts}")

    set(dev_c_opts "
        ${dev_c_opts}")

    # GCC
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    
        set(CMAKE_C_FLAGS "-fmax-errors=5 ${CMAKE_C_FLAGS}")

    # Clang
    elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        
        set(CMAKE_C_FLAGS "-ferror-limit=5 -fdeclspec -fno-exceptions -Wunknown-pragmas ${CMAKE_C_FLAGS}")

    endif()

endif()

# set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_CXX_FLAGS}")

# set(CMAKE_C_FLAGS_RELEASE       "${CMAKE_C_FLAGS} ${release_c_opts}")
# set(CMAKE_C_FLAGS_DEBUG         "${CMAKE_C_FLAGS} ${debug_c_opts}")
# set(CMAKE_C_FLAGS_RELEASE_DEV   "${CMAKE_C_FLAGS_RELEASE} ${dev_c_opts}")
# set(CMAKE_C_FLAGS_DEBUG_DEV     "${CMAKE_C_FLAGS_DEBUG} ${dev_c_opts}")

# set(CMAKE_CXX_FLAGS_RELEASE     "${CMAKE_CXX_FLAGS} ${release_c_opts}")
# set(CMAKE_CXX_FLAGS_DEBUG       "${CMAKE_CXX_FLAGS} ${debug_c_opts}")
# set(CMAKE_CXX_FLAGS_RELEASE_DEV "${CMAKE_CXX_FLAGS_RELEASE} ${dev_c_opts}")
# set(CMAKE_CXX_FLAGS_DEBUG_DEV   "${CMAKE_CXX_FLAGS_DEBUG} ${dev_c_opts}")

# set(CMAKE_EXE_LINKER_FLAGS_RELEASE      "${CMAKE_EXE_LINKER_FLAGS} ${release_link_opts}" )
# set(CMAKE_EXE_LINKER_FLAGS_DEBUG        "${CMAKE_EXE_LINKER_FLAGS} ${debug_link_opts}" )
# set(CMAKE_EXE_LINKER_FLAGS_RELEASE_DEV  "${CMAKE_EXE_LINKER_FLAGS_RELEASE}" )
# set(CMAKE_EXE_LINKER_FLAGS_DEBUG_DEV    "${CMAKE_EXE_LINKER_FLAGS_DEBUG}" )

# Main Sources
file(GLOB_RECURSE bb_sources
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES false
    src/*.cpp
    src/*.c
)

# Headrs
file(GLOB_RECURSE bb_headers
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES false
    src/*.h
    src/*.hpp
    src/*.inl
)

# Configure dependent on config/platform/architecture
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
            "/wd4244"
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
if(${CMAKE_BUILD_TYPE} MATCHES "(release|debug)_dev")
    file(GLOB_RECURSE src_dev RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
        CONFIGURE_DEPENDS LIST_DIRECTORIES false
        src/test/*.cpp
    )
    list(APPEND bb_sources ${src_dev})
endif()


# Exe
find_package(Threads REQUIRED)

add_executable(bladebit ${bb_sources} ${bb_headers})

target_compile_options(bladebit
    PUBLIC
    $<$<CONFIG:release>:${c_opts} ${release_c_opts}>
)
target_compile_options(bladebit
    PUBLIC
    $<$<CONFIG:debug>:${c_opts} ${debug_c_opts}>
)

target_link_options(bladebit
    PUBLIC
    $<$<CONFIG:release>:${link_opts} ${debug_link_opts}>
)

target_link_options(bladebit
    PUBLIC
    $<$<CONFIG:debug>:${link_opts} ${debug_link_opts}>
)

target_include_directories(bladebit
    PRIVATE
    ${INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${bls_SOURCE_DIR}/include
    ${relic_SOURCE_DIR}/include
)

set(platform_libs)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(platform_libs numa)
endif()

target_link_libraries(bladebit Threads::Threads bls ${platform_libs})



# message("${bb_sources}")
# message( "${CMAKE_SYSTEM_NAME}" )
# message( "${CMAKE_HOST_SYSTEM_PROCESSOR}" )

macro(dump_list ls)
    foreach(l ${bb_sources})
        message("> ${l}")
    endforeach()
endmacro()

# dump_list(${bb_sources})
