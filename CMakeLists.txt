cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(bladebit)

set(CMAKE_CONFIGURATION_TYPES release debug release.test debug.test)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "release"
        CACHE STRING "Possible values are: release, debug, release.test, debug.test"
        FORCE
   )
endif()
message("Config: ${CMAKE_BUILD_TYPE}")

# Grab BLS
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules
)

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


# Config
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_C_FLGS " ${CMAKE_C_FLAGS}")
else()

    set(CMAKE_C_FLGS "--include=src/pch.h -Wall -Wno-comment -g ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")

    if(${CMAKE_BUILD_TYPE} MATCHES "(release)\.?.*")
        set(CMAKE_C_FLGS "-O3 -flto ${CMAKE_C_FLAGS}")
    else()
        set(CMAKE_C_FLGS "-O0 ${CMAKE_C_FLAGS}")
    endif()

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(CMAKE_C_FLGS "-fmax-errors=5 ${CMAKE_C_FLAGS}")
    elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        set(CMAKE_C_FLGS "-ferror-limit=5 -fdeclspec -fno-exceptions -Wunknown-pragmas ${CMAKE_C_FLAGS}")
    endif()

endif()

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLGS} ${CMAKE_CXX_FLAGS}")

# Main Sources
file(GLOB_RECURSE bb_sources
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES false
    src/**/*.cpp
    src/**/*.c
)

# Configure dependent on config/platform/architecture
list(FILTER bb_sources EXCLUDE REGEX "src/(test|platform)/.+")
list(FILTER bb_sources EXCLUDE REGEX "src/b3/blake3_(avx|sse).+")

# Architecture
if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "x86_64")
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

# Test config?
if(${CMAKE_BUILD_TYPE} MATCHES "(release|debug)\.test")
    file(GLOB_RECURSE src_test RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
        CONFIGURE_DEPENDS LIST_DIRECTORIES false
        src/test/*.cpp
    )
    list(APPEND bb_sources ${src_test})
endif()


# Exe
find_package(Threads REQUIRED)

add_executable(bladebit ${bb_sources})

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
