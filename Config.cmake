# Base interface configuration project
add_library(bladebit_config INTERFACE)

target_compile_definitions(bladebit_config INTERFACE
    $<${is_release}:
        _NDEBUG=1
        NDEBUG=1
    >
    $<${is_debug}:
        _DEBUG=1
        DEBUG=1
    >

    $<$<CXX_COMPILER_ID:MSVC>:
        UNICODE=1
        WIN32_LEAN_AND_MEAN=1
        NOMINMAX=1
        _CRT_SECURE_NO_WARNINGS=1
        _HAS_EXCEPTIONS=0
    >
)

target_compile_options(bladebit_config INTERFACE

    # GCC or Clang
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wall
        -Wno-comment
        -Wno-unknown-pragmas
        -g

        $<${is_release}:
            -O3
        >

        $<${is_debug}:
            -O0
        >
    >
    
    # GCC
    $<$<CXX_COMPILER_ID:GNU>:
        -fmax-errors=5
    >

    # Clang
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:
        -ferror-limit=5
        -fdeclspec
        -Wno-empty-body
    >

    # MSVC
    $<${is_msvc_c_cpp}:
        /Zc:__cplusplus
        /MP
        /Zi
        # /EHsc-
        # /Wall
        /W3
        /WX
        /wd4068
        /wd4464
        /wd4668
        /wd4820
        /wd4514
        /wd4626
        /wd5027

        $<${is_release}:
            /Oi /O2 /Gy /GL
        >
        
        $<${is_debug}:
            /Od
        >
    >

    $<${is_x86}:
    >

    $<${is_arm}:
    >
)

target_link_options(bladebit_config INTERFACE

    # GCC or Clang
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -g
        -rdynamic
    >

    # MSVC
    $<${is_msvc_c_cpp}:

        /SUBSYSTEM:CONSOLE
        /STACK:33554432,1048576

        $<${is_release}:
            /DEBUG:FULL
            /LTCG
            /OPT:REF,ICF,LBR
        >

        $<${is_debug}:
            # /DEBUG:FASTLINK
            # /OPT:NOREF,NOICF,NOLBR
            # /INCREMENTAL
        >
    >
)

set_property(DIRECTORY . PROPERTY MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>)
set_property(DIRECTORY . PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(DIRECTORY . PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(preinclude_pch
    $<${is_cuda}:--pre-include pch.h>
    $<${is_c_cpp}:
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:--include=pch.h>
    >
    $<${is_msvc_c_cpp}:/FIpch.h>
)

# See: https://gitlab.kitware.com/cmake/cmake/-/issues/18265
cmake_policy(SET CMP0105 NEW)

set(cuda_archs

    $<${is_cuda_release}:
        -gencode=arch=compute_52,code=sm_52 # Maxwell
        -gencode=arch=compute_61,code=sm_61 # Pascal
        -gencode=arch=compute_70,code=sm_70 # Volta 
        -gencode=arch=compute_86,code=sm_86 # Ampere
        -gencode=arch=compute_89,code=sm_89 # Ada
    >

    $<${is_cuda_debug}:
        -arch=native
        # -gencode=arch=compute_52,code=sm_52 # Maxwell
    >
)
