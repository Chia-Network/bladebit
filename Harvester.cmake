add_library(bladebit_harvester_base INTERFACE)

add_library(bladebit_harvester STATIC)
target_link_libraries(bladebit_harvester PRIVATE bladebit_harvester_base)

add_library(bladebit_harvester_dynamic SHARED)
target_link_libraries(bladebit_harvester_dynamic PRIVATE bladebit_harvester_base)

if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set_target_properties(bladebit_harvester_dynamic PROPERTIES OUTPUT_NAME "bladebit_harvester")
endif()


set_property(TARGET bladebit_harvester bladebit_harvester_dynamic PROPERTY PUBLIC_HEADER 
    src/harvesting/GreenReaper.h 
    src/harvesting/GreenReaperPortable.h)

install(TARGETS bladebit_harvester bladebit_harvester_dynamic
    LIBRARY DESTINATION green_reaper/lib
    ARCHIVE DESTINATION green_reaper/lib
    PUBLIC_HEADER DESTINATION green_reaper/include
)

target_sources(bladebit_harvester_base INTERFACE
    src/pch.cpp

    src/pos/chacha8.cpp
    src/pos/chacha8.h

    src/fse/bitstream.h
    src/fse/compiler.h
    src/fse/debug.c
    src/fse/debug.h
    src/fse/entropy_common.c
    src/fse/error_private.h
    src/fse/error_public.h
    src/fse/fse_compress.c
    src/fse/fse_decompress.c
    src/fse/fse.h
    src/fse/hist.c
    src/fse/hist.h
    src/fse/huf.h
    src/fse/mem.h

    src/b3/blake3.c
    src/b3/blake3_dispatch.c
    src/b3/blake3.h
    src/b3/blake3_impl.h
    src/b3/blake3_portable.c

    $<${is_x86}:
        $<$<PLATFORM_ID:Windows>:
            src/b3/blake3_sse41.c
            src/b3/blake3_avx2.c
            src/b3/blake3_avx512.c
        >
        $<$<NOT:$<PLATFORM_ID:Windows>>:
            src/b3/blake3_avx2_x86-64_unix.S
            src/b3/blake3_avx512_x86-64_unix.S
            src/b3/blake3_sse41_x86-64_unix.S
        >
    >
    

    src/util/Log.cpp
    src/util/Util.cpp
    src/PlotContext.cpp
    src/io/HybridStream.cpp
    src/threading/AutoResetSignal.cpp
    src/threading/Fence.cpp
    src/threading/Semaphore.cpp
    src/threading/ThreadPool.cpp
    src/plotting/FSETableGenerator.cpp
    src/plotting/PlotWriter.cpp
    src/plotting/Compression.cpp
    src/plotting/matching/GroupScan.cpp
    src/plotdisk/DiskBufferQueue.cpp
    src/plotting/WorkHeap.cpp
    src/plotdisk/jobs/IOJob.cpp
    src/harvesting/GreenReaper.cpp

    src/bech32/segwit_addr.c

    $<${have_cuda}:
        cuda/harvesting/CudaThresher.cu
        cuda/harvesting/CudaThresherFactory.cu
        cuda/FxCuda.cu
        cuda/CudaF1.cu
        cuda/CudaMatch.cu
        cuda/CudaPlotUtil.cu

        # TODO: Remove this, ought not be needed in harvester
        cuda/GpuStreams.cu
    >

    $<$<NOT:${have_cuda}>:
        cuda/harvesting/CudaThresherDummy.cpp
    >

    $<$<PLATFORM_ID:Windows>:
        src/platform/win32/SysHost_Win32.cpp
        src/platform/win32/FileStream_Win32.cpp
        src/platform/win32/Thread_Win32.cpp
    >

    $<$<PLATFORM_ID:Linux>:
        src/platform/linux/SysHost_Linux.cpp
    >

    $<$<PLATFORM_ID:Darwin>:
        src/platform/macos/SysHost_Macos.cpp
    >

    $<$<PLATFORM_ID:Darwin,Linux>:
        src/platform/unix/FileStream_Unix.cpp
        src/platform/unix/Thread_Unix.cpp
    >

    $<$<CXX_COMPILER_ID:MSVC>:
        src/uint128_t/uint128_t.cpp
    >
)

target_include_directories(bladebit_harvester_base INTERFACE src SYSTEM cuda)

target_compile_features(bladebit_harvester_base INTERFACE cxx_std_17)

target_compile_definitions(bladebit_harvester_base INTERFACE
    BB_IS_HARVESTER=1
    THRUST_IGNORE_CUB_VERSION_CHECK=1
    GR_EXPORT=1
    $<${have_cuda}:
        BB_CUDA_ENABLED=1
    >
)

target_compile_options(bladebit_harvester_base INTERFACE 
    ${preinclude_pch}
    # $<${have_cuda}:${cuda_archs}>
)

target_link_options(bladebit_harvester_base INTERFACE $<DEVICE_LINK: ${cuda_archs}>)

target_link_libraries(bladebit_harvester_base INTERFACE 
    bladebit_config 
    Threads::Threads

    $<${have_cuda}:CUDA::cudart_static>

    INTERFACE
        $<$<PLATFORM_ID:Linux>:
            # ${NUMA_LIBRARY}
            # dl
        >
)

if(CUDAToolkit_FOUND)
    set_target_properties(bladebit_harvester_base PROPERTIES 
        EXCLUDE_FROM_ALL ON
        MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>
        CUDA_RUNTIME_LIBRARY Static
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        # CUDA_ARCHITECTURES OFF
    )
endif()

 # Disable blake3 conversion loss of data warnings
 if("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
    set_source_files_properties( 
        src/b3/blake3_avx2.c
        src/b3/blake3_avx512.c
        src/b3/blake3_sse41.c
        PROPERTIES COMPILE_FLAGS
        /wd4244
    )
 endif()

