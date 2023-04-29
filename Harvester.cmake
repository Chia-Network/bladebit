add_library(bladebit_harvester SHARED

    src/pch.cpp

    src/util/Log.cpp
    src/util/Util.cpp
    src/io/HybridStream.cpp
    src/util/KeyTools.cpp
    src/PlotContext.cpp
    src/threading/AutoResetSignal.cpp
    src/threading/Fence.cpp
    src/threading/Semaphore.cpp
    src/threading/ThreadPool.cpp
    src/plotting/PlotTools.cpp
    src/plotting/FSETableGenerator.cpp
    src/plotting/PlotWriter.cpp
    src/plotting/Compression.cpp
    src/plotting/matching/GroupScan.cpp
    src/plotdisk/DiskBufferQueue.cpp
    src/plotting/WorkHeap.cpp
    src/plotdisk/jobs/IOJob.cpp
    src/harvesting/GreenReaper.cpp

    src/bech32/segwit_addr.c

    cuda/harvesting/CudaThresher.cu
    cuda/harvesting/CudaThresherFactory.cu
    cuda/FxCuda.cu
    cuda/CudaF1.cu
    cuda/CudaMatch.cu
    cuda/CudaPlotUtil.cu

    # TODO: Remove this, ought not be needed in harvester
    cuda/GpuStreams.cu

    $<$<PLATFORM_ID:Windows>:
        src/platform/win32/SysHost_Win32.cpp
        src/platform/unix/FileStream_Win32.cpp
        src/platform/unix/Thread_Win32.cpp
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
)

set_target_properties(bladebit_harvester PROPERTIES 
    EXCLUDE_FROM_ALL ON
    MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>
    CUDA_RUNTIME_LIBRARY Static
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES OFF
)


target_include_directories(bladebit_harvester PUBLIC ${bb_include_dirs} src cuda)
target_include_directories(bladebit_harvester PRIVATE SYSTEM cuda)

target_compile_features(bladebit_harvester PRIVATE cxx_std_17)

target_compile_options(bladebit_harvester PRIVATE 
    ${c_opts}
    $<$<CONFIG:Release>:${release_c_opts}>
    $<$<CONFIG:Debug>:${debug_c_opts}>
    $<${is_cuda}:
        --pre-include pch.h
    >
    $<${is_c_cpp}:
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:--include=pch.h>
        $<$<CXX_COMPILER_ID:MSVC>:/FIpch.h>
    >
)

target_compile_definitions(bladebit_harvester PRIVATE
    BB_CUDA_ENABLED=1
    THRUST_IGNORE_CUB_VERSION_CHECK=1
)

target_link_libraries(bladebit_harvester
    PRIVATE Threads::Threads bls ${platform_libs}
    CUDA::cudart_static CUDA::cuda_driver
)

