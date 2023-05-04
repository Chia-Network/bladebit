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

set_property(TARGET bladebit_harvester PROPERTY PUBLIC_HEADER 
    src/harvesting/GreenReaper.h 
    src/harvesting/GreenReaperPortable.h)

install(TARGETS bladebit_harvester
    LIBRARY DESTINATION green_reaper/lib
    PUBLIC_HEADER DESTINATION green_reaper/include
)

target_include_directories(bladebit_harvester PRIVATE
    src
    $<${have_cuda}:
        cuda
        SYSTEM cuda
    >
)

target_compile_features(bladebit_harvester PRIVATE cxx_std_17)

target_compile_definitions(bladebit_harvester PRIVATE
    BB_CUDA_ENABLED=1
    THRUST_IGNORE_CUB_VERSION_CHECK=1
    GR_EXPORT=1
)

target_compile_options(bladebit_harvester PRIVATE 
    ${preinclude_pch}

    $<${have_cuda}:
        ${cuda_archs}
    >
)

target_link_libraries(bladebit_harvester PRIVATE 
    bladebit_config 
    Threads::Threads
    bls

    $<${have_cuda}:
        CUDA::cudart_static
    >

    INTERFACE
        $<$<PLATFORM_ID:Linux>:
            # ${NUMA_LIBRARY}
            dl
        >
)

if(CUDAToolkit_FOUND)
    set_target_properties(bladebit_harvester PROPERTIES 
        EXCLUDE_FROM_ALL ON
        MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>
        CUDA_RUNTIME_LIBRARY Static
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES OFF
    )
endif()
