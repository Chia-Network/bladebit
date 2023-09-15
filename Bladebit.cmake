add_library(bladebit_core)
target_link_libraries(bladebit_core PUBLIC bladebit_config)

target_include_directories(bladebit_core PUBLIC
    ${INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(bladebit_core PUBLIC
    GR_NO_IMPORT=1
    BB_NUMA_ENABLED=1
)

target_compile_options(bladebit_core PUBLIC ${preinclude_pch})

target_link_libraries(bladebit_core PUBLIC 
    Threads::Threads
    bls

    $<$<PLATFORM_ID:Linux>:
        ${NUMA_LIBRARY}
    >
)

add_executable(bladebit
    src/main.cpp
    cuda/harvesting/CudaThresherDummy.cpp)

target_link_libraries(bladebit PRIVATE bladebit_core)


# Sources
set(src_uint128
    src/uint128_t/endianness.h
    src/uint128_t/uint128_t.cpp
    src/uint128_t/uint128_t.h
)

set(src_chacha8
    src/pos/chacha8.cpp
    src/pos/chacha8.h
)

set(src_fse
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
)

set(src_blake3
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
)

set(src_bech32
    src/bech32/segwit_addr.c
    src/bech32/segwit_addr.h
)

set(src_bladebit

    # third party
    $<$<CXX_COMPILER_ID:MSVC>:
        ${src_uint128}
    >

    ${src_chacha8}
    ${src_fse}
    ${src_blake3}
    ${src_bech32}

    # bladebit
    $<$<PLATFORM_ID:Linux>:
        src/platform/linux
        src/platform/linux/SysHost_Linux.cpp
    >

    $<$<PLATFORM_ID:Darwin>:
        src/platform/macos/SysHost_Macos.cpp
    >

    $<$<PLATFORM_ID:Linux,Darwin>:
        src/platform/unix/FileStream_Unix.cpp
        src/platform/unix/Thread_Unix.cpp
    >

    $<$<PLATFORM_ID:Windows>:
        src/platform/win32/FileStream_Win32.cpp
        src/platform/win32/SysHost_Win32.cpp
        src/platform/win32/Thread_Win32.cpp
    >

    src/BLS.h
    src/Config.h
    src/ChiaConsts.h
    src/Globals.h
    src/Types.h
    src/Platform.h
    src/PlotContext.h
    src/PlotContext.cpp
    src/PlotWriter.h
    src/PlotWriter.cpp
    src/SysHost.cpp
    src/SysHost.h
    src/View.h
    src/pch.cpp
    src/pch.h
    src/Version.h

    src/algorithm/YSort.cpp
    src/algorithm/YSort.h
    src/algorithm/RadixSort.h

    src/io/BucketStream.cpp
    src/io/BucketStream.h
    src/io/FileStream.cpp
    src/io/FileStream.h
    src/io/HybridStream.cpp
    src/io/HybridStream.h
    src/io/IOUtil.cpp
    src/io/IOUtil.h
    src/io/IStream.h
    src/io/MemoryStream.h

    src/plotdisk/BlockWriter.h
    src/plotdisk/DiskFp.h
    src/plotdisk/DiskPairReader.h
    src/plotdisk/DiskPlotDebug.cpp
    src/plotdisk/DiskPlotDebug.h
    src/plotdisk/DiskPlotInfo.h
    src/plotdisk/DiskPlotPhase2.h
    src/plotdisk/DiskPlotPhase3.cpp.disabled
    src/plotdisk/DiskPlotPhase3.h
    src/plotdisk/FileId.h
    src/plotdisk/FpFxGen.h
    src/plotdisk/FpGroupMatcher.h
    src/plotdisk/MapWriter.h
    
    src/plotdisk/jobs/JobShared.h
    src/plotdisk/jobs/LookupMapJob.h
    src/plotdisk/jobs/UnpackMapJob.cpp
    src/plotdisk/jobs/UnpackMapJob.h
    src/plotdisk/jobs/IOJob.cpp
    src/plotdisk/jobs/IOJob.h
    
    src/plotdisk/k32/DiskPlotBounded.h
    src/plotdisk/k32/FpMatchBounded.inl
    src/plotdisk/k32/CTableWriterBounded.h
    src/plotdisk/k32/DiskPlotBounded.cpp
    src/plotdisk/k32/F1Bounded.inl
    src/plotdisk/k32/FxBounded.inl

    src/plotdisk/DiskF1.h
    src/plotdisk/DiskPlotConfig.h
    src/plotdisk/DiskPlotContext.h
    src/plotdisk/DiskPlotPhase2.cpp
    src/plotdisk/DiskPlotPhase3.cpp
    src/plotdisk/DiskPlotter.h
    src/plotdisk/DiskPlotter.cpp
    src/plotdisk/DiskBufferQueue.cpp
    src/plotdisk/DiskBufferQueue.h
    src/plotdisk/BitBucketWriter.h

    
    src/plotmem/DbgHelper.cpp
    src/plotmem/FxSort.h
    src/plotmem/MemPhase1.h
    src/plotmem/MemPhase2.h
    src/plotmem/MemPhase3.h
    src/plotmem/MemPlotter.h
    src/plotmem/ParkWriter.h
    src/plotmem/DbgHelper.h
    src/plotmem/LPGen.h
    src/plotmem/MemPhase1.cpp
    src/plotmem/MemPhase2.cpp
    src/plotmem/MemPhase3.cpp
    src/plotmem/MemPhase4.cpp
    src/plotmem/MemPhase4.h
    src/plotmem/MemPlotter.cpp

    
    src/plotting/DTables.h
    src/plotting/GenSortKey.h
    src/plotting/PlotValidation.cpp
    src/plotting/TableWriter.cpp
    src/plotting/PlotTypes.h
    src/plotting/TableWriter.h
    src/plotting/WorkHeap.cpp
    src/plotting/CTables.h
    src/plotting/Compression.cpp
    src/plotting/Compression.h
    src/plotting/FSETableGenerator.cpp
    src/plotting/GlobalPlotConfig.h
    src/plotting/IPlotter.h
    src/plotting/PlotHeader.h
    src/plotting/PlotTools.cpp
    src/plotting/PlotTools.h
    src/plotting/PlotValidation.h
    src/plotting/PlotWriter.cpp
    src/plotting/PlotWriter.h
    src/plotting/Tables.h
    src/plotting/BufferChain.h
    src/plotting/BufferChain.cpp
    
    src/plotting/f1/F1Gen.h
    src/plotting/f1/F1Gen.cpp
    
    src/plotting/fx/PlotFx.inl
    
    src/plotting/matching/GroupScan.cpp
    src/plotting/matching/GroupScan.h
    src/plotting/WorkHeap.h

    src/threading/AutoResetSignal.h
    src/threading/Semaphore.cpp
    src/threading/Semaphore.h
    src/threading/Fence.cpp
    src/threading/Fence.h
    src/threading/GenJob.h
    src/threading/MTJob.h
    src/threading/MonoJob.h
    src/threading/Thread.h
    src/threading/ThreadPool.cpp
    src/threading/ThreadPool.h
    src/threading/AutoResetSignal.cpp

    # src/tools/FSETableGenerator.cpp
    src/tools/MemTester.cpp
    src/tools/IOTester.cpp
    src/tools/PlotComparer.cpp
    src/tools/PlotFile.cpp
    src/tools/PlotReader.cpp
    src/tools/PlotReader.h
    src/tools/PlotValidator.cpp
    src/tools/PlotChecker.cpp

    src/util/Array.h
    src/util/Array.inl
    src/util/BitField.h
    src/util/SPCQueue.h
    src/util/SPCQueue.inl

    src/util/jobs/MemJobs.h
    src/util/jobs/SortKeyJob.h
    src/util/BitView.h
    src/util/CliParser.cpp
    src/util/KeyTools.cpp
    src/util/KeyTools.h
    src/util/Log.h
    src/util/CliParser.h
    src/util/Log.cpp
    src/util/Span.h
    src/util/StackAllocator.h
    src/util/Util.cpp
    src/util/Util.h
    src/util/VirtualAllocator.h

    src/commands/Commands.h
    src/commands/CmdPlotCheck.cpp
    src/commands/CmdSimulator.cpp
    src/commands/CmdCheckCUDA.cpp

    src/harvesting/GreenReaper.cpp
    src/harvesting/GreenReaper.h
    src/harvesting/GreenReaperInternal.h
    src/harvesting/Thresher.h

    src/plotting/DiskQueue.h
    src/plotting/DiskQueue.cpp
    src/plotting/DiskBuffer.h
    src/plotting/DiskBuffer.cpp
    src/plotting/DiskBucketBuffer.h
    src/plotting/DiskBucketBuffer.cpp
    src/plotting/DiskBufferBase.h
    src/plotting/DiskBufferBase.cpp

    src/util/MPMCQueue.h
    src/util/CommandQueue.h
)

target_sources(bladebit_core PUBLIC ${src_bladebit})

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
