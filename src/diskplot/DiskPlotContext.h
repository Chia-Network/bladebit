#pragma once
#include "DiskPlotConfig.h"
#include "threading/ThreadPool.h"
#include "plotshared/MTJob.h"
#include "ChiaConsts.h"

// Write intervals are expressed in bytes
struct DiskWriteInterval
{
    size_t matching;    // Write interval during matching
    size_t fxGen;       // Write interval during fx generation
};

struct DiskPlotContext
{
    const char*  tmpPath;       // Path in which to allocate temporary buffers

    size_t      heapSize;       // Size in bytes of our working heap. Some parts are preallocated.
    byte*       heapBuffer;     // Buffer allocated for in-memory work

    DiskWriteInterval writeIntervals[(uint)TableId::_Count];
    size_t      ioBufferSize;     // Largest write interval out of all the specified. 
    size_t      ioHeapSize;       // Allocation size for the IO heap
    byte*       ioHeap;           // Full allocation of the IO heap
    uint        ioBufferCount;    // How many single IO buffers can we allocate

    uint         threadCount;   // How many threads to use for in-memory plot work
    uint         ioThreadCount; // How many threads to use for the disk buffer writer/reader

    const byte*  plotId;
    const byte*  plotMemo;
    uint         plotMemoSize;

    
    ThreadPool* threadPool;
};

