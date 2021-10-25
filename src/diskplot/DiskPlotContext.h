#pragma once
#include "DiskPlotConfig.h"
#include "threading/ThreadPool.h"
#include "plotshared/MTJob.h"
#include "ChiaConsts.h"

struct DiskPlotContext
{
    const char*  tmpPath;             // Path in which to allocate temporary buffers

    size_t      bufferSizeBytes;      // Size in bytes of our work buffer
    byte*       workBuffer;           // Buffer allocated for in-memory work

    size_t      diskFlushSize;        // How many bytes to fill in order to flush the disk.
                                      // This divides work up in chunks of this size.
                                      // This is rounded down to entry sizes.

    uint        threadCount;          // How many threads to use for in-memory plot work
    uint        diskQueueThreadCount; // How many threads to use for the disk buffer writer/reader

    const byte*  plotId;
    const byte*  plotMemo;
    uint         plotMemoSize;

    
    ThreadPool* threadPool;
};

