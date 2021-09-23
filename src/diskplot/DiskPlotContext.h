#pragma once
#include "DiskPlotConfig.h"
#include "threading/ThreadPool.h"
#include "plotshared/MTJob.h"
#include "ChiaConsts.h"

struct DiskPlotContext
{
    size_t      ramSizeBytes;         // Size in bytes of our RAM buffer?
    uint        threadCount;          // How many threads to use when plotting
    uint        diskQueueThreadCount; 

    byte*       plotId;
    
    ThreadPool* threadPool;
};

