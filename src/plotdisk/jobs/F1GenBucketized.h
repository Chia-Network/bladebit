#pragma once
#include "plotdisk/DiskPlotConfig.h"

class ThreadPool;
class DiskBufferQueue;
struct chacha8_ctx;

// Bucketized, multi-threaded F1 Generation
struct F1GenBucketized
{
    static void GenerateF1Disk(
        const byte*      plotId,
        ThreadPool&      pool, 
        uint32           threadCount,
        DiskBufferQueue& diskQueue,
        const size_t     chunkSize,
        uint32*          bucketCounts,
        const uint32     numBuckets
    );    
};