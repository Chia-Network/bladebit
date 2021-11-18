#pragma once
#include "plotshared/MTJob.h"
#include "diskplot/DiskPlotConfig.h"

class ThreadPool;
class DiskBufferQueue;
struct chacha8_ctx;

// Bucketized, multi-threaded F1 Generation
struct F1GenBucketized
{
    // Generate the whole thing in memory
    static void GenerateF1Mem( 
        const byte* plotId,
        ThreadPool& pool, 
        uint        threadCount,
        byte*       blocks,         // Has to be big enough to hold a whole K set 
        uint32*     yBuckets,
        uint32*     xBuckets,
        uint32      bucketCounts[BB_DP_BUCKET_COUNT]
    );

    static void GenerateF1Disk(
        const byte*      plotId,
        ThreadPool&      pool, 
        uint             threadCount,
        DiskBufferQueue& diskQueue,
        const size_t     chunkSize,
        uint32           bucketCounts[BB_DP_BUCKET_COUNT]
    );    
};