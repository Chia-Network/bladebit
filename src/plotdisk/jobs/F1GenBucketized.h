#pragma once
#include "plotdisk/DiskPlotConfig.h"
#include "plotdisk/FileId.h"

class ThreadPool;
class DiskBufferQueue;
struct chacha8_ctx;

// Bucketized, multi-threaded F1 Generation
struct F1GenBucketized
{
    static size_t GetEntrySizeBits( const uint32 numBuckets, const uint32 k );

    static void GenerateF1Disk(
        const byte*      plotId,
        ThreadPool&      pool, 
        uint32           threadCount,
        DiskBufferQueue& diskQueue,
        uint32*          bucketCounts,
        const uint32     numBuckets,
        const FileId     fileId
    );    
};