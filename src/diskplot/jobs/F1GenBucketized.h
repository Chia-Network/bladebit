#pragma once
#include "plotshared/MTJob.h"
#include "diskplot/DiskPlotConfig.h"

class ThreadPool;
class DiskBufferQueue;
struct chacha8_ctx;

// Bucketized, multi-threaded F1 Generation
struct F1GenBucketized : MTJob<F1GenBucketized>
{
    const byte* key;
    
    uint32  blocksPerChunk;     // Total number of blocks for each chunk
    uint32  chunkCount;         // 
    uint32  blockCount;         // Block that this particular thread will process from the total blocks per chunk
    uint32  bucketBufferSize;
    uint32  trailingBlocks;     // Blocks not included in the last chunk
    uint32  x;
    byte*   buffer;

    uint32* counts;             // Each thread's entry count per bucket
    uint32* buckets;            // Set by the control thread when writing entries to buckets
    uint32* xBuffer;            // Buffer for sort key. Also set by the control thread

    uint32* totalBucketCounts;  // Total counts per for all buckets. Used by the control thread
    
    // To be used by the control thread:
    byte*   remaindersBuffer;

    DiskBufferQueue* diskQueue;

public:

    void GenerateF1( ThreadPool& pool, uint threadCount,
            DiskBufferQueue& diskQueue,
            const uint32 blockCount
    );

    void GenerateAndWriteToFile(
        ThreadPool& pool, uint threadCount,
        DiskBufferQueue& diskQueue
    );

    void Run() override;

private:
    template<bool WriteToDisk, bool SingleThreaded>
    void Generate(
        chacha8_ctx& chacha, 
        uint32  x,
        byte*   blocks,
        uint32  blockCount,
        uint32  entryCount,
        uint32* buckets,
        uint32* xBuffer,

        // For writing to disk variant
        size_t  fileBlockSize,
        uint32* sizes
     );

    void CalculateMultithreadedPredixSum( 
        uint32 counts[BB_DP_BUCKET_COUNT],
        uint32 pfxSum[BB_DP_BUCKET_COUNT],
        const size_t fileBlockSize
    );
};