#pragma once
#include "DiskPlotContext.h"
#include "plotshared/MTJob.h"
#include "DiskBufferQueue.h"

struct GenF1Job : MTJob<GenF1Job>
{
    const byte* key;
    
    uint32  blocksPerChunk; // Total number of blocks for each chunk
    uint32  chunkCount;     // 
    uint32  blockCount;     // Block that this particular thread will process from the total blocks per chunk
    uint32  x;

    byte*   buffer;

    uint32* counts;         // Each thread's entry count per bucket
    uint32* bucketCounts;   // Total counts per for all buckets. Used by the control thread
    uint32* buckets;        // Set by the control thread when writing entries to buckets
    uint32* xBuffer;        // Buffer for sort key. Also set by the control thread
    
    DiskBufferQueue* diskQueue;


    void Run() override;
};

class DiskPlotPhase1
{
public:
    DiskPlotPhase1( DiskPlotContext& cx );
    void Run();

private:
    void GenF1();

private:
    DiskPlotContext& _cx;
    DiskBufferQueue* _diskQueue;
};

