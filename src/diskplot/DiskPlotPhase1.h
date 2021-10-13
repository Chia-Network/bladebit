#pragma once
#include "DiskPlotContext.h"
#include "plotshared/MTJob.h"
#include "DiskBufferQueue.h"

struct GenF1Job : MTJob<GenF1Job>
{
    const byte* key;
    
    byte*   buffer;
    uint32* buckets;        // Bucketized entries
    uint32* xBuffer;        // Buffer for sort key, same size as buffer
    uint32* counts;         // Each thread's entry count per bucket
    uint32* bucketCounts;   // Total counts per for all buckets. Used by the control thread
    uint32  blockCount;
    uint32  chunkCount;
    uint32  x;
    
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
    DiskBufferQueue  _diskQueue;
};

