#pragma once
#include "DiskPlotContext.h"
#include "plotshared/MTJob.h"
#include "DiskBufferQueue.h"

#include "util/Log.h"

// Represents a double-buffered blob of data of the size of
// the block size of the device we're writing to * 2 (one for y one for x)
struct DoubleBuffer
{
    byte*           front;
    byte*           back;
    AutoResetSignal fence;

    inline DoubleBuffer()
    {
        // Has to be initially signaled, since the first swap doesn't need to wait.
        fence.Signal();
    }
    inline ~DoubleBuffer() {}

    inline void Flip()
    {
        fence.Wait();
        std::swap( front, back );
    }
};


struct Bucket
{
    uint32 index;
    uint32 entryCount;

    uint32* yInput;
    uint32* metaInput;

    uint32* yOutput;
    uint32* metaOutput;

    uint32* pairL;
    uint16* pairR;
};

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
    
    // To be used by the control thread:
    byte*   remaindersBuffer;

    DiskBufferQueue* diskQueue;

    void Run() override;

private:


    void SaveBlockRemainders( uint32* yBuffer, uint32* xBuffer, const uint32* bucketCounts, 
                              DoubleBuffer* remainderBuffers, uint32* remainderSizes );

    void WriteFinalBlockRemainders( DoubleBuffer* remainderBuffers, uint32* remainderSizes );
};

class DiskPlotPhase1
{
public:
    DiskPlotPhase1( DiskPlotContext& cx );
    void Run();

private:
    void GenF1();
    void ForwardPropagate();

    void ForwardPropagateBucket( uint bucketIdx );

private:
    DiskPlotContext& _cx;
    DiskBufferQueue* _diskQueue;

    uint32 _bucketCounts[BB_DP_BUCKET_COUNT];
};


struct MatchJob : MTJob<MatchJob>
{
    void Run() override;
};

