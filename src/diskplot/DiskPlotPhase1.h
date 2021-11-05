#pragma once
#include "DiskPlotContext.h"
#include "plotshared/MTJob.h"
#include "DiskBufferQueue.h"
#include "util/Log.h"
#include "ChiaConsts.h"

struct Pairs
{
    uint32* left ;
    uint16* right;
};

struct GroupInfo
{
    // When scanning groups
    uint32* groupBoundaries;
    uint32  groupCount;
    uint32  startIndex;

    // When matching
    uint32  entryCount;
    Pairs   pairs;
};

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

struct OverflowBuffer
{
    void Init( void* bucketBuffers, const size_t fileBlockSize );

    uint32       sizes  [BB_DP_BUCKET_COUNT];
    DoubleBuffer buffers[BB_DP_BUCKET_COUNT];
};

class DiskPlotPhase1
{
    struct Bucket
    {
        byte*   fpBuffer; // Root allocation

        uint32* y0     ;
        uint32* y1     ;
        uint32* sortKey;
        uint64* metaA0 ; 
        uint64* metaA1 ;
        uint64* metaB0 ;
        uint64* metaB1 ;
        Pairs   pairs  ;
        uint32* groupBoundaries;

        byte*   bucketId;   // Used during fx gen

        uint32* yTmp;
        uint64* metaATmp;
        uint64* metaBTmp;

        FileId  yFileId;
        FileId  metaAFileId;
        FileId  metaBFileId;

        AutoResetSignal frontFence;
        AutoResetSignal backFence;

        // Used for overflows
        OverflowBuffer yOverflow;
        OverflowBuffer metaAOverflow;
        OverflowBuffer metaBOverflow;
    };

public:
    DiskPlotPhase1( DiskPlotContext& cx );
    void Run();

private:
    void GenF1();
    void ForwardPropagate();

    template<TableId tableId>
    void ForwardPropagateTable();

    void ForwardPropagateBucket( TableId table, uint bucketIdx );

    template<TableId TableId>
    void SortBucket( uint32* y, uint32* yTmp, uint64* meta, uint64* metaTmp );

    uint32 ScanGroups( uint bucketIdx, const uint32* yBuffer, uint32 entryCount, uint32* groups, uint32 maxGroups, GroupInfo groupInfos[BB_MAX_JOBS] );

    uint32 Match( uint bucketIdx, uint maxPairsPerThread, const uint32* yBuffer, GroupInfo groupInfos[BB_MAX_JOBS] );

    void GenFx( TableId tableId, uint bucketIndex, Pairs pairs, uint pairCount );
    
    template<TableId tableId>
    void GenFxForTable( uint bucketIdx, uint entryCount, const Pairs pairs, 
                        const uint32* yIn, uint32* yOut, byte* bucketIdOut,
                        const uint64* metaInA, const uint64* metaInB,
                        uint64* metaOutA, uint64* metaOutB );

private:
    DiskPlotContext& _cx;
    DiskBufferQueue* _diskQueue;

//     uint32 _bucketCounts[BB_DP_BUCKET_COUNT];
    uint32 _maxBucketCount;

    Bucket*       _bucket;
    DoubleBuffer* _bucketBuffers;
};

template<typename TJob>
struct BucketJob : MTJob<TJob>
{
    const uint32*    counts;            // Each thread's entry count per bucket
    uint32*          totalBucketCounts; // Total counts per for all buckets. Used by the control thread
    
    DiskBufferQueue* diskQueue;
    uint32           chunkCount;
    
    void CalculatePrefixSum( const uint32 counts[BB_DP_BUCKET_COUNT], 
                             uint32 pfxSum[BB_DP_BUCKET_COUNT],
                             uint32 bucketCounts[BB_DP_BUCKET_COUNT] );
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

struct ScanGroupJob : MTJob<ScanGroupJob>
{
    const uint* yBuffer;
    uint*       groupBoundaries;
    uint        bucketIdx;
    uint        startIndex;
    uint        endIndex;
    uint        maxGroups;
    uint        groupCount;

    void Run() override;
};

struct MatchJob : MTJob<MatchJob>
{
    const uint32* yBuffer;
    GroupInfo*    groupInfo;
    uint          bucketIdx;
    uint          maxPairCount;
    uint          pairCount;

    uint32* copyLDst;             // Final contiguous pair buffer
    uint16* copyRDst;

    void Run() override;
};

struct FxJob : BucketJob<FxJob>
{
    TableId         tableId;
    uint32          bucketIdx;
    uint32          entriesPerChunk;        // Entries per chunk across all threads
    uint32          entryCount;             // Entry count for each individual job
    uint32          chunkCount;
    uint32          trailingChunkEntries;   // If greater than 0,
                                            // then we have an extra last chunk
                                            // with less entries than entryCount

//     uint32          offset;
    Pairs           pairs;
    const uint32*   yIn;
    uint32*         yOut;
    byte*           bucketIdOut;
    const uint64*   metaInA;
    const uint64*   metaInB;
    uint64*         metaOutA;
    uint64*         metaOutB;

    OverflowBuffer* yOverflows;
    OverflowBuffer* metaAOverflows;
    OverflowBuffer* metaBOverflows;

private:
    void* _bucketY    ;     // Buffer that will be used to write buckets to disk.
    void* _bucketMetaA;     // These buffers are assigned by the control thread.
    void* _bucketMetaB;

public:
    void Run() override;

    template<TableId tableId>
    void RunForTable();

    template<TableId tableId, typename TMetaA, typename TMetaB>
    void SortToBucket( uint entryCount, const byte* bucketIndices,
                       const uint32* inY, const TMetaA* metaInA, const TMetaB* metaInB,
                       uint bucketCounts[BB_DP_BUCKET_COUNT] );

    template<typename T>
    void SaveBlockRemainders( FileId fileId, const uint32* bucketCounts, const T* buffer, 
                              uint32* remainderSizes, DoubleBuffer* remainderBuffers );
};

