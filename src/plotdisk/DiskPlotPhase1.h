#pragma once
#include "DiskPlotContext.h"
#include "util/Log.h"
#include "ChiaConsts.h"


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

// Data covering the last 2 groups of a bucket.
// This is to match and calculate fx for groups
// that crossed the boundaries between 2 adjacent buckets.
struct AdjacentBucketInfo
{
    uint32* y    ;
    uint64* metaA;
    uint64* metaB;
    Pairs   pairs;
    uint32  groupCounts[2];
    uint32  groupOffset;        // Index in the previous table where the y
                                // of the groups copied here starts.
};


struct OverflowBuffer
{
    void Init( void* bucketBuffers, const size_t fileBlockSize );

    uint32       sizes  [BB_DP_BUCKET_COUNT];
    DoubleBuffer buffers[BB_DP_BUCKET_COUNT];
};


class DiskPlotPhase1
{   
    // Fence Ids used when performing forward propagation
    struct FPFenceId 
    {
        enum 
        {
            Start = 0,
            
            YLoaded,
            SortKeyLoaded,
            MetaALoaded,
            MetaBLoaded
        };
    };

    struct Bucket
    {
        uint32* y0     ;
        uint32* y1     ;
        uint32* sortKey0;
        uint32* sortKey1;
        uint64* map    ;
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

        uint32  tableEntryCount;    // Running entry count for the table being generated (accross all buckets)

        Fence   fence;              // Used by us, signalled by the IO thread
        Fence   ioFence;            // Signalled by us, used by the IO thread
        Fence   mapFence;           // Fence used for writing the lookup map generated from sort keys. With table 2, it is used for writing X back to disk.
        Fence   backPointersFence;  // Fence used for writing table backpointers

        // Used for overflows
        // OverflowBuffer yOverflow;
        // OverflowBuffer metaAOverflow;
        // OverflowBuffer metaBOverflow;

        AdjacentBucketInfo crossBucketInfo;
    };

public:
    DiskPlotPhase1( DiskPlotContext& cx );
    void Run();

private:
    void GenF1();

    void AllocateFPBuffers( Bucket& bucket );

    void SortAndCompressTable7();

    // Write the sort key as a reverse lookup map (map target (sorted) position to original position)
    void WriteReverseMap( TableId tableId, const uint32 bucketIdx, const uint32 count, 
                          const uint32* sortedLookupIndices, uint64* map, Fence* writeFence, bool releaseIndices );

    void ForwardPropagate();

    template<TableId tableId>
    void ForwardPropagateTable();

    template<TableId tableId>
    uint32 ForwardPropagateBucket( uint32 bucketIdx, Bucket& bucket, uint32 entryCount );

    uint32 MatchBucket( TableId table, uint32 bucketIdx, Bucket& bucket, uint32 entryCount, GroupInfo groupInfos[BB_MAX_JOBS] );

    void WritePendingBackPointers( const Pairs& pairs, TableId table, uint32 bucketIdx, uint32 entryCount );

    uint32 ScanGroups( uint bucketIdx, const uint32* yBuffer, uint32 entryCount, uint32* groups, uint32 maxGroups, GroupInfo groupInfos[BB_MAX_JOBS] );

    uint32 Match( uint bucketIdx, uint maxPairsPerThread, const uint32* yBuffer, GroupInfo groupInfos[BB_MAX_JOBS], Pairs dstPairs );

    // uint32 MatchAdjoiningBucketGroups( uint32* yTmp, uint32* curY, Pairs pairs, const uint32 prevGroupsCounts[2],
    //                                    uint32 curBucketLength, uint32 maxPairs, uint32 prevBucket, uint32 curBucket );

    template<TableId tableId>
    uint32 ProcessAdjoiningBuckets( 
        uint32 bucketIdx, Bucket& bucket, uint32 entryCount,
        const uint32* curY, const uint64* curMetaA, const uint64* curMetaB
    );

    template<TableId tableId, typename TMetaA, typename TMetaB>
    uint32 ProcessCrossBucketGroups(
        const uint32* prevBucketY,
        const TMetaA* prevBucketMetaA,
        const TMetaB* prevBucketMetaB,
        const uint32* curBucketY,
        const TMetaA* curBucketMetaA,
        const TMetaB* curBucketMetaB,
        uint32*       tmpY,
        TMetaA*       tmpMetaA,
        TMetaB*       tmpMetaB,
        uint32*       outY,
        uint64*       outMetaA,
        uint64*       outMetaB,
        uint32        prevBucketGroupCount,
        uint32        curBucketEntryCount,
        uint32        prevBucketIndex,
        uint32        curBucketIndex,
        Pairs         pairs,
        uint32        maxPairs,
        uint32        sortKeyOffset,
        uint32&       outCurGroupCount,
        uint32        pairsOffsetL,
        uint32        pairsOffsetR
    );

  
    void GetWriteFileIdsForBucket( TableId table, FileId& outYId, 
                                   FileId& outMetaAId, FileId& outMetaBId );



private:
    DiskPlotContext& _cx;
    DiskBufferQueue* _diskQueue;

//     uint32 _bucketCounts[BB_DP_BUCKET_COUNT];
    uint32 _maxBucketCount;

    Bucket*       _bucket;
    DoubleBuffer* _bucketBuffers;
    
    Duration      _readWaitTime     = Duration::zero();
    Duration      _writeWaitTime    = Duration::zero();
};


struct GenF1Job : MTJob<GenF1Job>
{
    const byte* key;
    
    uint32  blocksPerChunk;     // Total number of blocks for each chunk
    uint32  chunkCount;         // 
    uint32  blockCount;         // Block that this particular thread will process from the total blocks per chunk
    uint32  bucketBufferSize;
    uint32  trailingBlocks;     // Blocks not included in the last chunk
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

    // Final destination contiguous pair buffer
    uint32*       copyLDst;
    uint16*       copyRDst;

    // Fence ensures the copy destination buffer is not currently in-use
    Fence*        copyFence;

    void Run() override;
};
