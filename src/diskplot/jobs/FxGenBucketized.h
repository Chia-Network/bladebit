#pragma once

#include "threading/AutoResetSignal.h"
#include "../DiskBufferQueue.h"
#include "plotshared/Tables.h"
#include "JobShared.h"
#include "plotshared/PlotTypes.h"

// struct FxBucket
// {
//     // IO buffers
//     uint32* yFront    ;
//     uint32* yBack     ;
//     uint32* sortKey   ;
//     uint64* metaAFront; 
//     uint64* metaBFront;
//     uint64* metaABack ;
//     uint64* metaBBack ;
//     Pairs   pairs     ;

//     // Back/front buffer fences
//     AutoResetSignal frontFence;
//     AutoResetSignal backFence ;

//     // Fixed bufers
//     uint32* groupBoundaries;
//     byte*   bucketId       ;
//     uint32* yTmp           ;
//     uint64* metaATmp       ;
//     uint64* metaBTmp       ;

//     // Current file ids
//     FileId  yFileId    ;
//     FileId  metaAFileId;
//     FileId  metaBFileId;

//     // Cross-bucket data (for groups that overlap between 2 bucket boundaries)
//     uint32* yCross    ;
//     uint64* metaACross;
//     uint64* metaBCross;

//     // Used for overflow entries when using Direct IO (entries that not align to file block boundaries)
//     // OverflowBuffer yOverflow;
//     // OverflowBuffer metaAOverflow;
//     // OverflowBuffer metaBOverflow;
// };

template<TableId table>
struct FxGenBucketized
{
    // void GenerateFxBucket(
    //     ThreadPool&      pool, 
    //     uint             threadCount,
    //     const size_t     chunkSize
    // );

    static void GenerateFxBucketizedToDisk(
        DiskBufferQueue& diskQueue,
        size_t           writeInterval,
        ThreadPool&      pool, 
        uint32           threadCount,

        const uint32     bucketIdx,
        const uint32     entryCount,
        Pairs            pairs,
        byte*            bucketIndices,

        const uint32*    yIn,
        const uint64*    metaAIn,
        const uint64*    metaBIn,

        uint32*          yTmp,
        uint64*          metaATmp,
        uint64*          metaBTmp,

        uint32           bucketCounts[BB_DP_BUCKET_COUNT]
    );

    static void GenerateFxBucketizedInMemory(
        ThreadPool&      pool, 
        uint32           threadCount,
        uint32           bucketIdx,
        const uint32     entryCount,
        Pairs            pairs,
        byte*            bucketIndices,

        const uint32*    yIn,
        const uint64*    metaAIn,
        const uint64*    metaBIn,

        uint32*          yTmp,
        uint32*          metaATmp,
        uint32*          metaBTmp,

        uint32*          yOut,
        uint32*          metaAOut,
        uint32*          metaBOut,

        uint32           bucketCounts[BB_DP_BUCKET_COUNT]
    );
};

// Instantiate a version for each table
template struct FxGenBucketized<TableId::Table1>;
template struct FxGenBucketized<TableId::Table2>;
template struct FxGenBucketized<TableId::Table3>;
template struct FxGenBucketized<TableId::Table4>;
template struct FxGenBucketized<TableId::Table5>;
template struct FxGenBucketized<TableId::Table6>;
template struct FxGenBucketized<TableId::Table7>;