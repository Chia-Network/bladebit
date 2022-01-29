#pragma once

#include "plotshared/Tables.h"
#include "DiskPlotConfig.h"
#include "DiskBufferQueue.h"

class ThreadPool;
struct DiskPlotContext;

namespace Debug
{
    void ValidateYFileFromBuckets( FileId yFileId, ThreadPool& pool, DiskBufferQueue& queue, 
                                   TableId table, uint32 bucketCounts[BB_DP_BUCKET_COUNT] );

    void ValidateMetaFileFromBuckets( const uint64* metaA, const uint64* metaB,
                                      TableId table, uint32 entryCount, uint32 bucketIdx, 
                                      uint32 bucketCounts[BB_DP_BUCKET_COUNT] );


    void ValidateLookupIndex( TableId table, ThreadPool& pool, DiskBufferQueue& queue, const uint32 bucketCounts[BB_DP_BUCKET_COUNT] );

    void ValidateLinePoints( DiskPlotContext& context, TableId table, uint32 bucketCounts[BB_DPP3_LP_BUCKET_COUNT] );
}