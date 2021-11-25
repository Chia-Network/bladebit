#pragma once

#include "plotshared/Tables.h"
#include "DiskPlotConfig.h"
#include "DiskBufferQueue.h"

class ThreadPool;

namespace Debug
{
    void ValidateYFileFromBuckets( FileId yFileId,  ThreadPool& pool, DiskBufferQueue& queue, 
                                   TableId table, uint32 bucketCounts[BB_DP_BUCKET_COUNT] );
}