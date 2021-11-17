#pragma once

#include "plotshared/Tables.h"
#include "DiskPlotConfig.h"

class DiskBufferQueue;
class ThreadPool;

namespace Debug
{
    void ValidateYFileFromBuckets( ThreadPool& pool, DiskBufferQueue& queue, TableId table, uint32 bucketCounts[BB_DP_BUCKET_COUNT] );
}