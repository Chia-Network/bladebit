#pragma once

#include "plotdisk/DiskPlotConfig.h"
#include "threading/MTJob.h"

// template<typename TJob>
// struct BucketJob : MTJob<TJob>
// {
//     const uint32*    counts;            // Each thread's entry count per bucket
//     uint32*          totalBucketCounts; // Total counts per for all buckets. Used by the control thread
    
//     // DiskBufferQueue* diskQueue;
//     // uint32           chunkCount;
    
//     void CalculatePrefixSum( const uint32 counts      [BB_DP_BUCKET_COUNT], 
//                              uint32       pfxSum      [BB_DP_BUCKET_COUNT],
//                              uint32       bucketCounts[BB_DP_BUCKET_COUNT] );
// };
