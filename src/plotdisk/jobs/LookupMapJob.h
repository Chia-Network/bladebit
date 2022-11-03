#pragma once
#include "plotdisk/DiskPlotContext.h"
#include "threading/MTJob.h"
#include "threading/Fence.h"

template<uint BucketCount>
struct ReverseMapJob : MTJob<ReverseMapJob<BucketCount>>
{
    DiskBufferQueue* ioQueue;
    uint32           entryCount;
    uint32           sortedIndexOffset;     // Offset of the index at which each sorted source index is stored
    const uint32*    sortedSourceIndices;   // Read-only sorted final (absolute) position of the origin indices
    uint64*          mappedIndices;         // Write to position for the buckets

    uint32*          bucketCounts;

    // For internal use:
    const uint32*    counts;
    
    void Run() override;

    void CalculatePrefixSum( 
        uint32       counts      [BucketCount],
        uint32       pfxSum      [BucketCount],
        uint32       bucketCounts[BucketCount],
        const size_t fileBlockSize );
};

//-----------------------------------------------------------
template<uint BucketCount>
inline void ReverseMapJob<BucketCount>::Run()
{
    // #TODO: Determine bits needed bucket count statically.
    //        For now hard coded to 64 buckets.
    const uint32  bitShift = 32 - kExtraBits;

    const uint32  entryCount          = this->entryCount;
    const uint32* sortedOriginIndices = this->sortedSourceIndices;
    const uint32* end                 = sortedOriginIndices + entryCount;

    uint64* map                 = this->mappedIndices;

    uint32 counts[BucketCount];
    uint32 pfxSum[BucketCount];

    memset( counts, 0, sizeof( counts ) );

    // Count how many entries we have per bucket
    {
        const uint32* index = sortedOriginIndices;

        while( index < end )
        {
            const uint bucket = (*index) >> bitShift;
            counts[bucket]++;
            index++;
        }
    }

    // Calculate prefix sum
    // #TODO: Allow block-aligned prefix sum here
    this->CalculatePrefixSum( counts, pfxSum, this->bucketCounts, 0 );

    // Now distribute to the respective buckets
    const uint32 entriesPerThread  = this->GetJob( 0 ).entryCount;
    const uint32 sortedIndexOffset = this->sortedIndexOffset + this->JobId() * entriesPerThread;

    for( uint32 i = 0; i < entryCount; i++ )
    {
        const uint32 originIndex = sortedOriginIndices[i];      // Original index of this entry before y sort
        const uint64 sortedIndex = i + sortedIndexOffset;       // Index where this entry was placed after y sort
        const uint32 bucket      = (uint32)(originIndex >> bitShift);
        
        const uint32 dstIndex = --pfxSum[bucket];

        map[dstIndex] = ( sortedIndex << 32 ) | originIndex;
    }

    // Ensure all threads end at the same time (so that counts doesn't go out of scope) 
    this->SyncThreads();
}

// #TODO: Avoud code duplication here
//-----------------------------------------------------------
template<uint BucketCount>
inline void ReverseMapJob<BucketCount>::CalculatePrefixSum( 
    uint32       counts      [BucketCount],
    uint32       pfxSum      [BucketCount],
    uint32       bucketCounts[BucketCount],
    const size_t fileBlockSize )
{
    const uint32 jobId    = this->JobId();
    const uint32 jobCount = this->JobCount();

    // This holds the count of extra entries added per-bucket
    // to align each bucket starting address to disk block size.
    // Only used when fileBlockSize > 0
    uint32 entryPadding[BB_DP_BUCKET_COUNT];

    this->counts = counts;
    this->SyncThreads();

    // Add up all of the jobs counts
    memset( pfxSum, 0, sizeof( uint32 ) * BB_DP_BUCKET_COUNT );

    for( uint i = 0; i < jobCount; i++ )
    {
        const uint* tCounts = this->GetJob( i ).counts;

        for( uint j = 0; j < BB_DP_BUCKET_COUNT; j++ )
            pfxSum[j] += tCounts[j];
    }

    // If we're the control thread, retain the total bucket count
    if( this->IsControlThread() )
    {
        memcpy( bucketCounts, pfxSum, sizeof( uint32 ) * BB_DP_BUCKET_COUNT );
    }

    // Only do this if using Direct IO
    // We need to align our bucket totals to the  file block size boundary
    // so that each block buffer is properly aligned for direct io.
    if( fileBlockSize )
    {
        #if _DEBUG
            size_t bucketAddress = 0;
        #endif

        for( uint i = 0; i < BB_DP_BUCKET_COUNT-1; i++ )
        {
            const uint32 count = pfxSum[i];

            pfxSum[i]       = RoundUpToNextBoundary( count * sizeof( uint32 ), (int)fileBlockSize ) / sizeof( uint32 );
            entryPadding[i] = pfxSum[i] - count;

            #if _DEBUG
                bucketAddress += pfxSum[i] * sizeof( uint32 );
                ASSERT( bucketAddress / fileBlockSize * fileBlockSize == bucketAddress );
            #endif
        }

        #if _DEBUG
        // if( this->IsControlThread() )
        // {
        //     size_t totalSize = 0;
        //     for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
        //         totalSize += pfxSum[i];

        //     totalSize *= sizeof( uint32 );
        //     Log::Line( "Total Size: %llu", totalSize );
        // }   
        #endif
    }

    // Calculate the prefix sum
    for( uint i = 1; i < BB_DP_BUCKET_COUNT; i++ )
        pfxSum[i] += pfxSum[i-1];

    // Subtract the count from all threads after ours 
    // to get the correct prefix sum for this thread
    for( uint t = jobId+1; t < jobCount; t++ )
    {
        const uint* tCounts = this->GetJob( t ).counts;

        for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
            pfxSum[i] -= tCounts[i];
    }

    if( fileBlockSize )
    {
        // Now that we have the starting addresses of the buckets
        // at a block-aligned position, we need to substract
        // the padding that we added to align them, so that
        // the entries actually get writting to the starting
        // point of the address

        for( uint i = 0; i < BB_DP_BUCKET_COUNT-1; i++ )
            pfxSum[i] -= entryPadding[i];
    }
}

