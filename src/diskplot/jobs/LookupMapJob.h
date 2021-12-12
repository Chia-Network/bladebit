#pragma once
#include "diskplot/DiskPlotContext.h"
#include "plotshared/MTJob.h"

template<uint BucketCount>
struct ReverseMapJob : MTJob<ReverseMapJob<BucketCount>>
{
    DiskBufferQueue* ioQueue;
    uint32           entryCount;
    uint32           tgtIndexOffset;
    const uint32*    sortKey;

    uint32*          originalPos;
    uint32*          targetPos;
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
    const uint32  bitShift   = 32 - 6;

    const uint32  entryCount    = this->entryCount;
    const uint32* srcIndices    = this->sortKey;
    const uint32* end           = sortKey + entryCount;

    uint32* originalPos         = this->originalPos;
    uint32* targetPos           = this->targetPos;

    uint32 counts      [BucketCount];
    uint32 pfxSum      [BucketCount];
    uint32 bucketCounts[BucketCount];

    memset( counts, 0, sizeof( counts ) );

    // Count how many entries we have per bucket
    {
        const uint32* index = srcIndices;

        while( index < end )
        {
            const uint bucket = (*index) >> bitShift;
            counts[bucket]++;
            index++;
        }
    }

    // Calculate prefix sum
    this->counts = counts;
    this->CalculatePrefixSum( counts, pfxSum, bucketCounts );

    // Now distribute to the respective buckets
    const uint32 tgtIndexOffset = this->tgtIndexOffset + this->JobId() * this->GetJob( 0 )->entryCount;

    uint32* dstOriginalPos = this->originalPos;
    uint32* dstTargetPos   = this->targetPos;

    for( uint32 i = 0; i < entryCount; i++ )
    {
        const uint32 originIndex = srcIndices[i];               // Original index of this entry before y sort
        const uint32 tgtIndex    = i + tgtIndexOffset;          // Index where this entry was placed after y sort
        const uint32 bucket      = originIndex >> bitShift;
        
        const uint32 dstIndex = --pfxSum[bucket];
        
        dstOriginalPos[dstIndex] = originIndex;
        dstTargetPos  [dstIndex] = tgtIndex;
    }
}