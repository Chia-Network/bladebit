#pragma once
#include "plotdisk/DiskPlotContext.h"
#include "plotdisk/DiskPlotConfig.h"
#include "plotdisk/DiskBufferQueue.h"
#include "util/StackAllocator.h"


struct K32BoundedFpCrossBucketInfo
{
    struct Meta4{ uint32 m[4]; };

    uint32  groupCount[2];  // Groups sizes for the bucket's last 2 groups

    uint32  savedY   [BB_DP_CROSS_BUCKET_MAX_ENTRIES];
    Meta4   savedMeta[BB_DP_CROSS_BUCKET_MAX_ENTRIES];
    Pair    pair     [BB_DP_CROSS_BUCKET_MAX_ENTRIES];

    uint32* y           = nullptr;
    Meta4*  meta        = nullptr;
    uint32  matchCount  = 0;
    uint32  matchOffset = 0;

    inline uint64 EntryCount() const { return (uint64)groupCount[0] + groupCount[1]; }
};



class FxMatcherBounded
{
    using Job = AnonPrefixSumJob<uint32>;

public:
    //-----------------------------------------------------------
    FxMatcherBounded( const uint32 numBuckets )
        : _numBuckets( numBuckets ) 
    {}
    
    //-----------------------------------------------------------
    Span<Pair> Match( Job* self, 
        const uint32       bucket, 
        const Span<uint32> yEntries, 
              Span<uint32> groupsBuffer,
              Span<Pair>   pairs )
    {
        const uint32 id = self->JobId();

        _groupBuffers[id] = groupsBuffer;

        uint32 startIndex;
        const Span<uint32> groups = ScanGroups( self, bucket, yEntries, groupsBuffer, startIndex );
        ASSERT( groups.Length() );

        if( self->IsLastThread() && bucket > 0 )
        {
            // Perform cross bucket matches on previous bucket, now that we have the new groups
            // #TODO: This
        }

        const uint32 matchCount = MatchGroups( self,
                                               bucket,
                                               bucket,
                                               startIndex,
                                               groups,
                                               yEntries,
                                               pairs,
                                               pairs.Length() );

        // Save cross bucket matches
        if( self->IsLastThread() && bucket < _numBuckets )
        {
            SaveCrossBucketGroups( self, bucket, groups.Slice( groups.Length()-3 ), yEntries );
        }

        return pairs.Slice( 0, matchCount );
    }


    //-----------------------------------------------------------
    template<typename TMeta>
    inline void SaveCrossBucketMeta( const Span<TMeta> meta )
    {
        auto& info = _xBucketInfo;

        const size_t copyCount = info.groupCount[0] + info.groupCount[1];
        ASSERT( copyCount <= BB_DP_CROSS_BUCKET_MAX_ENTRIES );

        memcpy( info.savedY, yEntries.Ptr() + groupIndices[0], copyCount * sizeof( uint32 ) );
    }

    //-----------------------------------------------------------
    inline uint32 CrossBucketMatchCount() const
    {
        return _crossBucketMatchCount;
    }

    //-----------------------------------------------------------
    const Span<uint32> ScanGroups( 
        Job*               self, 
        const uint64       bucket, 
        const Span<uint32> yEntries, 
        Span<uint32>       groupBuffer,
        uint32&            outStartIndex )
    {
        const uint64 yMask = bucket << 32;
        const uint32 id    = self->JobId();

        int64 _, offset;
        GetThreadOffsets( self, (int64)yEntries.Length(), _, offset, _ );

        const uint32* start   = yEntries.Ptr();
        const uint32* entries = start + offset;

        // Find base start position
        uint64 curGroup = (yMask | (uint64)*entries) / kBC;
        while( entries > start )
        {
            if( ( yMask | entries[-1] ) / kBC != curGroup )
                break;
            --entries;
        }

        outStartIndex = (uint32)(uintptr_t)(entries - start);

        _startPositions[id] = entries;
        self->SyncThreads();

        const uint32* end = self->IsLastThread() ? yEntries.Ptr() + yEntries.Length() : _startPositions[id+1];

        // Now scan for all groups
        const uint32 maxGroups    = groupBuffer.Length();
        Span<uint32> groupIndices = groupBuffer;
        uint32       groupCount   = 0;
        while( ++entries < end )
        {
            const uint64 g = (yMask | (uint64)*entries) / kBC;
            if( g != curGroup )
            {
                ASSERT( groupCount < maxGroups );
                groupIndices[groupCount++] = (uint32)(uintptr_t)(entries - start);

                ASSERT( g - curGroup > 1 || groupCount == 1 || groupIndices[groupCount-1] - groupIndices[groupCount-2] <= 350 );
                curGroup = g;
            }
        }

        self->SyncThreads();

        // Add the end location of the last R group
        if( self->IsLastThread() )
        {
            ASSERT( groupCount < maxGroups );
            groupIndices[groupCount] = (uint32)yEntries.Length();
        }
        else
        {
            ASSERT( groupCount+1 < maxGroups );
            groupIndices[groupCount++] = (uint32)(uintptr_t)(_startPositions[id+1] - start);
            groupIndices[groupCount  ] = _groupBuffers[id+1][0];
        }

        return groupIndices.Slice( 0, groupCount + 1 ); // There's always an extra 'ghost' group used to get the end position of the last R group
    }

    //-----------------------------------------------------------
    uint32 MatchGroups( Job* self,
                        const uint32       lBucket,
                        const uint32       rBucket,
                        const uint32       startIndex,
                        const Span<uint32> groupBoundaries,
                        const Span<uint32> yEntries,
                        Span<Pair>         pairs,
                        const uint64       maxPairs )
    {
        const uint64 lGroupMask = ((uint64)lBucket) << 32;
        const uint64 rGroupMask = ((uint64)rBucket) << 32;

        const uint32 groupCount = groupBoundaries.Length() - 1; // Ignore the extra ghost group

        uint32 pairCount = 0;

        uint8  rMapCounts [kBC];
        uint16 rMapIndices[kBC];

        uint64 groupLStart = startIndex;
        uint64 groupL      = (lGroupMask | (uint64)yEntries[groupLStart]) / kBC;

        for( uint32 i = 0; i < groupCount; i++ )
        {
            const uint64 groupRStart = groupBoundaries[i];
            const uint64 groupR      = (rGroupMask | (uint64)yEntries[groupRStart]) / kBC;
            const uint64 groupLEnd   = groupRStart;

            if( groupR - groupL == 1 )
            {
                // Groups are adjacent, calculate matches
                const uint16 parity           = groupL & 1;
                const uint64 groupREnd        = groupBoundaries[i+1];

                const uint64 groupLRangeStart = groupL * kBC;
                const uint64 groupRRangeStart = groupR * kBC;

                ASSERT( groupREnd - groupRStart <= 350 );
                ASSERT( groupLRangeStart == groupRRangeStart - kBC );

                // Prepare a map of range kBC to store which indices from groupR are used
                // For now just iterate our bGroup to find the pairs

                // #NOTE: memset(0) works faster on average than keeping a separate a clearing buffer
                memset( rMapCounts, 0, sizeof( rMapCounts ) );

                for( uint64 iR = groupRStart; iR < groupREnd; iR++ )
                {
                    uint64 localRY = (rGroupMask | (uint64)yEntries[iR]) - groupRRangeStart;
                    ASSERT( (bucketMask | (uint64)yEntries[iR]) / kBC == groupR );

                    if( rMapCounts[localRY] == 0 )
                        rMapIndices[localRY] = (uint16)( iR - groupRStart );

                    rMapCounts[localRY] ++;
                }

                // For each group L entry
                for( uint64 iL = groupLStart; iL < groupLEnd; iL++ )
                {
                    const uint64 yL     = lGroupMask | (uint64)yEntries[iL];
                    const uint64 localL = yL - groupLRangeStart;

                    // Iterate kExtraBitsPow = 1 << kExtraBits = 1 << 6 == 64
                    // So iterate 64 times for each L entry.
                    for( int iK = 0; iK < kExtraBitsPow; iK++ )
                    {
                        const uint64 targetR = L_targets[parity][localL][iK];

                        for( uint j = 0; j < rMapCounts[targetR]; j++ )
                        {
                            const uint64 iR = groupRStart + rMapIndices[targetR] + j;
                            ASSERT( iL < iR );

                            // Add a new pair
                            Pair& pair = pairs[pairCount++];
                            pair.left  = (uint32)iL;
                            pair.right = (uint32)iR;

                            ASSERT( pairCount <= maxPairs );
                            if( pairCount == maxPairs )
                                return pairCount;
                        }
                    }
                }
            }
            // Else: Not an adjacent group, skip to next one.

            // Go to next group
            groupL      = groupR;
            groupLStart = groupRStart;
        }

        return pairCount;
    }

private:
    //-----------------------------------------------------------
    uint32 CrossBucketMatch( Job* self, 
        const uint32       bucket, 
        const Span<uint32> yEntries, 
              Span<uint32> groupBoundaries,
              Span<Pair>   pairs )
    {
        
    }

    //-----------------------------------------------------------
    inline void SaveCrossBucketGroups( Job* self, 
        const uint32                 bucket,
        const Span<uint32>           groupIndices,  // Last 3 group indices
        const Span<uint32>           yEntries )
    {
        auto& info = _xBucketInfo;
        
        info.matchOffsetPrev = info.matchOffsetCur;
        info.matchOffsetCur  = groupIndices[0];

        if( bucket == _numBuckets-1 )
            return;

        info.groupCount[0] = groupIndices[1] - groupIndices[0];
        info.groupCount[1] = groupIndices[2] - groupIndices[1];

        const size_t copyCount = info.groupCount[0] + info.groupCount[1];
        ASSERT( copyCount <= BB_DP_CROSS_BUCKET_MAX_ENTRIES );

        memcpy( info.savedY, yEntries.Ptr() + groupIndices[0], copyCount * sizeof( uint32 ) );
        // memcpy( info.savedMeta, meta     + groupIndices[0], copyCount * sizeof( TMeta  ) );

        // #TODO: Re-enable asserts w/ bucket mask
        // ASSERT( info.savedY[0] / kBC == info.savedY[info.groupCount[0]-1] / kBC );
        // ASSERT( info.savedY[info.groupCount[0]] / kBC == info.savedY[info.groupCount[0]+info.groupCount[1]-1] / kBC );
    }

// private:
public:
    const uint32*               _startPositions[BB_DP_MAX_JOBS];// = { 0 };
    Span<uint32>                _groupBuffers  [BB_DP_MAX_JOBS];
    K32BoundedFpCrossBucketInfo _xBucketInfo = {};
    uint32                      _bucket      = 0;
    uint32                      _numBuckets;
};
