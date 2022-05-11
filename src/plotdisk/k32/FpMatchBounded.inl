#pragma once
#include "plotdisk/DiskPlotContext.h"
#include "plotdisk/DiskPlotConfig.h"
#include "plotdisk/DiskBufferQueue.h"
#include "util/StackAllocator.h"


class FxMatcherBounded
{
    using Job = AnonPrefixSumJob<uint32>;

public:

    //-----------------------------------------------------------
    void Match( Job* self, const Span<uint32> yEntries, Span<Pair> pairs, const uint32 bucket )
    {
        const uint32 id = self->JobId();

        uint32 startIndex;
        const uint32 groupCount = ScanGroups( self, yEntries, bucket, startIndex );
        ASSERT( groupCount > 0 );

        const Span<uint32> groupBoundaries( _groupIndices[id], groupCount );

        uint32 matchCount = MatchGroups( self,
                                         startIndex,
                                         groupCount,
                                         groupBoundaries,
                                         yEntries,
                                         pairs,
                                         bucket,
                                         _maxMatches );

        _matchCounts[id] = matchCount;

        self->SyncThreads();
    }

private:

    //-----------------------------------------------------------
    uint32 ScanGroups( Job* self, const Span<uint32> yEntries, const uint64 bucket, uint32& outStartIndex )
    {
        const uint64 yMask = bucket << 32;
        const uint32 id    = self->JobId();

        int64 _, offset;
        GetThreadOffsets( self, (int64)yEntries.Length(), _, offset, _ );

        const uint32* start   = yEntries.Ptr();
        const uint32* entries = start + offset;

        // Find base start position
        uint64 curGroup = *entries / kBC;
        while( entries > start )
        {
            if( entries[-1] / kBC != curGroup )
                break;
            --entries;
        }

        outStartIndex = (uint32)(uintptr_t)(entries - start);

        _startPositions[id] = entries;
        self->SyncThreads();

        const uint32* end = self->IsLastThread() ? yEntries.Ptr() + yEntries.Length() : _startPositions[id+1];

        // Now scan for all groups
        uint32* groupIndices = _groupIndices[id];
        uint32  groupCount   = 0;
        while( ++entries < end )
        {
            const uint64 g = (yMask | (uint64)*entries) / kBC;
            if( g != curGroup )
            {
                ASSERT( groupCount < _maxMatches );
                groupIndices[groupCount++] = (uint32)(uintptr_t)(entries - start);

                ASSERT( g - curGroup > 1 || groupCount == 1 || groupIndices[groupCount-1] - groupIndices[groupCount-2] <= 350 );
                curGroup = g;
            }
        }

        self->SyncThreads();

        // Add the end location of the last R group
        if( self->IsLastThread() )
        {
            ASSERT( groupCount < _maxMatches );
            groupIndices[groupCount] = (uint32)yEntries.Length();
        }
        else
        {
            ASSERT( groupCount+1 < _maxMatches );
            groupIndices[groupCount++] = (uint32)(uintptr_t)(_startPositions[id+1] - start);
            groupIndices[groupCount  ] = _groupIndices[id+1][0];
        }

        return groupCount;
    }

    //-----------------------------------------------------------
    uint32 MatchGroups( Job* self,
                        const uint32 startIndex,
                        const uint32 groupCount,
                        const Span<uint32> groupBoundaries,
                        const Span<uint32> yEntries,
                        Span<Pair> pairs,
                        const uint32 bucket,
                        const uint64 maxPairs )
    {
        const uint64 bucketMask = ((uint64)bucket) << 32;

        uint32 pairCount = 0;

        uint8  rMapCounts [kBC];
        uint16 rMapIndices[kBC];

        uint64 groupLStart = startIndex;
        uint64 groupL      = (bucketMask | (uint64)yEntries[groupLStart]) / kBC;

        for( uint32 i = 0; i < groupCount; i++ )
        {
            const uint64 groupRStart = groupBoundaries[i];
            const uint64 groupR      = (bucketMask | (uint64)yEntries[groupRStart]) / kBC;
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
                    uint64 localRY = (bucketMask | (uint64)yEntries[iR]) - groupRRangeStart;
                    ASSERT( (bucketMask | (uint64)yEntries[iR]) / kBC == groupR );

                    if( rMapCounts[localRY] == 0 )
                        rMapIndices[localRY] = (uint16)( iR - groupRStart );

                    rMapCounts[localRY] ++;
                }

                // For each group L entry
                for( uint64 iL = groupLStart; iL < groupLEnd; iL++ )
                {
                    const uint64 yL     = bucketMask | (uint64)yEntries[iL];
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
    uint32           _maxMatches;
    const uint32*    _startPositions[BB_DP_MAX_JOBS];// = { 0 };
    uint32           _matchCounts   [BB_DP_MAX_JOBS];// = { 0 };
    uint32*          _groupIndices  [BB_DP_MAX_JOBS];
};
