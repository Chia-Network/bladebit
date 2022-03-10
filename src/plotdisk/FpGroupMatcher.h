#pragma once
#include "DiskPlotContext.h"
#include "DiskPlotInfo.h"
#include "threading/ThreadPool.h"

struct FpGroupMatcher
{
    DiskPlotContext& _context;
    uint64           _maxMatches;
    const uint64*    _startPositions[BB_DP_MAX_JOBS] = { 0 };
    uint64           _matchCounts   [BB_DP_MAX_JOBS] = { 0 };
    uint64*          _groupIndices  [BB_DP_MAX_JOBS];
    Pair*            _pairs         [BB_DP_MAX_JOBS];
    Pair*            _outPairs;

    //-----------------------------------------------------------
    inline FpGroupMatcher( DiskPlotContext& context, const uint64 maxEntries, 
                           uint64* groupBoundaries, Pair* pairs, Pair* outPairs )
        : _context( context )
        , _maxMatches( maxEntries / context.fpThreadCount )
        , _outPairs( outPairs )
    {
        for( uint32 i = 0; i < context.fpThreadCount; i++ )
        {
            _groupIndices[i] = groupBoundaries + _maxMatches * i;
            _pairs       [i] = pairs           + _maxMatches * i;
        }
    }

    //-----------------------------------------------------------
    inline uint64 Match( const int64 entryCount, const uint64* yEntries )
    {
        const uint32 threadCount = _context.fpThreadCount;

        AnonMTJob::Run( *_context.threadPool, threadCount, [=]( AnonMTJob* self ) {
            
            const uint32 id = self->_jobId;
            
            int64 _, offset;
            GetThreadOffsets( self, entryCount, _, offset, _ );

            const uint64* start   = yEntries;
            const uint64* entries = start + offset;

            // Find base start position
            uint64 curGroup = *entries / kBC;
            while( entries > start )
            {
                if( entries[-1] / kBC != curGroup )
                    break;
                --entries;
            }

            const uint64 startIndex = (uint64)(uintptr_t)(entries - start);

            _startPositions[id] = entries;
            self->SyncThreads();

            const uint64* end = self->IsLastThread() ? yEntries + entryCount : _startPositions[id+1];

            // Now scan for all groups
            uint64* groupIndices = _groupIndices[id];
            uint64  groupCount   = 0;
            while( ++entries < end )
            {
                const uint64 g = *entries / kBC;
                if( g != curGroup )
                {
                    ASSERT( groupCount < _maxMatches );
                    groupIndices[groupCount++] = (uint64)(uintptr_t)(entries - start);
                    
                    ASSERT( g - curGroup > 1 || groupCount == 1 || groupIndices[groupCount-1] - groupIndices[groupCount-2] <= 350 );
                    curGroup = g;
                }
            }

            self->SyncThreads();

            // Add the end location of the last R group
            if( self->IsLastThread() )
            {
                ASSERT( groupCount < _maxMatches );
                groupIndices[groupCount] = (uint64)entryCount;
            }
            else
            {
                ASSERT( groupCount+1 < _maxMatches );
                groupIndices[groupCount++] = (uint64)(uintptr_t)(_startPositions[id+1] - start);
                groupIndices[groupCount  ] = _groupIndices[id+1][0];
            }

            // Now perform matches
            _matchCounts[id] = MatchGroups( startIndex, groupCount, groupIndices, yEntries, _pairs[id], _maxMatches, id );

            // Copy to contiguous pair buffer
            self->SyncThreads();

            size_t copyOffset = 0;

            uint64* allMatches = _matchCounts;
            for( uint32 i = 0; i < id; i++ )
                copyOffset += allMatches[i];

            memcpy( _outPairs + copyOffset, _pairs[id], sizeof( Pair ) * _matchCounts[id] );
            
        });

        const uint64* allMatches = _matchCounts;
        uint64 matchCount = 0;
        for( uint32 i = 0; i < threadCount; i++ )
            matchCount += allMatches[i];

        return matchCount;
    }

    //-----------------------------------------------------------
    inline static uint64 MatchGroups( 
        const uint64 startIndex, const int64 groupCount, 
        const uint64* groupBoundaries, const uint64* yBuffer, 
        Pair* pairs, const uint64 maxPairs, const uint32 id = 0 )
    {
        uint64 pairCount = 0;

        uint8  rMapCounts [kBC];
        uint16 rMapIndices[kBC];

        uint64 groupLStart = startIndex;
        uint64 groupL      = yBuffer[groupLStart] / kBC;

        for( uint32 i = 0; i < groupCount; i++ )
        {
            const uint64 groupRStart = groupBoundaries[i];
            const uint64 groupR      = yBuffer[groupRStart] / kBC;

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
                    uint64 localRY = yBuffer[iR] - groupRRangeStart;
                    ASSERT( yBuffer[iR] / kBC == groupR );

                    if( rMapCounts[localRY] == 0 )
                        rMapIndices[localRY] = (uint16)( iR - groupRStart );

                    rMapCounts[localRY] ++;
                }

                // For each group L entry
                for( uint64 iL = groupLStart; iL < groupRStart; iL++ )
                {
                    const uint64 yL     = yBuffer[iL];
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
};

