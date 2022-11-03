#pragma once
#include "DiskPlotContext.h"
#include "DiskPlotInfo.h"
#include "threading/ThreadPool.h"

struct FpCrossBucketInfo
{
    uint32  groupCount[2];

    uint64  savedY   [BB_DP_CROSS_BUCKET_MAX_ENTRIES];
    Meta4   savedMeta[BB_DP_CROSS_BUCKET_MAX_ENTRIES];
    Pair    pair     [BB_DP_CROSS_BUCKET_MAX_ENTRIES];

    uint64* y              = nullptr;
    Meta4*  meta           = nullptr;
    uint64  matchCount     = 0;
    uint64  matchOffset[2] = { 0 }; // 0 = prev bucket, 1 = cur bucket
    uint32  bucket;
    uint32  maxBuckets;

    inline bool   IsLastBucket()  const { return bucket == maxBuckets-1; }
    inline bool   IsFirstBucket() const { return bucket == 0; }
    inline uint64 EntryCount()    const { return (uint64)groupCount[0] + groupCount[1]; }
};

struct FpGroupMatcher
{
    DiskPlotContext& _context;
    uint64           _maxMatches;
    const uint64*    _startPositions[BB_DP_MAX_JOBS];// = { 0 };
    uint64           _matchCounts   [BB_DP_MAX_JOBS];// = { 0 };
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
    template<typename TMeta>
    inline uint64 Match( const int64 entryCount, const uint64* yEntries, const TMeta* meta, FpCrossBucketInfo* crossBucketInfo )
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

            // Cross-bucket matching
            #if !BB_DP_DBG_UNBOUNDED_DISABLE_CROSS_BUCKET
                // Perform cross-bucket matches with the previous bucket
                if( self->IsControlThread() && !crossBucketInfo->IsFirstBucket() )
                    this->CrossBucketMatch<TMeta>( *crossBucketInfo, yEntries, meta, groupIndices );
            #endif

            // Now perform matches
            _matchCounts[id] = MatchGroups( startIndex, groupCount, groupIndices, yEntries, _pairs[id], _maxMatches, id );

            // Copy to contiguous pair buffer
            self->SyncThreads();

            size_t copyOffset = 0;

            uint64* allMatches = _matchCounts;
            for( uint32 i = 0; i < id; i++ )
                copyOffset += allMatches[i];

            memcpy( _outPairs + copyOffset, _pairs[id], sizeof( Pair ) * _matchCounts[id] );

            #if !BB_DP_DBG_UNBOUNDED_DISABLE_CROSS_BUCKET
                // Save the last 2 groups for cross-bucket matching
                if( self->IsLastThread()  )
                    SaveCrossBucketInfo( *crossBucketInfo, groupIndices + groupCount - 2, yEntries, meta );
            #endif
        });

        const uint64* allMatches = _matchCounts;
        uint64 matchCount = 0;
        for( uint32 i = 0; i < threadCount; i++ )
            matchCount += allMatches[i];

        return matchCount;
    }

    //-----------------------------------------------------------
    template<bool IdIsLGroupEnd = false>
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

            const uint64 groupLEnd   = IdIsLGroupEnd ? id : groupRStart;

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
                for( uint64 iL = groupLStart; iL < groupLEnd; iL++ )
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

private:
    //-----------------------------------------------------------
    template<typename TMeta>
    inline void CrossBucketMatch( FpCrossBucketInfo& info, const uint64* yEntries, const TMeta* meta, const uint64 curBucketIndices[2] )
    {
        const uint64 prevGroupsEntryCount = info.EntryCount();

        uint64 groupBoundaries[3] = { 
            prevGroupsEntryCount,
            curBucketIndices[0] + prevGroupsEntryCount,
            curBucketIndices[1] + prevGroupsEntryCount
        };

        // Copy to the area before our working buffer starts. It is reserved for cross-bucket entries
        info.y    = (uint64*)( yEntries - prevGroupsEntryCount );
        info.meta = (Meta4*)( meta - prevGroupsEntryCount );

        memcpy( info.y   , info.savedY   , sizeof( uint64 ) * prevGroupsEntryCount );
        memcpy( info.meta, info.savedMeta, sizeof( TMeta  ) * prevGroupsEntryCount );

        const uint64* yStart = info.y;
        uint64 matches = 0;

        uint32 lastGrpMatchIndx = 1;

        // If the first entry group is the same from the prev's bucket last group,
        // then we can perform matches with the penultimate bucket
        if( yEntries[0] / kBC == yEntries[-1] / kBC )
        {
            matches = MatchGroups<true>( 0, 1, groupBoundaries, yStart, info.pair, 
                                         BB_DP_CROSS_BUCKET_MAX_ENTRIES, (uint32)info.groupCount[0] );
        }
        else
        {
            // We have different groups at the bucket boundary, so update the boundaries for the next match accordingly
            lastGrpMatchIndx = 0;
        }

        matches += MatchGroups<true>( info.groupCount[0], 1, &groupBoundaries[lastGrpMatchIndx], yStart, &info.pair[matches], 
                                      BB_DP_CROSS_BUCKET_MAX_ENTRIES-matches, (uint32)prevGroupsEntryCount );

        info.matchCount = matches;
    }

    //-----------------------------------------------------------
    template<typename TMeta>
    inline void SaveCrossBucketInfo( FpCrossBucketInfo& info, const uint64 groupIndices[3], const uint64* y, const TMeta* meta )
    {
        info.matchOffset[0] = info.matchOffset[1];
        info.matchOffset[1] = groupIndices[0];

        if( info.IsLastBucket() )
            return;

        info.groupCount[0] = (uint32)( groupIndices[1] - groupIndices[0] );
        info.groupCount[1] = (uint32)( groupIndices[2] - groupIndices[1] );

        const size_t copyCount = info.groupCount[0] + info.groupCount[1];
        ASSERT( copyCount <= BB_DP_CROSS_BUCKET_MAX_ENTRIES );

        memcpy( info.savedY   , y    + groupIndices[0], copyCount * sizeof( uint64 ) );
        memcpy( info.savedMeta, meta + groupIndices[0], copyCount * sizeof( TMeta  ) );

        ASSERT( info.savedY[0] / kBC == info.savedY[info.groupCount[0]-1] / kBC );
        ASSERT( info.savedY[info.groupCount[0]] / kBC == info.savedY[info.groupCount[0]+info.groupCount[1]-1] / kBC );
    }
};


