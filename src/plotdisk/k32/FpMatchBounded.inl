#pragma once
#include "plotdisk/DiskPlotContext.h"
#include "plotdisk/DiskPlotConfig.h"
#include "plotdisk/DiskBufferQueue.h"
#include "util/StackAllocator.h"


struct K32CrossBucketEntries
{
    struct Meta4{ uint32 m[4]; };

    uint32 length;
    uint32 y    [BB_DP_CROSS_BUCKET_MAX_ENTRIES];
    uint32 index[BB_DP_CROSS_BUCKET_MAX_ENTRIES];
    Meta4  meta [BB_DP_CROSS_BUCKET_MAX_ENTRIES];

    inline Span<uint32> CopyYAndExtend( const Span<uint32> dst ) const
    {
        if( length == 0 )
            return dst;
        bbmemcpy_t<uint32>( dst.Ptr() + dst.Length(), y, length );
        return Span<uint32>( dst.Ptr(), dst.Length() + length );
    }

    inline Span<uint32> CopyIndexAndExtend( const Span<uint32> dst ) const
    {
        if( length == 0 )
            return dst;
        bbmemcpy_t<uint32>( dst.Ptr() + dst.Length(), index, length );
        return Span<uint32>( dst.Ptr(), dst.Length() + length );
    }

    template<typename TMeta>
    inline Span<TMeta> CopyMetaAndExtend( const Span<TMeta> dst ) const
    {
        if( length == 0 )
            return dst;
        bbmemcpy_t<TMeta>( dst.Ptr() + dst.Length(), (TMeta*)meta, length );
        return Span<TMeta>( dst.Ptr(), dst.Length() + length );
    }
};


struct K32BoundedFpCrossBucketInfo
{
    struct Meta4{ uint32 m[4]; };

    uint32  groupCount[2];  // Groups sizes for the prev bucket's last 2 groups, and
                            // the current bucket's first 2 groups

    uint32  savedY   [BB_DP_CROSS_BUCKET_MAX_ENTRIES];
    Meta4   savedMeta[BB_DP_CROSS_BUCKET_MAX_ENTRIES];
    Pair    pair     [BB_DP_CROSS_BUCKET_MAX_ENTRIES];

    uint32  matchCount          = 0;
    uint32  matchOffset         = 0;
    uint32  curBucketMetaLength = 0;   // Number of meta entries needed to be kept from the current bucket
                                       // (Length of first 2 groups of current bucket)


    inline uint32 PrevBucketEntryCount() const { return groupCount[0] + groupCount[1]; }

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

        Span<uint32> groups = ScanGroups( self, bucket, yEntries, groupsBuffer, startIndex );
        ASSERT( groups.Length() > 0 );
        
        _groupBuffers[id] = groups;

        #if BB_DP_FP_MATCH_X_BUCKET
            if( self->IsLastThread() )
            {
                // Save last 2 group's data for this bucket (to be used during the next bucket)
                if( bucket < _numBuckets )
                    SaveCrossBucketGroups( self, bucket, groups.Slice( groups.Length()-3 ), yEntries );
            }

            if( bucket > 0 )
            {
                // Perform cross-bucket matching for the previous bucket, with the first 2 groups of this bucket
                if( self->BeginLockBlock() )
                {
                    auto& info = GetCrossBucketInfo( bucket-1 );
                    Span<Pair> pairs( info.pair, BB_DP_CROSS_BUCKET_MAX_ENTRIES );
                    
                    CrossBucketMatch( self, bucket-1, yEntries, groups );
                }
                self->EndLockBlock();
            }
        #endif // BB_DP_FP_MATCH_X_BUCKET

        // const uint32 startIndex = (uint32)(uintptr_t)(_startPositions[id] - yEntries.Ptr());

        const uint32 matchCount = MatchGroups( self, bucket, bucket, startIndex, _groupBuffers[id],
                                               yEntries, pairs, pairs.Length() );

        return pairs.Slice( 0, matchCount );

    }
    
    //-----------------------------------------------------------
    // Span<Pair> Match( Job* self, 
    //     const uint32       bucket, 
    //     const Span<uint32> yEntries, 
    //           Span<Pair>   pairs )
    // {
    //     const uint32 id = self->JobId();

    //     // _groupBuffers[id] = groupsBuffer;

    //     // uint32 startIndex;
    //     // _groupBuffers[id] = ScanGroups( self, bucket, yEntries, groupsBuffer, startIndex );
    //     // ASSERT( _groupBuffers[id].Length() > 0 );

    //     if( self->IsLastThread() && bucket > 0 )
    //     {
    //         // Perform cross bucket matches on previous bucket, now that we have the new groups
    //         // #TODO: This
    //     }

    //     const uint32 startIndex = (uint32)(uintptr_t)(_startPositions[id] - yEntries.Ptr());

    //     const uint32 matchCount = MatchGroups( self,
    //                                            bucket,
    //                                            bucket,
    //                                            startIndex,
    //                                            _groupBuffers[id],
    //                                            yEntries,
    //                                            pairs,
    //                                            pairs.Length() );

    //     // Save cross bucket data
    //     // if( self->IsLastThread() && bucket < _numBuckets )
    //     // {
    //     //     SaveCrossBucketGroups( self, bucket, groups.Slice( groups.Length()-3 ), yEntries );
    //     // }

    //     return pairs.Slice( 0, matchCount );
    // }


    //-----------------------------------------------------------
    template<typename TMeta>
    inline void SaveCrossBucketMeta( const uint32 bucket, const Span<TMeta> meta )
    {
        auto& info = GetCrossBucketInfo( bucket );

        if( bucket > 0 )
        {
            auto& prevInfo = GetCrossBucketInfo( bucket - 1 );

            const uint32 prevEntryCount = prevInfo.PrevBucketEntryCount();
            ASSERT( prevEntryCount + prevInfo.curBucketMetaLength <= BB_DP_CROSS_BUCKET_MAX_ENTRIES );

            // Copy current metadata for previous bucket fx
            bbmemcpy_t<TMeta>( (TMeta*)prevInfo.savedMeta + prevEntryCount, meta.Ptr(), prevInfo.curBucketMetaLength  );
        }

        if( bucket == _numBuckets -1 )
            return;
        
        const size_t copyCount = info.PrevBucketEntryCount();
        ASSERT( copyCount <= BB_DP_CROSS_BUCKET_MAX_ENTRIES );
        ASSERT( meta.Length() - info.matchOffset >= copyCount );

        bbmemcpy_t<TMeta>( (TMeta*)info.savedMeta, meta.Ptr() + info.matchOffset, copyCount );
    }

    //-----------------------------------------------------------
    inline uint32 CrossBucketMatchCount( const uint32 bucket ) const
    {
        ASSERT( bucket > 0 );
        return const_cast<FxMatcherBounded*>( this )->GetCrossBucketInfo( bucket ).matchCount;
    }

    //-----------------------------------------------------------
    const Span<uint32> ScanGroups( 
        Job*               self, 
        const uint64       bucket, 
        const Span<uint32> yEntries, 
        Span<uint32>       groupBuffer,
        uint32&            outStartIndex )
    {
        const uint32 k          = 32;
        const uint32 bucketBits = bblog2( _numBuckets );
        const uint32 yBits      = k + kExtraBits - bucketBits;
        const uint64 yMask      = ((uint64)bucket) << yBits;

        const uint32 id         = self->JobId();

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
        const uint32 maxGroups    = (uint32)groupBuffer.Length();
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
    template<bool overrideLGroupEnd = false>
    uint32 MatchGroups( Job* self,
                  const uint32       lBucket,
                  const uint32       rBucket,
                  const uint32       startIndex,
                  const Span<uint32> groupBoundaries,
                  const Span<uint32> yEntries,
                        Span<Pair>   pairs,
                  const uint64       maxPairs,
                  const uint32       groupLEndOverride = 0 )
    {
        const uint32 k          = 32;
        const uint32 bucketBits = bblog2( _numBuckets );
        const uint32 yBits      = k + kExtraBits - bucketBits;

        const uint64 lGroupMask = ((uint64)lBucket) << yBits;
        const uint64 rGroupMask = ((uint64)rBucket) << yBits;

        const uint32 groupCount = (uint32)(groupBoundaries.Length() - 1); // Ignore the extra ghost group

        uint32 pairCount = 0;

        uint8  rMapCounts [kBC];
        uint16 rMapIndices[kBC];

        uint64 groupLStart = startIndex;
        uint64 groupL      = (lGroupMask | (uint64)yEntries[groupLStart]) / kBC;

        for( uint32 i = 0; i < groupCount; i++ )
        {
            const uint64 groupRStart = groupBoundaries[i];
            const uint64 groupR      = (rGroupMask | (uint64)yEntries[groupRStart]) / kBC;
                  uint64 groupLEnd   = groupRStart;

            if constexpr ( overrideLGroupEnd )
                groupLEnd = groupLEndOverride;

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
                    ASSERT( (rGroupMask | (uint64)yEntries[iR]) / kBC == groupR );

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

    //-----------------------------------------------------------
    inline K32BoundedFpCrossBucketInfo& GetCrossBucketInfo( const uint32 bucket )
    {
        return _xBucketInfo[bucket & 1]; // bucket % 2
    }

private:    
    //-----------------------------------------------------------
    uint32 CrossBucketMatch( Job* self, 
                       const uint32       bucket, 
                       const Span<uint32> curYEntries, 
                             Span<uint32> curBucketGroupBoundaries )
    {
        ASSERT( self->JobId() == 0 );
        ASSERT( curBucketGroupBoundaries.Length() > 2 );

        auto& info = GetCrossBucketInfo( bucket );
        info.curBucketMetaLength = curBucketGroupBoundaries[1];

        Span<Pair> pairs( info.pair, BB_DP_CROSS_BUCKET_MAX_ENTRIES );

        const uint32 prevBucketLength = info.groupCount[0] + info.groupCount[1];
        
        // Grab the first 2 groups from this bucket
        const uint32 _groupBoundaries[3] = {
            prevBucketLength,
            curBucketGroupBoundaries[0] + prevBucketLength,
            curBucketGroupBoundaries[1] + prevBucketLength
        };

        const Span<uint32> groupBoundaries( (uint32*)_groupBoundaries, 3 );

        const uint32 curBucketLength = curBucketGroupBoundaries[1];
        ASSERT( curBucketLength <= BB_DP_CROSS_BUCKET_MAX_ENTRIES - prevBucketLength );

        // Copy y from the 2 first groups of the current bucket to our existing y buffer
        // Buffer y now looks like this:
        // [prev_bcuket_penultimate_group][prev_bucket_last_group][cur_bucket_first_group][cur_bucket_second_group]
        bbmemcpy_t( info.savedY + prevBucketLength, curYEntries.Ptr(), curBucketLength );
        Span<uint32> yEntries( info.savedY, prevBucketLength + curBucketLength );
        
        // Do matches
        const uint32 k          = 32;
        const uint32 bucketBits = bblog2( _numBuckets );
        const uint32 yBits      = k + kExtraBits - bucketBits;

        const uint64 lGroupMask = ((uint64)bucket)   << yBits;
        const uint64 rGroupMask = ((uint64)bucket+1) << yBits;
        
        uint32 lastGrpMatchIndex = 1;
        uint32 matchCount        = 0;

        // If the cur bucket's first group is the same from the prev's bucket last group,
        // then we can perform matches against the penultimate bucket
        const uint64 lastGroup  = ( lGroupMask | yEntries[info.groupCount[0]] ) / kBC;
        const uint64 firstGroup = ( rGroupMask | yEntries[prevBucketLength] ) / kBC;
        
        if( lastGroup == firstGroup )
        {
            matchCount = MatchGroups<true>( self, bucket, bucket+1, 0, groupBoundaries, yEntries, pairs, pairs.Length(), info.groupCount[0] );
        }
        else
        {
            ASSERT( firstGroup > lastGroup );
            ASSERT( firstGroup - lastGroup <= 4 );

            // We have different groups at the bucket boundary, so update the boundaries for the next match accordingly
            lastGrpMatchIndex = 0;
        }   

        auto remPairs = pairs.Slice( matchCount );
        ASSERT( remPairs.Length() > 0 );

        matchCount += MatchGroups<true>( self, bucket, bucket+1, info.groupCount[0], groupBoundaries.Slice(lastGrpMatchIndex), yEntries, remPairs, remPairs.Length(), prevBucketLength );

        info.matchCount = matchCount;

        // Update offset on the pairs
        // for( uint32 i = 0; i < matchCount; i++ )
        // {
        //     info.pair[i].left  += info.matchOffset;
        //     info.pair[i].right += info.matchOffset;
        // }

        return matchCount;
    }

    //-----------------------------------------------------------
    inline void SaveCrossBucketGroups( Job* self, 
        const uint32                 bucket,
        const Span<uint32>           groupIndices,  // Last 3 group indices
        const Span<uint32>           yEntries )
    {
        auto& info = GetCrossBucketInfo( bucket );
        
        // GetCrossBucketInfo( bucket - 1 ).matchOffset = info.matchOffset;
        info.matchOffset = groupIndices[0];

        if( bucket == _numBuckets-1 )
            return;

        info.groupCount[0] = groupIndices[1] - groupIndices[0];
        info.groupCount[1] = groupIndices[2] - groupIndices[1];
        
        const size_t copyCount = info.groupCount[0] + info.groupCount[1];
        ASSERT( copyCount < BB_DP_CROSS_BUCKET_MAX_ENTRIES );

        memcpy( info.savedY, yEntries.Ptr() + info.matchOffset, copyCount * sizeof( uint32 ) );
        // memcpy( info.savedMeta, meta     + groupIndices[0], copyCount * sizeof( TMeta  ) );

        // #TODO: Re-enable asserts w/ bucket mask
        // ASSERT( info.savedY[0] / kBC == info.savedY[info.groupCount[0]-1] / kBC );
        // ASSERT( info.savedY[info.groupCount[0]] / kBC == info.savedY[info.groupCount[0]+info.groupCount[1]-1] / kBC );
    }


// private:
public:
    const uint32*               _startPositions[BB_DP_MAX_JOBS];// = { 0 };
    Span<uint32>                _groupBuffers  [BB_DP_MAX_JOBS];
    // uint32                      _startIndices  [BB_DP_MAX_JOBS];
    K32BoundedFpCrossBucketInfo _xBucketInfo[2] = {};
    uint32                      _bucket         = 0;
    uint32                      _numBuckets;
};
