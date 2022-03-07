#pragma once
#include "DiskPlotInfo.h"

/*
template<TableId table>
struct FpGroupMatcher
{
    ThreadPool& _pool;
    struct Group
    {
        int64 index;
        int64 length;
    };


    //-----------------------------------------------------------
    // int64 ScanGroups(  const uint32* yBuffer, uint32 entryCount, uint32* groups, uint32 maxGroups, GroupInfo groupInfos[BB_MAX_JOBS] )
    int64 ScanGroups( const int64 entryCount, const int64 maxGroups, const uint64* yEntries )
    {
        AnonMTJob::Run( _pool, _threadCount, [=]( AnonMTJob* self ) {

            int64 count, offset, end;
            GetThreadOffsets( self, entryCount, count, offset, end );

            const int64 maxGroupsPerThread = maxGroups / _threadCount;

            // Find the start of the group
            uint64* y = yEntries + offset;
            uint64 yGroup = *y / kBC;
        });

        for( uint i = 1; i < threadCount; i++ )
        {
          
            job.yBuffer         = yBuffer;
            job.groupBoundaries = groups + maxGroupsPerThread * i;
            job.bucketIdx       = bucketIdx;
            job.maxGroups       = maxGroupsPerThread;
            job.groupCount      = 0;

            const uint32 idx           = entryCount / threadCount * i;
            const uint64 y             = bucket | yBuffer[idx];
            const uint64 curGroup      = y / kBC;
            const uint64 groupLocalIdx = y - curGroup * kBC;

            uint64 targetGroup;

            // If we are already at the start of a group, just use this index
            if( groupLocalIdx == 0 )
            {
                job.startIndex = idx;
            }
            else
            {
                // Choose if we should find the upper boundary or the lower boundary
                const uint64 remainder = kBC - groupLocalIdx;
                
                #if _DEBUG
                    bool foundBoundary = false;
                #endif
                if( remainder <= kBC / 2 )
                {
                    // Look for the upper boundary
                    for( uint32 j = idx+1; j < entryCount; j++ )
                    {
                        targetGroup = (bucket | yBuffer[j] ) / kBC;
                        if( targetGroup != curGroup )
                        {
                            #if _DEBUG
                                foundBoundary = true;
                            #endif
                            job.startIndex = j; break;
                        }   
                    }
                }
                else
                {
                    // Look for the lower boundary
                    for( uint32 j = idx-1; j >= 0; j-- )
                    {
                        targetGroup = ( bucket | yBuffer[j] ) / kBC;
                        if( targetGroup != curGroup )
                        {
                            #if _DEBUG
                                foundBoundary = true;
                            #endif
                            job.startIndex = j+1; break;
                        }  
                    }
                }

                #if _DEBUG
                    ASSERT( foundBoundary );
                #endif
            }

            auto& lastJob = jobs[i-1];
            ASSERT( job.startIndex > lastJob.startIndex );  // #TODO: This should not happen but there should
                                                            //        be a pre-check in the off chance that the thread count is really high.
                                                            //        Again, should not happen with the hard-coded thread limit,
                                                            //        but we can check if entryCount / threadCount <= kBC 


            // We add +1 so that the next group boundary is added to the list, and we can tell where the R group ends.
            lastJob.endIndex = job.startIndex + 1;

            ASSERT( ( bucket | yBuffer[job.startIndex-1] ) / kBC != 
                    ( bucket | yBuffer[job.startIndex] ) / kBC );

            job.groupBoundaries = groups + maxGroupsPerThread * i;
        }

        // Fill in missing data for the last job
        jobs[threadCount-1].endIndex = entryCount;

        // Run the scan job
        const double elapsed = jobs.Run( threadCount );
        Log::Verbose( "  Finished group scan in %.2lf seconds." );

        // Get the total group count
        uint groupCount = 0;

        for( uint i = 0; i < threadCount-1; i++ )
        {
            auto& job = jobs[i];

            // Add a trailing end index (but don't count it) so that we can test against it
            job.groupBoundaries[job.groupCount] = jobs[i+1].groupBoundaries[0];

            groupInfos[i].groupBoundaries = job.groupBoundaries;
            groupInfos[i].groupCount      = job.groupCount;
            groupInfos[i].startIndex      = job.startIndex;

            groupCount += job.groupCount;
        }
        
        // Let the last job know where its R group is
        auto& lastJob = jobs[threadCount-1];
        lastJob.groupBoundaries[lastJob.groupCount] = entryCount;

        groupInfos[threadCount-1].groupBoundaries = lastJob.groupBoundaries;
        groupInfos[threadCount-1].groupCount      = lastJob.groupCount;
        groupInfos[threadCount-1].startIndex      = lastJob.startIndex;

        // Log::Line( "  Found %u groups.", groupCount );

        return groupCount;
    }

    //-----------------------------------------------------------
    inline int64 GroupScan( const int64 entryCount, const uint64* yEntries, Group* outGroups, const int64 maxGroups )
    {
        

        // const uint32 maxGroups = this->maxGroups;

        // uint32* groupBoundaries = this->groupBoundaries;
        // uint32  groupCount      = 0;

        // const uint32* yBuffer = this->yBuffer;
        // const uint32  start   = this->startIndex;
        // const uint32  end     = this->endIndex;

        // const uint64  bucket  = ( (uint64)this->bucketIdx ) << 32;

        // uint64 lastGroup = ( bucket | yBuffer[start] ) / kBC;

        // for( uint32 i = start+1; i < end; i++ )
        // {
        //     const uint64 group = ( bucket | yBuffer[i] ) / kBC;

        //     if( group != lastGroup )
        //     {
        //         ASSERT( group > lastGroup );

        //         groupBoundaries[groupCount++] = i;
        //         lastGroup = group;

        //         if( groupCount == maxGroups )
        //         {
        //             ASSERT( 0 );    // We ought to always have enough space
        //                             // So this should be an error
        //             break;
        //         }
        //     }
        // }

        // this->groupCount = groupCount;
    }
    // void MatchGroups( const int64 entryCount, )
};
*/



