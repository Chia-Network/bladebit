#pragma once

#include "plotdisk/DiskPlotContext.h"

// struct UnpackMapJob : MTJob<UnpackMapJob>
// {
//     // uint32        bucketCount;
//     uint32        bucket;
//     uint32        entryCount;
//     const uint64* mapSrc;
//     uint32*       mapDst;

//     static void RunJob( ThreadPool& pool, const uint32 threadCount, 
//                         const uint32 bucket, const uint32 bucketCount,
//                         const uint32 entryCount, const uint64* mapSrc, uint32* mapDst );

//     void Run() override;
// };



struct UnpackMapJob : MTJob<UnpackMapJob>
{
    uint32        bucket;
    uint32        entryCount;
    const uint64* mapSrc;
    uint32*       mapDst;

    //-----------------------------------------------------------
    static void RunJob( ThreadPool& pool, const uint32 threadCount, const uint32 bucket,
                        const uint32 entryCount, const uint64* mapSrc, uint32* mapDst )
    {
        MTJobRunner<UnpackMapJob> jobs( pool );

        for( uint32 i = 0; i < threadCount; i++ )
        {
            auto& job = jobs[i];
            job.bucket     = bucket;
            job.entryCount = entryCount;
            job.mapSrc     = mapSrc;
            job.mapDst     = mapDst;
        }

        jobs.Run( threadCount );
    }

    //-----------------------------------------------------------
    void Run() override
    {
        const uint64 maxEntries         = 1ull << _K ;
        const uint32 fixedBucketLength  = (uint32)( maxEntries / BB_DP_BUCKET_COUNT );
        const uint32 bucketOffset       = fixedBucketLength * this->bucket;


        const uint32 threadCount = this->JobCount();
        uint32 entriesPerThread = this->entryCount / threadCount;

        const uint32 offset = entriesPerThread * this->JobId();

        if( this->IsLastThread() )
            entriesPerThread += this->entryCount - entriesPerThread * threadCount;

        const uint64* mapSrc = this->mapSrc + offset;
        uint32*       mapDst = this->mapDst;

        // Unpack with the bucket id
        for( uint32 i = 0; i < entriesPerThread; i++ )
        {
            const uint64 m   = mapSrc[i];
            const uint32 idx = (uint32)m - bucketOffset;
            
            ASSERT( idx >> 26 == bucket );
            ASSERT( idx < this->entryCount );

            mapDst[idx] = (uint32)(m >> 32);
        }
    }
};
