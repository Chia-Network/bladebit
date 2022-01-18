#include "StripAndSortMap.h"
#include "algorithm/RadixSort.h"

//-----------------------------------------------------------
void StripMapJob::RunJob( 
    ThreadPool&   pool, 
    uint32        threadCount, 
    uint32        entryCount,
    const uint64* inMap,
    uint32*       outKey,
    uint32*       outMap )
{
    MTJobRunner<StripMapJob> jobs( pool );

    const uint32 entriesPerThread = entryCount / threadCount;

    // uint32* outMap = (uint32*)_tmpMap;
    // uint32* key    = outMap + entryCount;

    for( uint32 i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];

        job.entryCount = entriesPerThread;
        job.inMap      = inMap  + entriesPerThread * i;
        job.outKey     = outKey + entriesPerThread * i;
        job.outMap     = outMap + entriesPerThread * i;
    }

    const uint32 trailingEntries = entryCount - entriesPerThread * threadCount;
    jobs[threadCount-1].entryCount += trailingEntries;

    jobs.Run( threadCount );

    // Now sort it
    uint32* tmpKey = (uint32*)inMap;
    uint32* tmpMap = tmpKey + entryCount;
    
    RadixSort256::SortWithKey<BB_MAX_JOBS>( 
        pool, outKey, tmpKey, outMap, tmpMap, entryCount );

}

//-----------------------------------------------------------
void StripMapJob::Run()
{
    const int64   entryCount = (int64)this->entryCount;
    const uint64* inMap      = this->inMap;
    uint32*       key        = this->outKey;
    uint32*       map        = this->outMap;

    for( int64 i = 0; i < entryCount; i++ )
    {
        const uint64 m = inMap[i];
        key[i] = (uint32)m;
        map[i] = (uint32)(m >> 32);
    }
}