#pragma once

#include "algorithm/RadixSort.h"
#include "algorithm/YSort.h"
#include "threading/ThreadPool.h"
#include "ChiaConsts.h"
#include "PlotContext.h"

template<typename TMeta>
struct MapFxJob
{
    uint64        offset;
    uint64        length;
    const uint32* sortKey;
    const TMeta*  metaSrc;
    TMeta*        metaDst;
    const Pair*   pairSrc;
    Pair*         pairDst;
};

struct GenSortKeyJob
{
    uint64  offset;
    uint64  length;
    uint32* keyBuffer;
};

template<size_t MAX_JOBS>
void GenSortKey( ThreadPool& pool, uint64 length, uint32* keyBuffer );

template<typename TMeta>
void MapFxThread( MapFxJob<TMeta>* job );
void GenSortKeyThread( GenSortKeyJob* job );

//-----------------------------------------------------------
template<size_t MAX_JOBS>
inline void SortFx(
    ThreadPool&   pool,    uint64  length,  
    uint64*       yBuffer, uint64* yTmp,
    uint32*       sortKey, uint32* sortKeyTmp )
{
    // Generate a sort key
    GenSortKey<MAX_JOBS>( pool, length, sortKey );

    YSorter sorter( pool );
    sorter.Sort( length, yBuffer, yTmp, sortKey, sortKeyTmp );
}


//-----------------------------------------------------------
template<typename TMeta, size_t MAX_JOBS>
inline void MapFxWithSortKey(
    ThreadPool&   pool,    uint64  length,  
    const uint32* sortKey,
    const TMeta*  metaSrc, TMeta*  metaDst,
    const Pair*   pairSrc, Pair*   pairDst )
{
    // Sort metadata and pairs on y via the sort key
    const uint32 threadCount      = pool.ThreadCount();
    const uint64 entriesPerThread = length / threadCount;
    const uint64 trailingEntries  = length - ( entriesPerThread * threadCount );

    MapFxJob<TMeta> jobs[MAX_JOBS];

    for( uint32 i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];

        job.offset  = i * entriesPerThread;
        job.length  = entriesPerThread;
        job.sortKey = sortKey;
        job.metaSrc = metaSrc;
        job.metaDst = metaDst;
        job.pairSrc = pairSrc;
        job.pairDst = pairDst;
    }

    jobs[threadCount-1].length += trailingEntries;

    pool.RunJob( MapFxThread<TMeta>, jobs, threadCount );
}

//-----------------------------------------------------------
template<typename TMeta>
void MapFxThread( MapFxJob<TMeta>* job )
{
    const uint64 length   = job->length;
    const uint64 offset   = job->offset;

    const uint32* sortKey = job->sortKey + offset;

    // Map metadata
    const TMeta* metaSrc  = job->metaSrc;
    TMeta*       metaDst  = job->metaDst + offset;

    for( uint64 i = 0; i < length; i++ )
        metaDst[i] = metaSrc[sortKey[i]];

    // Map pairs
    const Pair*  pairSrc  = job->pairSrc;
    Pair*        pairDst  = job->pairDst + offset;

    for( uint64 i = 0; i < length; i++ )
        pairDst[i] = pairSrc[sortKey[i]];
}


//-----------------------------------------------------------
template<size_t MAX_JOBS>
inline void GenSortKey( 
    ThreadPool& pool,
    uint64 length,
    uint32* keyBuffer )
{
    GenSortKeyJob jobs[MAX_JOBS];
    
    const uint threadCount = pool.ThreadCount();
    ASSERT( MAX_JOBS >= threadCount );

    const uint64 entriesPerThread = length / threadCount;
    const uint64 tailingEntries   = length - ( entriesPerThread * threadCount );

    for( uint64 i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];

        job.length    = entriesPerThread;
        job.offset    = i * entriesPerThread;
        job.keyBuffer = keyBuffer;
    }

    jobs[threadCount-1].length += tailingEntries;

    pool.RunJob( GenSortKeyThread, jobs, threadCount );
}


//-----------------------------------------------------------
inline void GenSortKeyThread( GenSortKeyJob* job )
{   
    const uint64 offset = job->offset;
    uint64 i = offset;
    const uint64 end = i + job->length;

    uint32* buffer = job->keyBuffer;

    for( ; i < end; i++ )
        buffer[i] = (uint32)i;
}
