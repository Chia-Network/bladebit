#pragma once

#include "threading/ThreadPool.h"

struct SortKeyGen
{
    template<uint MAX_JOBS>
    static void Generate( ThreadPool& pool, uint64 length, uint32* keyBuffer );

//     template<uint MAX_JOBS>
//     static void Invert( ThreadPool& pool, uint64 length, const uint32* srcKeyBuffer, uint32* dstKeyBuffer );

    template<uint MAX_JOBS, typename T>
    static void Sort( ThreadPool& pool, int64 length, const uint32* keys, const T* src, T* dst );

    template<uint MAX_JOBS, typename T>
    static void Sort( ThreadPool& pool, const uint32 desiredThreadCount, int64 length, const uint32* keys, const T* src, T* dst );

private:
    struct GenJob
    {
        uint64  offset;
        uint64  length;
        uint32* keyBuffer;
    };

    template<typename T>
    struct SortJob
    {
        int64         length;
        int64         offset;
        const uint32* keys;
        const T*      src;
        T*            dst;
    };

    static void GenJobThread( GenJob* job );

    template<typename T>
    static void SortThread( SortJob<T>* job );
};

//-----------------------------------------------------------
template<uint MAX_JOBS>
inline void SortKeyGen::Generate( ThreadPool& pool, uint64 length, uint32* keyBuffer )
{
    ASSERT( pool.ThreadCount() <= MAX_JOBS );

    uint threadCount = pool.ThreadCount();

    const uint64 entriesPerThread = length / threadCount;
    const uint64 tailingEntries = length - ( entriesPerThread * threadCount );

    GenJob jobs[MAX_JOBS];

    for( uint64 i = 0; i < threadCount; i++ )
    {
        GenJob& job = jobs[i];

        job.length    = entriesPerThread;
        job.offset    = i * entriesPerThread;
        job.keyBuffer = keyBuffer;
    }

    jobs[threadCount-1].length += tailingEntries;

    pool.RunJob( GenJobThread, jobs, threadCount );
}

//-----------------------------------------------------------
template<uint MAX_JOBS, typename T>
inline void SortKeyGen::Sort( ThreadPool& pool, int64 length, const uint32* keys, const T* src, T* dst )
{
    Sort( pool, pool.ThreadCount(), length, keys, src, dst );
}

//-----------------------------------------------------------
template<uint MAX_JOBS, typename T>
inline void SortKeyGen::Sort( ThreadPool& pool, const uint32 desiredThreadCount, int64 length, const uint32* keys, const T* src, T* dst )
{
    ASSERT( pool.ThreadCount() <= MAX_JOBS );

    const int64 threadCount = desiredThreadCount == 0 ? (int64)pool.ThreadCount() : (int64)std::min( desiredThreadCount, pool.ThreadCount() );

    const int64 entriesPerThread = length / threadCount;
    const int64 tailingEntries   = length - ( entriesPerThread * threadCount );

    SortJob<T> jobs[MAX_JOBS];

    for( int64 i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];

        job.length = entriesPerThread;
        job.offset = i * entriesPerThread;
        job.keys   = keys;
        job.src    = src;
        job.dst    = dst;
    }

    jobs[threadCount-1].length += tailingEntries;

    pool.RunJob( SortThread, jobs, (uint)threadCount );
}

// //-----------------------------------------------------------
// template<uint MAX_JOBS>
// inline void SortKeyGen::Invert( ThreadPool& pool, uint64 length, const uint32* srcKeyBuffer, uint32* dstKeyBuffer )
// {
// 
// }

//-----------------------------------------------------------
inline void SortKeyGen::GenJobThread( GenJob* job )
{   
    const uint64 offset = job->offset;
    uint64 i = offset;
    const uint64 end = i + job->length;

    uint32* buffer = job->keyBuffer;

    for( ; i < end; i++ )
        buffer[i] = (uint32)i;
}

//-----------------------------------------------------------
template<typename T>
inline void SortKeyGen::SortThread( SortJob<T>* job )
{
    const int64   length = job->length;
    const int64   offset = job->offset;
    const uint32* keys   = job->keys + offset;
    const T*      src    = job->src;
    T*            dst    = job->dst + offset;

    for( int64 i = 0; i < length; i++ )
    {
        dst[i] = src[keys[i]];
    }
}
