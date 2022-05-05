#pragma once
#include "threading/MTJob.h"
#include "util/Util.h"

struct FaultMemoryPages : MTJob<FaultMemoryPages>
{
    byte*  pages;
    size_t pageSize;
    uint64 pageCount;

    //-----------------------------------------------------------
    inline void Run() override
    {
        const size_t pageSize = this->pageSize;

        byte*       page = this->pages;
        const byte* end  = page + this->pageCount * pageSize;

        do {
            *page = 0;
            page += pageSize;
            
        } while( page < end );
    }

    //-----------------------------------------------------------
    inline static void RunJob( ThreadPool& pool, uint32 threadCount, void* buffer, const size_t size )
    {
        threadCount = std::max( 1u, std::min( threadCount, pool.ThreadCount() ) );

        const size_t pageSize  = SysHost::GetPageSize();
        const uint64 pageCount = CDiv( size, (int)pageSize );
        if( threadCount == 1 )
        {
            FaultMemoryPages job;
            job.pages     = (byte*)buffer;
            job.pageCount = pageCount;
            job.pageSize  = pageSize;
            
            job.Run();
            return;
        }

        MTJobRunner<FaultMemoryPages> jobs( pool );

        const uint64 pagesPerThread    = pageCount / threadCount;
        uint64       numRemainderPages = pageCount - ( pagesPerThread * threadCount );

        byte* pages = (byte*)buffer;
        for( uint i = 0; i < threadCount; i++ )
        {
            auto& job = jobs[i];

            job.pages     = pages;
            job.pageSize  = pageSize;
            job.pageCount = pagesPerThread;

            if( numRemainderPages )
            {
                job.pageCount ++;
                numRemainderPages --;
            }

            pages += pageSize * job.pageCount;
        }

        jobs.Run( threadCount );
    }
};


struct MemCpyMT : MTJob<MemCpyMT>
{
    //-----------------------------------------------------------
    inline static void Copy( void* dst, const void* src, size_t size, ThreadPool& pool, uint32 threadCount )
    {
        if( size < 1 )
            return;

        ASSERT( dst );
        ASSERT( src );
        ASSERT( threadCount );

        threadCount = std::max( 1u, std::min( threadCount, pool.ThreadCount() ) );
        if( threadCount == 1 )
        {
            memcpy( dst, src, size );
            return;
        }

        MTJobRunner<MemCpyMT> jobs( pool );

              byte* dstBytes = (byte*)dst;
        const byte* srcBytes = (byte*)src;

        const size_t sizePerThread = size / threadCount;
        const size_t remainder     = size - sizePerThread * threadCount;

        for( uint i = 0; i < threadCount; i++ )
        {
            auto& job = jobs[i];

            job._dst  = dstBytes;
            job._src  = srcBytes;
            job._size = sizePerThread;

            dstBytes += sizePerThread;
            srcBytes += sizePerThread;
        }

        jobs[threadCount-1]._size += remainder;

        jobs.Run( threadCount );
    }

    //-----------------------------------------------------------
    inline void Run() override
    {
        memcpy( _dst, _src, _size );
    }

private:
    void*        _dst;
    const void*  _src;
    size_t       _size;

};
