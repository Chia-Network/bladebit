#pragma once
#include "threading/MTJob.h"

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
