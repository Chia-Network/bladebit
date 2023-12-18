#include "PlotContext.h"
#include "SysHost.h"

uint16_t L_targets[2][kBC][kExtraBitsPow];

const size_t MemPlotContext::RequiredMemory() {
    return
        t1XBufferSize   +
        t2LRBufferSize  +
        t3LRBufferSize  +
        t4LRBufferSize  +
        t5LRBufferSize  +
        t6LRBufferSize  +
        t7LRBufferSize  +
        t7YBufferSize   +
        yBuffer0Size    +
        yBuffer1Size    +
        metaBuffer0Size +
        metaBuffer1Size;

}

void MemPlotContext::AllocateBuffers(bool warmStart, const NumaInfo* numa) {
    t1XBuffer   = SafeAlloc<uint32>( t1XBufferSize  , warmStart, numa );
    t2LRBuffer  = SafeAlloc<Pair>  ( t2LRBufferSize , warmStart, numa );
    t3LRBuffer  = SafeAlloc<Pair>  ( t3LRBufferSize , warmStart, numa );
    t4LRBuffer  = SafeAlloc<Pair>  ( t4LRBufferSize , warmStart, numa );
    t5LRBuffer  = SafeAlloc<Pair>  ( t5LRBufferSize , warmStart, numa );
    t6LRBuffer  = SafeAlloc<Pair>  ( t6LRBufferSize , warmStart, numa );

    t7YBuffer   = SafeAlloc<uint32>( t7YBufferSize  , warmStart, numa );
    t7LRBuffer  = SafeAlloc<Pair>  ( t7LRBufferSize , warmStart, numa );

    yBuffer0    = SafeAlloc<uint64>( yBuffer0Size   , warmStart, numa );
    yBuffer1    = SafeAlloc<uint64>( yBuffer1Size   , warmStart, numa );
    metaBuffer0 = SafeAlloc<uint64>( metaBuffer0Size, warmStart, numa );
    metaBuffer1 = SafeAlloc<uint64>( metaBuffer1Size, warmStart, numa );
}



///
/// Internal methods
///
//-----------------------------------------------------------
template<typename T>
T* MemPlotContext::SafeAlloc( size_t size, bool warmStart, const NumaInfo* numa )
{
#if DEBUG || BOUNDS_PROTECTION

    const size_t originalSize = size;
    const size_t pageSize     = SysHost::GetPageSize();
    size = pageSize * 2 + RoundUpToNextBoundary( size, (int)pageSize );

#endif

    T* ptr = (T*)SysHost::VirtualAlloc( size, false );

    if( !ptr )
    {
        Fatal( "Error: Failed to allocate required buffers." );
    }

    if( numa )
    {
        if( !SysHost::NumaSetMemoryInterleavedMode( ptr, size ) )
            Log::Error( "Warning: Failed to bind NUMA memory." );
    }

    // Protect memory boundaries
#if DEBUG || BOUNDS_PROTECTION
    {
        byte* p = (byte*)ptr;
        ptr = (T*)(p + pageSize);

        SysHost::VirtualProtect( p, pageSize, VProtect::NoAccess );
        SysHost::VirtualProtect( p + size - pageSize, pageSize, VProtect::NoAccess );
    }
#endif

    // Touch pages to initialize them, if specified
    if( warmStart )
    {
        struct InitJob
        {
            byte*  pages;
            size_t pageSize;
            uint64 pageCount;

            inline static void Run( InitJob* job )
            {
                const size_t pageSize = job->pageSize;

                byte*       page = job->pages;
                const byte* end  = page + job->pageCount * pageSize;

                do {
                    *page = 0;
                    page += pageSize;

                } while ( page < end );
            }
        };

#if DEBUG || BOUNDS_PROTECTION
        size = originalSize;
#endif

        InitJob jobs[MAX_THREADS];

        const uint   threadCount    = threadPool->ThreadCount();
        const size_t pageSize       = SysHost::GetPageSize();
        const uint64 pageCount      = CDiv( size, (int)pageSize );
        const uint64 pagesPerThread = pageCount / threadCount;

        uint64 numRemainderPages = pageCount - ( pagesPerThread * threadCount );

        byte* pages = (byte*)ptr;
        for( uint i = 0; i < threadCount; i++ )
        {
            InitJob& job = jobs[i];

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

        threadPool->RunJob( InitJob::Run, jobs, threadCount );
    }

    return ptr;
}
