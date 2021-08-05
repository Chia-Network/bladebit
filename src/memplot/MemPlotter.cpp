#include "MemPlotter.h"
#include "threading/ThreadPool.h"
#include "Util.h"
#include "util/Log.h"
#include "SysHost.h"

#include "MemPhase1.h"
#include "MemPhase2.h"
#include "MemPhase3.h"
#include "MemPhase4.h"


//----------------------------------------------------------
MemPlotter::MemPlotter( uint threadCount, bool warmStart, bool noNUMA )
{
    ZeroMem( &_context );

    const NumaInfo* numa = nullptr;
    if( !noNUMA )
        numa = SysHost::GetNUMAInfo();
    
    if( numa && numa->nodeCount < 2 )
        numa = nullptr;
    
    if( numa )
    {
        if( !SysHost::NumaSetThreadInterleavedMode() )
            Log::Error( "Warning: Failed to set NUMA interleaved mode." );
    }

    _context.threadCount = threadCount;
    
    // Create a thread pool
    _context.threadPool = new ThreadPool( threadCount, ThreadPool::Mode::Greedy );

    // Allocate buffers
    {
        const size_t totalMemory = SysHost::GetTotalSystemMemory();
        const size_t availMemory = SysHost::GetAvailableSystemMemory();

        Log::Line( "System Memory: %llu/%llu GiB.", availMemory BtoGB , totalMemory BtoGB );

        // YBuffers need to round up to chacha block size, so we just add an extra block always
        const size_t chachaBlockSize  = kF1BlockSizeBits / 8;

        const size_t t1XBuffer   = 16ull GB;
        const size_t t2LRBuffer  = 32ull GB;
        const size_t t3LRBuffer  = 32ull GB;
        const size_t t4LRBuffer  = 32ull GB;
        const size_t t5LRBuffer  = 32ull GB;
        const size_t t6LRBuffer  = 32ull GB;
        const size_t t7LRBuffer  = 32ull GB;
        const size_t t7YBuffer   = 16ull GB;

        const size_t yBuffer0    = 32ull GB + chachaBlockSize;
        const size_t yBuffer1    = 32ull GB + chachaBlockSize;
        const size_t metaBuffer0 = 64ull GB;
        const size_t metaBuffer1 = 64ull GB;

        const size_t reqMem = 
            t1XBuffer   +
            t2LRBuffer  +
            t3LRBuffer  +
            t4LRBuffer  +
            t5LRBuffer  +
            t6LRBuffer  +
            t7LRBuffer  +
            t7YBuffer   +
            yBuffer0    +
            yBuffer1    +
            metaBuffer0 +
            metaBuffer1;

        Log::Line( "Memory required: %lu GiB.", reqMem BtoGB );
        if( availMemory < reqMem  )
            Log::Line( "Warning: Not enough memory available. Buffer allocation may fail." );

        Log::Line( "Allocating buffers." );
        _context.t1XBuffer   = SafeAlloc<uint32>( t1XBuffer  , warmStart, numa );

        _context.t2LRBuffer  = SafeAlloc<Pair>  ( t2LRBuffer , warmStart, numa );
        _context.t3LRBuffer  = SafeAlloc<Pair>  ( t3LRBuffer , warmStart, numa );
        _context.t4LRBuffer  = SafeAlloc<Pair>  ( t4LRBuffer , warmStart, numa );
        _context.t5LRBuffer  = SafeAlloc<Pair>  ( t5LRBuffer , warmStart, numa );
        _context.t6LRBuffer  = SafeAlloc<Pair>  ( t6LRBuffer , warmStart, numa );

        _context.t7YBuffer   = SafeAlloc<uint32>( t7YBuffer  , warmStart, numa );
        _context.t7LRBuffer  = SafeAlloc<Pair>  ( t7LRBuffer , warmStart, numa );

        _context.yBuffer0    = SafeAlloc<uint64>( yBuffer0   , warmStart, numa );
        _context.yBuffer1    = SafeAlloc<uint64>( yBuffer1   , warmStart, numa );
        _context.metaBuffer0 = SafeAlloc<uint64>( metaBuffer0, warmStart, numa );
        _context.metaBuffer1 = SafeAlloc<uint64>( metaBuffer1, warmStart, numa );


        // Some table's kBC group pairings yield more values than 2^k. 
        // Therefore, we need to have some overflow space for kBC pairs.
        // We get an average of 236 entries per group.
        // We use a yBuffer for to mark group boundaries, 
        // so we fit as many as we can in it.
        const size_t maxKbcGroups  = yBuffer0 / sizeof( uint32 );

        // Since we use a meta buffer (64GiB) for pairing,
        // we can just use all its space to fit pairs.
        const size_t maxPairs      = metaBuffer0 / sizeof( Pair );

        _context.maxPairs     = maxPairs;
        _context.maxKBCGroups = maxKbcGroups;
    }
}

//----------------------------------------------------------
MemPlotter::~MemPlotter()
{}

//----------------------------------------------------------
bool MemPlotter::Run( const PlotRequest& request )
{
    auto& cx = _context;

    // Prepare context
    cx.plotId       = request.plotId;
    cx.plotMemo     = request.memo;
    cx.plotMemoSize = request.memoSize;
    
    // Open the plot file for writing before we actually start plotting
    const int PLOT_FILE_RETRIES = 16;
    FileStream* plotfile = new FileStream();
    ASSERT( plotfile );

    for( int i = 0;; )
    {
        if( !plotfile->Open( request.outPath, FileMode::Create, FileAccess::Write, FileFlags::NoBuffering | FileFlags::LargeFile ) )
        {
            if( ++i > PLOT_FILE_RETRIES )
            {
                Log::Error( "Error: Failed to open plot output file at %s for writing after %d tries.", request.outPath, PLOT_FILE_RETRIES );
                delete plotfile;
                return false;
            }
        }

        ASSERT( plotfile->IsOpen() );
        break;
    }
    
    // Start plotting
    auto plotTimer = TimerBegin();

    #if DBG_READ_PHASE_1_TABLES
    if( cx.plotCount > 0 )
    #endif
    {
        auto timeStart = plotTimer;
        Log::Line( "Running Phase 1" );

        MemPhase1 phase1( cx );
        phase1.Run();

        double elapsed = TimerEnd( timeStart );
        Log::Line( "Finished Phase 1 in %.2lf seconds.", elapsed );
    }

    {
        MemPhase2 phase2( cx );
        auto timeStart = TimerBegin();
        Log::Line( "Running Phase 2" );

        phase2.Run();

        double elapsed = TimerEnd( timeStart );
        Log::Line( "Finished Phase 2 in %.2lf seconds.", elapsed );
    }

    // Start writing the plot file
    if( !_context.plotWriter )
        _context.plotWriter = new DiskPlotWriter();
    
    cx.plotWriter->BeginPlot( request.outPath, *plotfile, request.plotId, request.memo, request.memoSize );

    {
        auto timeStart = TimerBegin();
        Log::Line( "Running Phase 3" );

        MemPhase3 phase3( cx );
        phase3.Run();

        double elapsed = TimerEnd( timeStart );
        Log::Line( "Finished Phase 3 in %.2lf seconds.", elapsed );
    }

    {
        auto timeStart = TimerBegin();
        Log::Line( "Running Phase 4" );

        MemPhase4 phase4( cx );
        phase4.Run();

        double elapsed = TimerEnd( timeStart );
        Log::Line( "Finished Phase 4 in %.2lf seconds.", elapsed );
    }

    // Wait flush writer, if this is the final plot
    if( request.IsFinalPlot )
    {
        auto timeStart = TimerBegin();
        Log::Line( "Writing final plot tables to disk" );

        WaitPlotWriter();

        double elapsed = TimerEnd( timeStart );
        Log::Line( "Finished writing tables to disk in %.2lf seconds.", elapsed );
        Log::Flush();
    }

    double plotElapsed = TimerEnd( plotTimer );
    Log::Line( "Finished plotting in %.2lf seconds (%.2lf minutes).", 
        plotElapsed, plotElapsed / 60.0 );

    cx.plotCount ++;
    return true;
}

//-----------------------------------------------------------
void MemPlotter::WaitPlotWriter()
{
    // if( !_context.plotWriter )
    //     _context.plotWriter = new DiskPlotWriter();
    // else
    // {
        // Wait until the current plot has finished writing
        if( !_context.plotWriter->WaitUntilFinishedWriting() )
            Fatal( "Failed to write plot file %s with error: %d", 
                _context.plotWriter->FilePath().c_str(),
                _context.plotWriter->GetError() );

        // Rename plot file to final plot file name (remove .tmp suffix)
        const char*  tmpName       = _context.plotWriter->FilePath().c_str();
        const size_t tmpNameLength = strlen( tmpName );

        char* plotName = new char[tmpNameLength - 3];  ASSERT( plotName );

        memcpy( plotName, tmpName, tmpNameLength - 4 );
        plotName[tmpNameLength-4] = 0;

        int r = rename( tmpName, plotName );
        
        if( r )
        {
            Log::Error( "Error: Failed to rename plot file %s.", tmpName );
            Log::Error( " Please rename it manually." );
        }

        Log::Line( "" );
        Log::Line( "Plot %s finished writing to disk:", r ? tmpName : plotName );

        delete[] plotName;

        // Print final pointer offsets
        const uint64* tablePointers = _context.plotWriter->GetTablePointers();
        for( uint i = 0; i < 7; i++ )
        {
            const uint64 ptr = Swap64( tablePointers[i] );
            Log::Line( "  Table %u pointer  : %16lu ( 0x%016lx )", i+1, ptr, ptr );
        }

        for( uint i = 7; i < 10; i++ )
        {
            const uint64 ptr = Swap64( tablePointers[i] );
            Log::Line( "  C%u table pointer : %16lu ( 0x%016lx )", i+1-7, ptr, ptr);
        }
        Log::Line( "" );
    // }
}

///
/// Internal methods
///
//-----------------------------------------------------------
template<typename T>
T* MemPlotter::SafeAlloc( size_t size, bool warmStart, const NumaInfo* numa )
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

        const uint   threadCount    = _context.threadPool->ThreadCount();
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

        _context.threadPool->RunJob( InitJob::Run, jobs, threadCount );
    }

    return ptr;
}

