#include "MemPlotter.h"
#include "threading/ThreadPool.h"
#include "util/Util.h"
#include "util/Log.h"
#include "SysHost.h"

#include "MemPhase1.h"
#include "MemPhase2.h"
#include "MemPhase3.h"
#include "MemPhase4.h"

//----------------------------------------------------------
void MemPlotter::ParseCLI( const GlobalPlotConfig& gCfg, CliParser& cli )
{
    _context.cfg.gCfg = &gCfg;
}

//----------------------------------------------------------
void MemPlotter::Init()
{
    auto& cfg = *_context.cfg.gCfg;

    const bool warmStart = cfg.warmStart;

    const NumaInfo* numa = nullptr;
    if( !cfg.disableNuma )
        numa = SysHost::GetNUMAInfo();
    
    if( numa && numa->nodeCount < 2 )
        numa = nullptr;
    
    if( numa )
    {
        // if( !SysHost::NumaSetThreadInterleavedMode() )
        //     Log::Error( "Warning: Failed to set NUMA interleaved mode." );
    }

    _context.threadCount = cfg.threadCount;
    
    // Create a thread pool
    _context.threadPool = new ThreadPool( cfg.threadCount, ThreadPool::Mode::Fixed, cfg.disableCpuAffinity );

    // Allocate buffers
    {
        const size_t totalMemory = SysHost::GetTotalSystemMemory();
        const size_t availMemory = SysHost::GetAvailableSystemMemory();

        Log::Line( "System Memory: %llu/%llu GiB.", availMemory BtoGB , totalMemory BtoGB );

        const size_t reqMem = _context.RequiredMemory();

        Log::Line( "Memory required: %llu GiB.", reqMem BtoGB );
        if( availMemory < reqMem  )
            Log::Line( "Warning: Not enough memory available. Buffer allocation may fail." );

        Log::Line( "Allocating buffers." );
        _context.AllocateBuffers(warmStart, numa);

        // Some table's kBC group pairings yield more values than 2^k. 
        // Therefore, we need to have some overflow space for kBC pairs.
        // We get an average of 236 entries per group.
        // We use a yBuffer for to mark group boundaries, 
        // so we fit as many as we can in it.
        const size_t maxKbcGroups  = _context.yBuffer0Size / sizeof( uint32 );

        // Since we use a meta buffer (64GiB) for pairing,
        // we can just use all its space to fit pairs.
        const size_t maxPairs      = _context.metaBuffer0Size / sizeof( Pair );

        _context.maxPairs     = maxPairs;
        _context.maxKBCGroups = maxKbcGroups;
    }
}

//----------------------------------------------------------
void MemPlotter::Run( const PlotRequest& request )
{
    auto& cx = _context;

    // Prepare context
    cx.plotId       = request.plotId;
    cx.plotMemo     = request.memo;
    cx.plotMemoSize = request.memoSize;

    // Start the first plot immediately, to exit early in case of error    
    if( request.isFirstPlot )
        BeginPlotFile( request );
    
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

    // Start the new plot file
    if( !request.isFirstPlot )
        BeginPlotFile( request );

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
    _context.plotWriter->EndPlot( true );

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

    cx.plotCount++;
}

//-----------------------------------------------------------
void MemPlotter::BeginPlotFile( const PlotRequest& request )
{
    // Re-create the serializer for now to workaround multiple-run serializer bug 
    if( _context.plotWriter )
    {
        delete _context.plotWriter;
        _context.plotWriter = nullptr;
    }

    if( !_context.plotWriter )
        _context.plotWriter = new PlotWriter();
    
    FatalIf( !_context.plotWriter->BeginPlot( PlotVersion::v2_0, request.outDir, request.plotFileName, 
              request.plotId, request.memo, request.memoSize, _context.cfg.gCfg->compressionLevel ),
            "Failed to open plot file with error: %d", _context.plotWriter->GetError() );
}

//-----------------------------------------------------------
void MemPlotter::WaitPlotWriter()
{
    _context.plotWriter->WaitForPlotToComplete();
    _context.plotWriter->DumpTables();
}


