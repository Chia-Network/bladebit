#include "Commands.h"
#include "plotting/GlobalPlotConfig.h"
#include "tools/PlotReader.h"
#include "threading/MTJob.h"
#include "harvesting/GreenReaper.h"

static constexpr double SECS_PER_DAY             = 24 * 60 * 60;
static constexpr double CHALLENGE_INTERVAL       = 9.375;
static constexpr uint32 CHALLENGES_PER_DAY       = (uint32)(SECS_PER_DAY / CHALLENGE_INTERVAL);

enum class SubCommand
{
    None,
    Farm,
};

struct Config
{
    GlobalPlotConfig* gCfg    = nullptr;
    SubCommand  subcommand    = SubCommand::Farm;
    const char* plotPath      = "";
    uint64      fetchCount    = 100;
    uint32      parallelCount = 1;
    size_t      farmSize      = 1 TBSi;
    double      maxLookupTime = 8.0;
    uint32      filterBits    = 512;
    uint32      partials      = 300;
    bool        powerUsage    = false;

    // Internally set
    // bool        hasFarmSize   = false;
    double      partialRatio  = 0;


    // Set by the simulation job
    size_t      jobsMemoryUsed = 0;
};

struct SimulatorJob : MTJob<SimulatorJob>
{
    Config*                cfg;
    Span<FilePlot>         plots;
    std::atomic<uint64>*   totalTimeNano;
    std::atomic<uint64>*   totalFullProofTimeNano;
    std::atomic<uint64>*   totalQualitiesTimeNano;
    std::atomic<uint64>*   maxFetchTimeNano;
    std::atomic<uint64>*   minFPFetchTimeNano;
    std::atomic<uint64>*   nFullProofsRequested;
    std::atomic<uint64>*   nQualitiesOnlyRequested;
    uint32                 decompressorThreadCount;

    virtual void Run() override;

    void RunFarm( PlotReader& reader, uint64 fetchCount, uint32 partialsCount );
};


static size_t CalculatePlotSizeBytes( const uint32 k, const uint32 compressionLevel );
static void DumpCompressedPlotCapacity( const Config& cfg, const uint32 k, const uint32 compressionLevel, const double fetchAverageSecs );

void CmdSimulateMain( GlobalPlotConfig& gCfg, CliParser& cli )
{
    Config cfg = {};
    cfg.gCfg = &gCfg;

    while( cli.HasArgs() )
    {
        if( cli.ArgConsume( "-h", "--help" ) )
        {
            CmdSimulateHelp();
            Exit( 0 );
        }
        else if( cli.ReadU64( cfg.fetchCount, "-n", "--iterations" ) ) continue;
        else if( cli.ReadU32( cfg.parallelCount, "-p", "--parallel" ) ) continue;
        else if( cli.ReadF64( cfg.maxLookupTime, "-l", "--lookup" ) ) continue;
        else if( cli.ReadU32( cfg.filterBits, "-f", "--filter" ) ) continue;
        else if( cli.ReadU32( cfg.partials, "--partials" ) ) continue;
        else if( cli.ReadSwitch( cfg.powerUsage, "--power" ) ) continue;
        // else if( cli.ReadSize( cfg.farmSize, "-s", "--size" ) )
        // {
        //     cfg.hasFarmSize = true;
        //     continue;
        // }
        // else if( cli.ArgConsume( "power" ) )
        // {
        //     cfg.subcommand = SubCommand::Farm;
            
        //     while( cli.HasArgs() )
        //     {
        //         else
        //             break;
        //     }
        //     break;
        // }
        else
            break;
    }

    FatalIf( !cli.HasArgs(), "Expected a path to a plot file." );
    {
        cfg.plotPath = cli.Arg();
        cli.NextArg();

        if( cli.HasArgs() )
        {
            Fatal( "Unexpected argument '%s'.", cli.Arg() );
            Exit( 1 );
        }
    }

    FatalIf( cfg.fetchCount < 1, "Invalid iteration count of %u.", cfg.fetchCount );
    FatalIf( cfg.maxLookupTime <= 0.0, "Invalid max lookup time of %lf.", cfg.maxLookupTime );
    FatalIf( cfg.filterBits < 1, "Invalid filter bits value of %u.", cfg.filterBits );
    FatalIf( cfg.parallelCount * (uint64)gCfg.threadCount > MAX_THREADS, 
        "Too many thread combination (%llu) between -t and -p", cfg.parallelCount * (llu)gCfg.threadCount );
    
    if( cfg.powerUsage && cfg.parallelCount > 1 )
    {
        Log::Line( "Power Simulation Info: Increasing the number of fetches requested times the number of contexts %llu -> %llu.",
                    (llu)cfg.fetchCount, (llu)cfg.fetchCount * cfg.parallelCount );
        cfg.fetchCount *= cfg.parallelCount;
    }

    // Lower the parallel count until all instances have at least 1 lookup
    if( cfg.parallelCount > cfg.fetchCount )
    {
        Log::Line( "Warning: Limiting parallel context count to %u, as it must be <= than the fetch count of %llu",
            cfg.parallelCount, (llu)cfg.fetchCount );
        cfg.parallelCount = (uint32)cfg.fetchCount;
    }


    FilePlot* plot = new FilePlot[cfg.parallelCount];
    for( uint32 i = 0; i < cfg.parallelCount; i++ )
    {
        if( !plot[i].Open( cfg.plotPath ) )
            Fatal( "Failed to open plot file at '%s' with error %d.", cfg.plotPath, plot[i].GetError() );
    }

    const uint32 compressionLevel = plot[0].CompressionLevel();
    FatalIf( compressionLevel < 1, "The plot %s is not compressed.", cfg.plotPath );

    Log::Line( "[Simulator for harvester farm capacity for K%2u C%u plots]", plot->K(), compressionLevel );
    Log::Line( " Simulating..." );
    Log::NewLine();


    ThreadPool pool( cfg.parallelCount, ThreadPool::Mode::Fixed, true );

    const uint32 decompressorThreadCount = std::min( gCfg.threadCount == 0 ? 8 : gCfg.threadCount, SysHost::GetLogicalCPUCount() );
    std::atomic<uint64> totalTimeNano           = 0;
    std::atomic<uint64> totalFullProofTimeNano  = 0;
    std::atomic<uint64> totalQualitiesTimeNano  = 0;
    std::atomic<uint64> maxFetchTimeNano        = 0;
    std::atomic<uint64> minFPFetchTimeNano      = std::numeric_limits<uint64>::max();
    std::atomic<uint64> nFullProofsRequested    = 0;
    std::atomic<uint64> nQualitiesOnlyRequested = 0;

    {
        SimulatorJob job = {};
        job.cfg                     = &cfg;
        job.plots                   = Span<FilePlot>( plot, cfg.parallelCount );
        job.totalTimeNano           = &totalTimeNano;
        job.decompressorThreadCount = decompressorThreadCount;
        job.totalFullProofTimeNano  = &totalFullProofTimeNano;
        job.totalQualitiesTimeNano  = &totalQualitiesTimeNano;
        job.maxFetchTimeNano        = &maxFetchTimeNano;
        job.minFPFetchTimeNano      = &minFPFetchTimeNano;
        job.nFullProofsRequested    = &nFullProofsRequested;
        job.nQualitiesOnlyRequested = &nQualitiesOnlyRequested;

        MTJobRunner<SimulatorJob>::RunFromInstance( pool, cfg.parallelCount, job );
    }

    // Report
    {
        const uint64 fetchCountAdjusted = CDiv( cfg.fetchCount, cfg.parallelCount ) * cfg.parallelCount;

        const uint64 actualPartials            = nFullProofsRequested;//(uint32)(cfg.partials * (cfg.fetchCount / (double)CHALLENGES_PER_DAY) );
        const uint64 totalTimeNanoAdjusted     = totalTimeNano / cfg.parallelCount;
        const uint64 fetchAverageNano          = fetchCountAdjusted == 0 ? 0 : totalTimeNanoAdjusted / fetchCountAdjusted;
        const uint64 actualQualitiesOnly       = nQualitiesOnlyRequested;
        const double fetchAverageSecs          = NanoSecondsToSeconds( fetchAverageNano );
        const double fetchAverageQualitiesOnly = nQualitiesOnlyRequested > 0 ?  NanoSecondsToSeconds( totalQualitiesTimeNano / nQualitiesOnlyRequested ) : 0;
        const double fetchMaxSecs              = NanoSecondsToSeconds( maxFetchTimeNano );
        const double fetchFpAverageSecs        = actualPartials > 0 ? NanoSecondsToSeconds( totalFullProofTimeNano / actualPartials ) : 0;
        const size_t memoryUsed                = cfg.jobsMemoryUsed * cfg.parallelCount;

        Log::Line( " Context count                   : %llu", (llu)cfg.parallelCount );
        Log::Line( " Thread per context instance     : %llu", (llu)gCfg.threadCount );
        Log::Line( " Memory used                     : %.1lfMiB ( %.1lfGiB )", (double)memoryUsed BtoMB, (double)memoryUsed BtoGB );
        Log::Line( " Challenge count                 : %llu", (llu)cfg.fetchCount );
        Log::Line( " Filter bits                     : %u", cfg.filterBits );
        Log::Line( " Effective partials              : %u ( %.2lf%% )", actualPartials, actualPartials / (double)cfg.fetchCount );
        // Log::Line( " Quality only lookups            : %u ( %.2lf%% )", actualQualitiesOnly, actualQualitiesOnly / (double)cfg.fetchCount );
        Log::Line( " Total time elapsed              : %.3lf seconds", NanoSecondsToSeconds( totalTimeNanoAdjusted ) );
        Log::Line( " Average time per plot lookup    : %.3lf seconds", fetchAverageSecs );
        Log::Line( " Average quality-only lookup time: %.3lf seconds", fetchAverageQualitiesOnly );
        Log::Line( " Worst plot lookup time          : %.3lf seconds", fetchMaxSecs );
        Log::Line( " Average full proof lookup time  : %.3lf seconds", fetchFpAverageSecs );
        Log::Line( " Fastest full proof lookup time  : %.3lf seconds", actualPartials == 0 ? 0.0 : NanoSecondsToSeconds( minFPFetchTimeNano ) );
        Log::NewLine();
        
        if( fetchMaxSecs >= cfg.maxLookupTime )
        {
            Log::Line( "*** Warning *** : Your worst plot lookup time of %.3lf was over the maximum set of %.3lf.", 
                fetchMaxSecs, cfg.maxLookupTime );
            Log::NewLine();
        }

        // Calculate farm size for this compression level
        Log::Line( " %10s | %-10s | %-10s | %-10s ", "compression", "plot count", "size TB", "size PB" );
        Log::Line( "------------------------------------------------" );
        DumpCompressedPlotCapacity( cfg, plot->K(), compressionLevel, fetchAverageSecs );
    }

    Log::NewLine();
    Exit( 0 );
}

void DumpCompressedPlotCapacity( const Config& cfg, const uint32 k, const uint32 compressionLevel, const double fetchAverageSecs )
{
    // Calculate farm size for this compression level
    const size_t plotSize  = CalculatePlotSizeBytes( k, compressionLevel );
    const uint64 plotCount = (uint64)(cfg.maxLookupTime / fetchAverageSecs * cfg.filterBits);

    const size_t farmSizeBytes = plotCount * plotSize;
    const size_t farmSizeTB    = BtoTBSi( farmSizeBytes );
    const auto   farmSizePB    = BtoPBSiF( farmSizeBytes );

    Log::Line( " C%-10u | %-10llu | %-10llu | %-10.2lf ", compressionLevel, plotCount, farmSizeTB, farmSizePB );
}

size_t CalculatePlotSizeBytes( const uint32 k, const uint32 compressionLevel )
{
    ASSERT( compressionLevel > 0 );

    auto info = GetCompressionInfoForLevel( compressionLevel );

    const uint64 tableEntryCount      = 1ull << k;
    const double tablePrunedFactors[] = { 0.798, 0.801, 0.807, 0.823, 0.865, 1, 1 };

    const size_t parkSizes[] = {
        0, // Table 1 is dropped
        compressionLevel >= 9 ? 0 : info.tableParkSize,
        compressionLevel >= 9 ? info.tableParkSize : CalculateParkSize( TableId::Table3 ),
        CalculateParkSize( TableId::Table4 ),
        CalculateParkSize( TableId::Table5 ),
        CalculateParkSize( TableId::Table6 ),
        CalculatePark7Size( k )
    };

    size_t tableSizes[7] = {};
    for( uint32 table = (uint32)TableId::Table2; table <= (uint32)TableId::Table7; table++ )
    {
        const uint64 prunedEntryCount = (uint64)(tableEntryCount * tablePrunedFactors[table]);
        const uint64 parkCount        = CDiv( prunedEntryCount, kEntriesPerPark );

        tableSizes[table] = parkCount * parkSizes[table];
    }

    const size_t c1EntrySize = RoundUpToNextBoundary( k, 8 );
    const size_t c3ParkCount = CDiv( tableEntryCount, kCheckpoint1Interval );
    const size_t c3Size      = c3ParkCount * CalculateC3Size(); 
    const size_t c1Size      = c1EntrySize * ( tableEntryCount / kCheckpoint1Interval ) + c1EntrySize;
    const size_t c2Size      = c1EntrySize * ( tableEntryCount / (kCheckpoint1Interval * kCheckpoint2Interval) ) + c1EntrySize;

    const size_t plotSize = c3Size + c1Size + c2Size +
        tableSizes[0] +
        tableSizes[1] +
        tableSizes[2] +
        tableSizes[3] +
        tableSizes[4] +
        tableSizes[5] +
        tableSizes[6];
    
    return plotSize;
}


void SimulatorJob::Run()
{
    FilePlot& plot = plots[JobId()];

    PlotReader reader( plot );
    reader.ConfigDecompressor( decompressorThreadCount, cfg->gCfg->disableCpuAffinity, decompressorThreadCount * JobId() );

    const double challengeRatio = cfg->fetchCount / (double)CHALLENGES_PER_DAY;
    const uint64 actualPartials = (uint64)(cfg->partials * challengeRatio);

    uint64 fetchCountForJob, partialsForJob;
    {
        uint64 _;
        GetThreadOffsets( this, cfg->fetchCount, fetchCountForJob, _, _  );
        GetThreadOffsets( this, actualPartials, partialsForJob, _, _  );
        ASSERT( fetchCountForJob > 0 );
    }

    switch( cfg->subcommand )
    {
        case SubCommand::Farm:
            RunFarm( reader, fetchCountForJob, (uint32)partialsForJob );
            break;
    
        default:
            break;
    }

    if( IsControlThread() )
        cfg->jobsMemoryUsed = grGetMemoryUsage( reader.GetDecompressorContext() );
}

void SimulatorJob::RunFarm( PlotReader& reader, const uint64 fetchCount, const uint32 partialCount )
{
    const uint64 startF7 = fetchCount * JobId();
    const size_t f7Size  = CDiv( reader.PlotFile().K(), 8 );

    Duration totalFetchDuration      = Duration::zero();
    Duration totalFullProofDuration  = Duration::zero();
    Duration totalQualitiesDuration  = Duration::zero();
    uint64   maxFetchDurationNano    = 0;
    uint64   minFetchDurationNano    = std::numeric_limits<uint64>::max();
    uint64   nFullProofsRequested    = 0;
    uint64   nQualitiesOnlyRequested = 0;

    uint64 fullProofXs[BB_PLOT_PROOF_X_COUNT] = {};
    byte   quality    [BB_CHIA_QUALITY_SIZE]  = {};

    byte challenge[BB_CHIA_CHALLENGE_SIZE] = {};
    {
        const char* challengeStr = "00000000000000ee9355068689bd558eafe07cc7af47ad1574b074fc34d6913a";
        HexStrToBytes( challengeStr, sizeof( challenge )*2, challenge, sizeof( challenge ) );
    }

    const int64 partialInterval = partialCount == 0 ? std::numeric_limits<int64>::max() : (int64)std::min( fetchCount, fetchCount / partialCount );
          int64 nextPartial     = partialCount == 0 ? std::numeric_limits<int64>::max() : 1;

    for( uint64 f7 = startF7; f7 < startF7 + fetchCount; f7++ )
    {
        // Embed f7 into challenge as BE
        for( size_t i = 0; i < f7Size; i++ )
            challenge[i] = (byte)(f7 >> ((f7Size - i - 1) * 8));
        const auto timer = TimerBegin();

        // Read indices for F7
              uint64 p7IndexBase = 0;
        const uint64 matchCount  = reader.GetP7IndicesForF7( f7, p7IndexBase );

        const bool fetchFullProof = --nextPartial <= 0;
        if( fetchFullProof )
            nextPartial = partialInterval;

        for( uint64 i = 0; i < matchCount; i++ )
        {
            uint64 p7Entry;
            FatalIf( ! reader.ReadP7Entry( p7IndexBase + i, p7Entry ), 
                "Failed to read P7 entry at %llu. (F7 = %llu)", (llu)p7IndexBase + i, (llu)f7 );

            ProofFetchResult rQ, rP = ProofFetchResult::OK;

            rQ = reader.FetchQualityForP7Entry( p7Entry, challenge, quality );
            if( fetchFullProof )
                rP = reader.FetchProof( p7Entry, fullProofXs );

            const auto errR = rQ != ProofFetchResult::OK ? rQ : rP;
            if( errR != ProofFetchResult::OK )
            {
                FatalIf( errR == ProofFetchResult::Error, "Error while fetching proof for F7 %llu.", (llu)f7 );
                FatalIf( errR == ProofFetchResult::CompressionError, "Decompression error while fetching proof for F7 %llu.", (llu)f7 );
            }
        }

        const auto elapsed = TimerEndTicks( timer );
        totalFetchDuration += elapsed;

        if( fetchFullProof )
        {
            totalFullProofDuration += elapsed;
            nFullProofsRequested++;

            if( matchCount > 0 )
                minFetchDurationNano = std::min( minFetchDurationNano, (uint64)TicksToNanoSeconds( elapsed ) );
        }
        else {
            totalQualitiesDuration += elapsed;
            nQualitiesOnlyRequested++;
        }
        
        maxFetchDurationNano = std::max( maxFetchDurationNano, (uint64)TicksToNanoSeconds( elapsed ) );
        
        // Wait for next proof when in power usage mode
        if( cfg->powerUsage )
        {
            Thread::Sleep( (long)(CHALLENGE_INTERVAL * 1000.0) );
        }
    }

    *this->totalTimeNano           += (uint64)TicksToNanoSeconds( totalFetchDuration );
    *this->totalFullProofTimeNano  += (uint64)TicksToNanoSeconds( totalFullProofDuration );
    *this->totalQualitiesTimeNano  += (uint64)TicksToNanoSeconds( totalQualitiesDuration );
    *this->nFullProofsRequested    += nFullProofsRequested;
    *this->nQualitiesOnlyRequested += nQualitiesOnlyRequested;

    {
        uint64 curMaxFetch = this->maxFetchTimeNano->load( std::memory_order_relaxed );
        while( curMaxFetch < maxFetchDurationNano && 
            !this->maxFetchTimeNano->compare_exchange_weak( curMaxFetch, maxFetchDurationNano, 
                                                            std::memory_order_release, std::memory_order_relaxed ) );
    }

    {
        uint64 curMinFetch = this->minFPFetchTimeNano->load( std::memory_order_relaxed );
        while( curMinFetch > minFetchDurationNano && 
            !this->minFPFetchTimeNano->compare_exchange_weak( curMinFetch, minFetchDurationNano, 
                                                              std::memory_order_release, std::memory_order_relaxed ) );
    }
}


static const char _help[] = R"(simulate [OPTIONS] <plot_file_path>
OPTIONS:
 -h, --help           : Display this help message and exit.
 -n, --iterations     : How many iterations to run (default = %llu)
 -p, --parallel       : How many instances to run in parallel (default = 1)
 -l, --lookup         : Maximum allowed time per proof lookup (default = %.2lf)
 -f, --filter         : Plot filter bit count (default = %u)
 --partials           : Partials per-day simulation. (default = %u)
)";

void CmdSimulateHelp()
{
    Config cfg = {};
    printf( _help, 
        (llu)cfg.fetchCount,
        cfg.maxLookupTime,
        cfg.filterBits,
        cfg.partials
    );
}