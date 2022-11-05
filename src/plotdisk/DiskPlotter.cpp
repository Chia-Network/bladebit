#include "DiskPlotter.h"
#include "util/Log.h"
#include "util/Util.h"
#include "util/CliParser.h"
#include "util/jobs/MemJobs.h"
#include "io/FileStream.h"

#include "DiskFp.h"
#include "DiskPlotPhase1.h"
#include "DiskPlotPhase2.h"
#include "DiskPlotPhase3.h"
#include "SysHost.h"

#include "k32/DiskPlotBounded.h"


size_t ValidateTmpPathAndGetBlockSize( DiskPlotter::Config& cfg );


//-----------------------------------------------------------
// DiskPlotter::DiskPlotter()
// {
// }

//-----------------------------------------------------------
DiskPlotter::DiskPlotter( const Config& cfg )
{
    ASSERT( cfg.tmpPath  );
    ASSERT( cfg.tmpPath2 );

    // Initialize tables for matching
    LoadLTargets();
    
    ZeroMem( &_cx );
    
    GlobalPlotConfig& gCfg = *cfg.globalCfg;

    FatalIf( !GetTmpPathsBlockSizes( cfg.tmpPath, cfg.tmpPath2, _cx.tmp1BlockSize, _cx.tmp2BlockSize ),
        "Failed to obtain temp paths block size from t1: '%s' or %s t2: '%s'.", cfg.tmpPath, cfg.tmpPath2 );

    FatalIf( _cx.tmp1BlockSize < 8 || _cx.tmp2BlockSize < 8,"File system block size is too small.." );

    const uint  sysLogicalCoreCount = SysHost::GetLogicalCPUCount();
    const auto* numa                = SysHost::GetNUMAInfo();

    // _cx.threadCount   = gCfg.threadCount;
    _cx.ioThreadCount = cfg.ioThreadCount;
    _cx.f1ThreadCount = cfg.f1ThreadCount == 0 ? gCfg.threadCount : std::min( cfg.f1ThreadCount, sysLogicalCoreCount );
    _cx.fpThreadCount = cfg.fpThreadCount == 0 ? gCfg.threadCount : std::min( cfg.fpThreadCount, sysLogicalCoreCount );
    _cx.cThreadCount  = cfg.cThreadCount  == 0 ? gCfg.threadCount : std::min( cfg.cThreadCount , sysLogicalCoreCount );
    _cx.p2ThreadCount = cfg.p2ThreadCount == 0 ? gCfg.threadCount : std::min( cfg.p2ThreadCount, sysLogicalCoreCount );
    _cx.p3ThreadCount = cfg.p3ThreadCount == 0 ? gCfg.threadCount : std::min( cfg.p3ThreadCount, sysLogicalCoreCount );

    const size_t heapSize = GetRequiredSizeForBuckets( cfg.bounded, cfg.numBuckets, _cx.tmp1BlockSize, _cx.tmp2BlockSize, _cx.fpThreadCount );
    ASSERT( heapSize );

    _cfg            = cfg;
    _cx.cfg         = &_cfg;
    _cx.tmpPath     = cfg.tmpPath;
    _cx.tmpPath2    = cfg.tmpPath2;
    _cx.numBuckets  = cfg.numBuckets;
    _cx.heapSize    = heapSize;
    _cx.cacheSize   = cfg.cacheSize;

    Log::Line( "[Bladebit Disk Plotter]" );
    Log::Line( " Heap size      : %.2lf GiB ( %.2lf MiB )", (double)_cx.heapSize BtoGB, (double)_cx.heapSize BtoMB );
    Log::Line( " Cache size     : %.2lf GiB ( %.2lf MiB )", (double)_cx.cacheSize BtoGB, (double)_cx.cacheSize BtoMB );
    Log::Line( " Bucket count   : %u"       , _cx.numBuckets    );
    Log::Line( " Alternating I/O: %s"       , cfg.alternateBuckets ? "true" : "false" );
    Log::Line( " F1  threads    : %u"       , _cx.f1ThreadCount );
    Log::Line( " FP  threads    : %u"       , _cx.fpThreadCount );
    Log::Line( " C   threads    : %u"       , _cx.cThreadCount  );
    Log::Line( " P2  threads    : %u"       , _cx.p2ThreadCount );
    Log::Line( " P3  threads    : %u"       , _cx.p3ThreadCount );
    Log::Line( " I/O threads    : %u"       , _cx.ioThreadCount );
    Log::Line( " Temp1 block sz : %u"       , _cx.tmp1BlockSize );
    Log::Line( " Temp2 block sz : %u"       , _cx.tmp2BlockSize );
    Log::Line( " Temp1 path     : %s"       , _cx.tmpPath       );
    Log::Line( " Temp2 path     : %s"       , _cx.tmpPath2      );
#if BB_IO_METRICS_ON
    Log::Line( " I/O metrices enabled." );
#endif

    Log::Line( " Allocating memory" );
    _cx.heapBuffer = bbvirtalloc<byte>( _cx.heapSize );
    if( numa && !gCfg.disableNuma )
    {
        if( !SysHost::NumaSetMemoryInterleavedMode( _cx.heapBuffer, _cx.heapSize  ) )
            Log::Error( "WARNING: Failed to bind NUMA memory on the heap." );
    }

    if( _cx.cacheSize )
    {
        // We need to align the cache size to the block size of the temp2 dir
        const size_t cachePerBucket        = _cx.cacheSize / _cx.numBuckets;
        const size_t cachePerBucketAligned = CDivT( cachePerBucket, _cx.tmp2BlockSize ) * _cx.tmp2BlockSize;
        const size_t alignedCacheSize      = cachePerBucketAligned * _cx.numBuckets;

        if( alignedCacheSize != _cx.cacheSize )
        {
            Log::Line( "WARNING: Cache size has been adjusted from %.2lf to %.2lf MiB to make it block-aligned.",
                (double)_cx.cacheSize BtoMB, (double)alignedCacheSize BtoMB );
            _cx.cacheSize = alignedCacheSize;
        }

        _cx.cache = bbvirtalloc<byte>( _cx.cacheSize );
        if( numa && !gCfg.disableNuma )
        {
            if( !SysHost::NumaSetMemoryInterleavedMode( _cx.cache, _cx.cacheSize  ) )
                Log::Error( "WARNING: Failed to bind NUMA memory on the cache." );
        }
    }

    // Initialize our Thread Pool and IO Queue
    const int32 ioThreadId = -1;    // Force unpinned IO thread for now. We should bind it to the last used thread, of the max threads used...
    _cx.threadPool = new ThreadPool( sysLogicalCoreCount, ThreadPool::Mode::Fixed, gCfg.disableCpuAffinity );
    _cx.ioQueue    = new DiskBufferQueue( _cx.tmpPath, _cx.tmpPath2, gCfg.outputFolder, _cx.heapBuffer, _cx.heapSize, _cx.ioThreadCount, ioThreadId );
    _cx.fencePool  = new FencePool( 8 );

    if( cfg.globalCfg->warmStart )
    {
        Log::Line( "Warm start: Pre-faulting memory pages..." );

        const uint32 threadCount = cfg.globalCfg->threadCount == 0 ? sysLogicalCoreCount :
                                    std::min( cfg.globalCfg->threadCount, sysLogicalCoreCount );

        FaultMemoryPages::RunJob( *_cx.threadPool, threadCount, _cx.heapBuffer, _cx.heapSize );

        if( _cx.cacheSize )
            FaultMemoryPages::RunJob( *_cx.threadPool, threadCount, _cx.cache, _cx.cacheSize );

        Log::Line( "Memory initialized." );
    }
}

//-----------------------------------------------------------
void DiskPlotter::Plot( const PlotRequest& req )
{
    // Reset state
    memset( _cx.plotTablePointers   , 0, sizeof( _cx.plotTablePointers ) );
    memset( _cx.plotTableSizes      , 0, sizeof( _cx.plotTableSizes ) );
    memset( _cx.bucketCounts        , 0, sizeof( _cx.bucketCounts ) );
    memset( _cx.entryCounts         , 0, sizeof( _cx.entryCounts ) );
    memset( _cx.ptrTableBucketCounts, 0, sizeof( _cx.ptrTableBucketCounts ) );
    memset( _cx.bucketSlices        , 0, sizeof( _cx.bucketSlices ) );
    memset( _cx.p1TableWaitTime     , 0, sizeof( _cx.p1TableWaitTime ) );
    _cx.ioWaitTime      = Duration::zero();
    _cx.cTableWaitTime  = Duration::zero();
    _cx.p7WaitTime      = Duration::zero();

    const bool bounded = _cx.cfg->bounded;

    _cx.plotId       = req.plotId;
    _cx.plotMemo     = req.plotMemo;
    _cx.plotMemoSize = req.plotMemoSize;

    _cx.ioQueue->OpenPlotFile( req.plotFileName, req.plotId, req.plotMemo, req.plotMemoSize );

    #if ( _DEBUG && ( BB_DP_DBG_SKIP_PHASE_1 || BB_DP_P1_SKIP_TO_TABLE || BB_DP_DBG_SKIP_TO_C_TABLES ) )
        BB_DP_DBG_ReadTableCounts( _cx );
    #endif

    Log::Line( "Started plot." );
    auto plotTimer = TimerBegin();

    {
        Log::Line( "Running Phase 1" );
        const auto timer = TimerBegin();

        if( bounded )
        {
            K32BoundedPhase1 phase1( _cx );
            #if !( _DEBUG && BB_DP_DBG_SKIP_PHASE_1 )
                phase1.Run();
            #endif
        }
        else
        {
            DiskPlotPhase1 phase1( _cx );
            #if !( _DEBUG && BB_DP_DBG_SKIP_PHASE_1 )
                phase1.Run();
            #endif
        }

        #if ( _DEBUG && !BB_DP_DBG_SKIP_PHASE_1 )
            BB_DP_DBG_WriteTableCounts( _cx );
        #endif

        const double elapsed = TimerEnd( timer );
        Log::Line( "Finished Phase 1 in %.2lf seconds ( %.1lf minutes ).", elapsed, elapsed / 60 );
    }

    {
        Log::Line( "Running Phase 2" );
        const auto timer = TimerBegin();

        // if( bounded )
        // {
        //     Fatal( "Phase 2 bounded not implemented." );
        // }
        // else
        {
            DiskPlotPhase2 phase2( _cx );
            phase2.Run();
        }

        const double elapsed = TimerEnd( timer );
        Log::Line( "Finished Phase 2 in %.2lf seconds ( %.1lf minutes ).", elapsed, elapsed / 60 );
    }

    {
        Log::Line( "Running Phase 3" );
        const auto timer = TimerBegin();

        // if( bounded )
        // {
        //     Fatal( "Phase 3 bounded not implemented." );
        // }
        // else
        {
            DiskPlotPhase3 phase3( _cx );
            phase3.Run();
        }

        const double elapsed = TimerEnd( timer );
        Log::Line( "Finished Phase 3 in %.2lf seconds ( %.1lf minutes ).", elapsed, elapsed / 60 );
    }
    Log::Line("Total plot I/O wait time: %.2lf seconds.", TicksToSeconds( _cx.ioWaitTime ) );


    {
        // Now we need to update the table sizes on the file
        Log::Line( "Waiting for plot file to complete pending writes..." );
        const auto timer = TimerBegin();

        // Update the table pointers location
        DiskBufferQueue& ioQueue = *_cx.ioQueue;
        ASSERT( sizeof( _cx.plotTablePointers ) == sizeof( uint64 ) * 10 );

        // Convert them to big endian
        for( int i = 0; i < 10; i++ )
            _cx.plotTablePointers[i] = Swap64( _cx.plotTablePointers[i] );

        const int64 tablePtrsStart = (int64)ioQueue.PlotTablePointersAddress();
        ioQueue.SeekFile( FileId::PLOT, 0, tablePtrsStart, SeekOrigin::Begin );
        ioQueue.WriteFile( FileId::PLOT, 0, _cx.plotTablePointers, sizeof( _cx.plotTablePointers ) );
        
        // Wait for all IO commands to finish
        Fence& fence = _cx.fencePool->RequireFence();
        ioQueue.SignalFence( fence );
        ioQueue.CommitCommands();
        fence.Wait();
        _cx.fencePool->ReleaseFence( fence );
        
        const double elapsed = TimerEnd( timer );
        Log::Line( "Completed pending writes in %.2lf seconds.", elapsed );
        Log::Line( "Finished writing plot %s.", req.plotFileName );
        Log::Line( "Final plot table pointers: " );

        for( int i = 0; i < 10; i++ )
        {
            const uint64 addy = Swap64( _cx.plotTablePointers[i] );

            if( i < 7 )
                Log::Line( " Table %d: %16lu ( 0x%016lx )", i+1, addy, addy );
            else
                Log::Line( " C %d    : %16lu ( 0x%016lx )", i-6, addy, addy );
        }
        Log::Line( "" );

        double plotElapsed = TimerEnd( plotTimer );
        Log::Line( "Finished plotting in %.2lf seconds ( %.1lf minutes ).", plotElapsed, plotElapsed / 60 );

        // Rename plot file
        ioQueue.FinishPlot( fence );
    }
}

//-----------------------------------------------------------
void DiskPlotter::ParseCommandLine( CliParser& cli, Config& cfg )
{
    while( cli.HasArgs() )
    {
        if( cli.ReadU32( cfg.numBuckets,  "-b", "--buckets" ) ) 
            continue;
        if( cli.ReadUnswitch( cfg.bounded, "--unbounded" ) )
            continue;
        if( cli.ReadSwitch( cfg.alternateBuckets, "-a", "--alternate" ) )
            continue;
        if( cli.ReadStr( cfg.tmpPath, "-t1", "--temp1" ) )
            continue;
        if( cli.ReadStr( cfg.tmpPath2, "-t2", "--temp2" ) )
            continue;
        if( cli.ReadSwitch( cfg.noTmp1DirectIO, "--no-t1-direct" ) )
            continue;
        if( cli.ReadSwitch( cfg.noTmp2DirectIO, "--no-t2-direct" ) )
            continue;
        if( cli.ReadSize( cfg.cacheSize, "--cache" ) )
            continue;
        if( cli.ReadU32( cfg.f1ThreadCount, "--f1-threads" ) )
            continue;
        if( cli.ReadU32( cfg.fpThreadCount, "--fp-threads" ) )
            continue;
        if( cli.ReadU32( cfg.cThreadCount, "--c-threads" ) )
            continue;
        if( cli.ReadU32( cfg.p2ThreadCount, "--p2-threads" ) )
            continue;
        if( cli.ReadU32( cfg.p3ThreadCount, "--p3-threads" ) )
            continue;
        if( cli.ArgConsume( "-s", "--sizes" ) )
        {
            FatalIf( cfg.numBuckets < BB_DP_MIN_BUCKET_COUNT || cfg.numBuckets > BB_DP_MAX_BUCKET_COUNT,
                "Buckets must be between %u and %u, inclusive.", (uint)BB_DP_MIN_BUCKET_COUNT, (uint)BB_DP_MAX_BUCKET_COUNT );
            FatalIf( ( cfg.numBuckets & ( cfg.numBuckets - 1 ) ) != 0, "Buckets must be power of 2." );
            FatalIf( !cfg.tmpPath, "Please specify at least 1 temporary path." );

            const uint32 threadCount = bbclamp<uint32>( cfg.globalCfg->threadCount, 1u, SysHost::GetLogicalCPUCount() );

            size_t heapSize = 0;
            cfg.tmpPath2 = cfg.tmpPath2 ? cfg.tmpPath2 : cfg.tmpPath;
            heapSize = GetRequiredSizeForBuckets( cfg.bounded, cfg.numBuckets, cfg.tmpPath2, cfg.tmpPath, threadCount );
            
            Log::Line( "Buckets: %u | Heap Sizes: %.2lf GiB", cfg.numBuckets, (double)heapSize BtoGB );
            exit( 0 );
        }
        if( cli.ArgConsume( "-h", "--help" ) )
        {
            PrintUsage();
            exit( 0 );
        }
        else if( cli.Arg()[0] == '-' )
        {
            Fatal( "Unexpected argument '%s'.", cli.Arg() );
        }
        else 
        {
            cfg.globalCfg->outputFolder = cli.ArgConsume();

            FatalIf( strlen( cfg.globalCfg->outputFolder ) == 0, "Invalid plot output directory." );
            FatalIf( cli.HasArgs(), "Unexpected argument '%s'.", cli.Arg() );
            break;
        }
    }

    ///
    /// Validate some parameters
    ///
    FatalIf( cfg.tmpPath == nullptr, "At least 1 temporary path (--temp) must be specified." );
    if( cfg.tmpPath2 == nullptr )
        cfg.tmpPath2 = cfg.tmpPath;

    FatalIf( cfg.numBuckets < BB_DP_MIN_BUCKET_COUNT || cfg.numBuckets > BB_DP_MAX_BUCKET_COUNT,
        "Buckets must be between %u and %u, inclusive.", (uint)BB_DP_MIN_BUCKET_COUNT, (uint)BB_DP_MAX_BUCKET_COUNT );

    FatalIf( ( cfg.numBuckets & ( cfg.numBuckets - 1 ) ) != 0, "Buckets must be power of 2." );

    FatalIf( cfg.numBuckets >= 1024, "1024 buckets are not allowed for plots < k33." );
    FatalIf( cfg.numBuckets < 128 && !cfg.bounded, "64 buckets is only allowed for bounded k=32 plots." );

    const uint32 sysLogicalCoreCount = SysHost::GetLogicalCPUCount();

    if( cfg.ioThreadCount == 0 )
        cfg.ioThreadCount = 1;        // #TODO: figure out a reasonable default. Probably 1 or 2 for current consumer NVMes running on PCIe3...
    else if( cfg.ioThreadCount > sysLogicalCoreCount )
    {
        Log::Line( "Warning: Limiting disk queue threads to %u, which is the system's logical CPU count.", sysLogicalCoreCount );
        cfg.ioThreadCount = sysLogicalCoreCount;
    }

    const uint32 defaultThreads = cfg.globalCfg->threadCount == 0 ? sysLogicalCoreCount :
                                 std::min( sysLogicalCoreCount, cfg.globalCfg->threadCount );
    
    auto validateThreads = [&]( uint32& targetValue ) {
            targetValue = targetValue == 0 ? defaultThreads : std::min( sysLogicalCoreCount, targetValue );
    };

    validateThreads( cfg.f1ThreadCount );
    validateThreads( cfg.fpThreadCount );
    validateThreads( cfg.cThreadCount  );
    validateThreads( cfg.p2ThreadCount );
    validateThreads( cfg.p3ThreadCount );
}

//-----------------------------------------------------------
bool DiskPlotter::GetTmpPathsBlockSizes( const char* tmpPath1, const char* tmpPath2, size_t& tmpPath1Size, size_t& tmpPath2Size )
{
    ASSERT( tmpPath1 && *tmpPath1 );
    ASSERT( tmpPath2 && *tmpPath2 );

    tmpPath1Size = FileStream::GetBlockSizeForPath( tmpPath1 );
    tmpPath2Size = FileStream::GetBlockSizeForPath( tmpPath2 );

    return tmpPath1Size && tmpPath2Size;
}

//-----------------------------------------------------------
size_t DiskPlotter::GetRequiredSizeForBuckets( const bool bounded, const uint32 numBuckets, const char* tmpPath1, const char* tmpPath2, const uint32 threadCount )
{
    size_t blockSizes[2] = { 0 };

    if( !GetTmpPathsBlockSizes( tmpPath1, tmpPath2, blockSizes[0], blockSizes[1] ) )
        return 0;

    return GetRequiredSizeForBuckets( bounded, numBuckets, blockSizes[0], blockSizes[1], threadCount );
}

//-----------------------------------------------------------
size_t DiskPlotter::GetRequiredSizeForBuckets( const bool bounded, const uint32 numBuckets, const size_t fxBlockSize, const size_t pairsBlockSize, const uint32 threadCount )
{
    if( bounded )
    {
        const size_t p1HeapSize = K32BoundedPhase1::GetRequiredSize( numBuckets, pairsBlockSize, fxBlockSize, threadCount );
        const size_t p3HeapSize = DiskPlotPhase3::GetRequiredHeapSize( numBuckets, bounded, pairsBlockSize, fxBlockSize );

        return std::max( p1HeapSize, p3HeapSize );
    }

    switch( numBuckets )
    {
        case 128 : return DiskFp<TableId::Table4, 128 >::GetRequiredHeapSize( fxBlockSize, pairsBlockSize );
        case 256 : return DiskFp<TableId::Table4, 256 >::GetRequiredHeapSize( fxBlockSize, pairsBlockSize );
        case 512 : return DiskFp<TableId::Table4, 512 >::GetRequiredHeapSize( fxBlockSize, pairsBlockSize );
        case 1024:
            Fatal( "1024 buckets are currently unsupported." );
            return 0;
            // We need to add a bit more here (at least 1GiB) to have enough space for P2, which keeps
            // 2 marking table bitfields in-memory: k^32 / 8 = 0.5GiB
//            return 1032ull MB + DiskFp<TableId::Table4, 1024>::GetRequiredHeapSize( fxBlockSize, pairsBlockSize );

    default:
        break;
    }

    Fatal( "Invalid bucket size: %u.", numBuckets );
    return 0;
}


//-----------------------------------------------------------
size_t ValidateTmpPathAndGetBlockSize( DiskPlotter::Config& cfg )
{
    FatalIf( cfg.tmpPath == nullptr, "No temporary path specified." );

    size_t pathLen = strlen( cfg.tmpPath );
    FatalIf( pathLen < 1, "Invalid temporary path." );

    char* tmpPath = bbmalloc<char>( pathLen + 2 );
    memcpy( tmpPath, cfg.tmpPath, pathLen );

    if( cfg.tmpPath[pathLen - 1] != '/'
    #ifdef _WIN32
        && cfg.tmpPath[pathLen - 1] != '\\'
    #endif
        )
    {
        tmpPath[pathLen++] = '/';
    }

    tmpPath[pathLen] = (char)0;
    cfg.tmpPath = tmpPath;

    // Open a file in the temp dir to obtain the block size
    uint64 randNum = 0;
    SysHost::Random( (byte*)&randNum, sizeof( randNum ) );

    char* randFileName = bbmalloc<char>( pathLen + 32 );
    
    int r = snprintf( randFileName, pathLen + 32, "%s.%llx.blk", tmpPath, (llu)randNum );
    FatalIf( r < 1, "Unexpected error validating temp directory." );

    FileStream tmpFile;

    if( !tmpFile.Open( randFileName, FileMode::Create, FileAccess::ReadWrite ) )
    {
        int err = tmpFile.GetError();
        Fatal( "Failed to open a file in the temp directory with error %d (0x%x).", err, err );
    }
    

    cfg.expectedTmpDirBlockSize = tmpFile.BlockSize();

    remove( randFileName );
    free( randFileName );

    return cfg.expectedTmpDirBlockSize;
}


static const char* USAGE = R"(diskplot [OPTIONS] <out_dir>

Creates plots by making use of a disk to temporarily store and read values.

<out_dir> : The output directory where the plot will be copied to after completion.

[OPTIONS]
 -b, --buckets <n>  : The number of buckets to use. The default is 256.
                      You may specify one of: 128, 256, 512, 1024 and 64 for if --k32-bounded is enabled.
                      1024 is not available for plots of k < 33.

 --unbounded        : Create an unbounded k32 plot. That is a plot that does not cut-off entries that 
                      overflow 2^32;
 
 -a, --alternate    : Halves the temp2 cache size requirements by alternating bucket writing methods
                      between tables.

 -t1, --temp1 <dir> : The temporary directory to use when plotting.
                      *REQUIRED*

 -t2, --temp2 <dir> : Specify a secondary temporary directory, which will be used for data
                      that needs to be read/written from constantly.
                      If nothing is specified, --temp will be used instead.

 --no-t1-direct     : Disable direct I/O on the temp 1 directory.

 --no-t2-direct     : Disable direct I/O on the temp 2 directory.

 -s, --sizes        : Output the memory requirements for a specific bucket count.
                      To change the bucket count from the default, pass a value to -b
                      before using this argument. You may also pass a value to --temp and --temp2
                      to get file system block-aligned values when using direct IO.

 --cache <n>        : Size of cache to reserve for I/O. This is memory
                      reserved for files that incur frequent I/O.
                      You need about 192GiB(+|-) for high-frequency I/O Phase 1 calculations
                      to be completely in-memory.

 --f1-threads <n>   : Override the thread count for F1 generation.

 --fp-threads <n>   : Override the thread count for forward propagation.

 --c-threads <n>    : Override the thread count for C table processing.
                      (Equivalent to Phase 4 in chiapos, but performed 
                      at the end of Phase 1.)

--p2-threads <n>    : Override the thread count for Phase 2.

--p3-threads <n>    : Override the thread count for Phase 3.

-h, --help          : Print this help text and exit.


[NOTES]
If you don't specify any thread count overrides, the default thread count
specified in the global options will be used
(specified as -t <thread_count> before the diskplot command).

Phases 2 and 3 are typically more I/O bound than Phase 1 as these
phases perform less computational work than Phase 1 and thus the CPU
finishes the currently loaded workload quicker and will proceed to
grab another buffer from the disk within a shorter time frame. Because of this
you would typically lower the thread count for these phases if you are
incurring high I/O waits.

[EXAMPLES]
# Simple config:
bladebit -t 24 -f <farmer_pub_key> -c <contract_address> diskplot -t1 /my/temporary/plot/dir /my/output/dir

# With fine-grained control over threads per phase/section (see bladebit -h diskplot):
bladebit -t 30 -f <farmer_pub_key> -c <contract_address> diskplot --f1-threads 16 --c-threads 16 --p2-threads 8 -t1 /my/temporary/plot/dir /my/output/dir
)";

//-----------------------------------------------------------
void DiskPlotter::PrintUsage()
{
    Log::Line( USAGE );
}

