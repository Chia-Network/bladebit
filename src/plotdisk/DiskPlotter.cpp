#include "DiskPlotter.h"
#include "util/Log.h"
#include "util/Util.h"
#include "util/CliParser.h"
#include "util/jobs/MemJobs.h"
#include "io/FileStream.h"

#include "DiskFp.h"
#include "DiskPlotPhase1.h"
// #include "DiskPlotPhase2.h"
// #include "DiskPlotPhase3.h"
#include "SysHost.h"


size_t ValidateTmpPathAndGetBlockSize( DiskPlotter::Config& cfg );


//-----------------------------------------------------------
// DiskPlotter::DiskPlotter()
// {
// }

//-----------------------------------------------------------
DiskPlotter::DiskPlotter( const Config cfg )
{
    ASSERT( cfg.tmpPath  );
    ASSERT( cfg.tmpPath2 );

    // Initialize tables for matching
    LoadLTargets();
    
    GlobalPlotConfig& gCfg = *cfg.globalCfg;

    ZeroMem( &_cx );
    
    FatalIf( !GetTmpPathsBlockSizes( cfg.tmpPath, cfg.tmpPath2, _cx.tmp1BlockSize, _cx.tmp2BlockSize ),
        "Failed to obtain temp paths block size." );

    const size_t heapSize = GetRequiredSizeForBuckets( cfg.numBuckets, _cx.tmp1BlockSize, _cx.tmp2BlockSize );

    _cx.tmpPath     = cfg.tmpPath;
    _cx.tmpPath2    = cfg.tmpPath2;
    _cx.numBuckets  = cfg.numBuckets;
    _cx.heapSize    = heapSize;
    _cx.cacheSize   = cfg.cacheSize;
    _cx.useDirectIO = cfg.enableDirectIO;

    const uint sysLogicalCoreCount = SysHost::GetLogicalCPUCount();

    // _cx.threadCount   = gCfg.threadCount;
    _cx.ioThreadCount = cfg.ioThreadCount;
    _cx.f1ThreadCount = cfg.f1ThreadCount == 0 ? gCfg.threadCount : std::min( cfg.f1ThreadCount, sysLogicalCoreCount );
    _cx.fpThreadCount = cfg.fpThreadCount == 0 ? gCfg.threadCount : std::min( cfg.fpThreadCount, sysLogicalCoreCount );
    _cx.cThreadCount  = cfg.cThreadCount  == 0 ? gCfg.threadCount : std::min( cfg.cThreadCount , sysLogicalCoreCount );
    _cx.p2ThreadCount = cfg.p2ThreadCount == 0 ? gCfg.threadCount : std::min( cfg.p2ThreadCount, sysLogicalCoreCount );
    _cx.p3ThreadCount = cfg.p3ThreadCount == 0 ? gCfg.threadCount : std::min( cfg.p3ThreadCount, sysLogicalCoreCount );

    Log::Line( "[Bladebit Disk PLotter]" );
    Log::Line( " Heap size      : %.2lf GiB ( %.2lf MiB )", (double)_cx.heapSize BtoGB, (double)_cx.heapSize BtoMB );
    Log::Line( " Cache size     : %.2lf GiB ( %.2lf MiB )", (double)_cx.cacheSize BtoGB, (double)_cx.cacheSize BtoMB );
    Log::Line( " Bucket count   : %u"       , _cx.numBuckets    );
    Log::Line( " F1 threads     : %u"       , _cx.f1ThreadCount );
    Log::Line( " FP threads     : %u"       , _cx.fpThreadCount );
    Log::Line( " C  threads     : %u"       , _cx.cThreadCount  );
    Log::Line( " P2 threads     : %u"       , _cx.p2ThreadCount );
    Log::Line( " P3 threads     : %u"       , _cx.p3ThreadCount );
    Log::Line( " IO threads     : %u"       , _cx.ioThreadCount );
    Log::Line( " Unbuffered IO  : %s"       , _cx.useDirectIO ? "true" : "false" );

    Log::Line( " Allocating memory" );
    _cx.heapBuffer = (byte*)SysHost::VirtualAlloc( _cx.heapSize );
    FatalIf( !_cx.heapBuffer, "Failed to allocated heap buffer. Make sure you have enough free memory." );

    if( _cx.cacheSize )
        _cx.cache = bbvirtalloc<byte>( _cx.cacheSize );
  
    // Initialize our Thread Pool and IO Queue
    _cx.threadPool = new ThreadPool( sysLogicalCoreCount, ThreadPool::Mode::Fixed, gCfg.disableCpuAffinity );
    _cx.ioQueue    = new DiskBufferQueue( _cx.tmpPath, _cx.heapBuffer, _cx.heapSize, _cx.ioThreadCount, _cx.useDirectIO );

    if( cfg.globalCfg->warmStart )
    {
        Log::Line( "Warm start: Pre-faulting memory pages..." );

        const uint32 threadCount = cfg.globalCfg->threadCount == 0 ? sysLogicalCoreCount :
                                    std::min( cfg.globalCfg->threadCount, sysLogicalCoreCount );

        FaultMemoryPages::RunJob( *_cx.threadPool, threadCount, _cx.heapBuffer, _cx.heapSize );
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
    // #TODO: Reset the rest of the state, including the heap & the ioQueue


    Log::Line( "Started plot." );
    auto plotTimer = TimerBegin();

    _cx.plotId       = req.plotId;
    _cx.plotMemo     = req.plotMemo;
    _cx.plotMemoSize = req.plotMemoSize;

    _cx.ioQueue->OpenPlotFile( req.plotFileName, req.plotId, req.plotMemo, req.plotMemoSize );

    // #TODO: I think we can get rid of this structure.
    //        If not, place it on the context.
    // Phase3Data p3Data;
    // ZeroMem( &p3Data );

    {
        Log::Line( "Running Phase 1" );
        const auto timer = TimerBegin();

        DiskPlotPhase1 phase1( _cx );
        phase1.Run();

        const double elapsed = TimerEnd( timer );
        Log::Line( "Finished Phase 1 in %.2lf seconds ( %.2lf minutes ).", elapsed, elapsed / 60 );
    }

    // {
    //     Log::Line( "Running Phase 2" );
    //     const auto timer = TimerBegin();

    //     DiskPlotPhase2 phase2( _cx );
    //     phase2.Run();

    //     const double elapsed = TimerEnd( timer );
    //     Log::Line( "Finished Phase 2 in %.2lf seconds ( %.2lf minutes ).", elapsed, elapsed / 60 );

    //     p3Data = phase2.GetPhase3Data();
    // }

    // {
    //     Log::Line( "Running Phase 3" );
    //     const auto timer = TimerBegin();

    //     DiskPlotPhase3 phase3( _cx, p3Data );
    //     phase3.Run();

    //     const double elapsed = TimerEnd( timer );
    //     Log::Line( "Finished Phase 3 in %.2lf seconds ( %.2lf minutes ).", elapsed, elapsed / 60 );
    // }

    // {
    //     // Now we need to update the table sizes on the file
    //     Log::Line( "Waiting for plot file to complete pending writes..." );
    //     const auto timer = TimerBegin();

    //     // Update the table pointers location
    //     DiskBufferQueue& ioQueue = *_cx.ioQueue;
    //     ASSERT( sizeof( _cx.plotTablePointers ) == sizeof( uint64 ) * 10 );

    //     // Convert them to big endian
    //     for( int i = 0; i < 10; i++ )
    //         _cx.plotTablePointers[i] = Swap64( _cx.plotTablePointers[i] );

    //     const int64 tablePtrsStart = (int64)ioQueue.PlotTablePointersAddress();
    //     ioQueue.SeekFile( FileId::PLOT, 0, tablePtrsStart, SeekOrigin::Begin );
    //     ioQueue.WriteFile( FileId::PLOT, 0, _cx.plotTablePointers, sizeof( _cx.plotTablePointers ) );
        
    //     // Wait for all IO commands to finish
    //     Fence fence;
    //     ioQueue.SignalFence( fence );
    //     ioQueue.CommitCommands();
    //     fence.Wait();
        
    //     const double elapsed = TimerEnd( timer );
    //     Log::Line( "Completed pending writes in %.2lf seconds.", elapsed );
    //     Log::Line( "Finished writing plot %s.", req.plotFileName );
    //     Log::Line( "Final plot table pointers: " );

    //     for( int i = 0; i < 10; i++ )
    //     {
    //         const uint64 addy = Swap64( _cx.plotTablePointers[i] );

    //         if( i < 7 )
    //             Log::Line( " Table %d: %16lu ( 0x%016lx )", i+1, addy, addy );
    //         else
    //             Log::Line( " C %d    : %16lu ( 0x%016lx )", i-6, addy, addy );
    //     }
    //     Log::Line( "" );
    // }

    double plotElapsed = TimerEnd( plotTimer );
    Log::Line( "Finished plotting in %.2lf seconds ( %.2lf minutes ).", plotElapsed, plotElapsed / 60 );
}

//-----------------------------------------------------------
void DiskPlotter::ParseCommandLine( CliParser& cli, Config& cfg )
{
    while( cli.HasArgs() )
    {
        if( cli.ReadValue( cfg.numBuckets,  "-b", "--buckets" ) ) 
            continue;
        if( cli.ReadValue( cfg.tmpPath, "-t", "--temp" ) )
            continue;
        if( cli.ReadValue( cfg.tmpPath2, "-t2", "--temp2" ) )
            continue;
        if( cli.ReadValue( cfg.cacheSize, "--cache" ) )
            continue;
        if( cli.ReadValue( cfg.f1ThreadCount, "--f1-threads" ) )
            continue;
        if( cli.ReadValue( cfg.fpThreadCount, "--fp-threads" ) )
            continue;
        if( cli.ReadValue( cfg.cThreadCount, "--c-threads" ) )
            continue;
        if( cli.ReadValue( cfg.p2ThreadCount, "--p2-threads" ) )
            continue;
        if( cli.ReadValue( cfg.p3ThreadCount, "--p3-threads" ) )
            continue;
        if( cli.ArgConsume( "-s", "--sizes" ) )
        {
            FatalIf( cfg.numBuckets < BB_DP_MIN_BUCKET_COUNT || cfg.numBuckets > BB_DP_MAX_BUCKET_COUNT,
                "Buckets must be between %u and %u, inclusive.", (uint)BB_DP_MIN_BUCKET_COUNT, (uint)BB_DP_MAX_BUCKET_COUNT );
            FatalIf( ( cfg.numBuckets & ( cfg.numBuckets - 1 ) ) != 0, "Buckets must be power of 2." );

            size_t heapSize = 0;
            if( cfg.tmpPath )
            {
                cfg.tmpPath2 = cfg.tmpPath2 ? cfg.tmpPath2 : cfg.tmpPath;
                heapSize = GetRequiredSizeForBuckets( cfg.numBuckets, cfg.tmpPath2, cfg.tmpPath );
            }
            else
                heapSize = GetRequiredSizeForBuckets( cfg.numBuckets, 1, 1 );
                
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
            break;
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
    ASSERT( tmpPath1 );
    ASSERT( tmpPath2 );

    bool success = false;

    const char*  paths[2]  = { tmpPath1, tmpPath2 };
    const size_t lengths[2] = { 
        strlen( tmpPath1 ),
        strlen( tmpPath2 ),
    };

    const size_t RAND_PART     =  16;
    const size_t RAND_FILE_SIZE = RAND_PART + 4;    // 5 = '.' + ".tmp"
    const size_t MAX_LENGTH     = 1024 + RAND_FILE_SIZE + 1;
    char stackPath[MAX_LENGTH+1];

    const size_t pathLength = std::max( lengths[0], lengths[1] ) + RAND_FILE_SIZE + 1;

    char* path = nullptr;
    if( pathLength > MAX_LENGTH )
        path = bbmalloc<char>( pathLength + RAND_FILE_SIZE + 2 ); // +2 = '/' + '\0'
    else
        path = stackPath;

    size_t blockSizes[2] = { 0 };

    for( int32 i = 0; i < 2; i++ )
    {
        size_t len = lengths[i];
        memcpy( path, paths[i], len );

        if( path[len-1] != '/' && path[len-1] != '\\' )
            path[len++] = '/';
    
        path[len++] = '.';

        byte filename[RAND_PART/2];
        SysHost::Random( filename, sizeof( filename ) );

        size_t encoded;
        if( BytesToHexStr( filename, sizeof( filename ), path+len, RAND_PART, encoded ) != 0 )
        {
            Log::Error( "GetTmpPathsBlockSizes: Hex conversion failed." );
            goto EXIT;
        }

        len += RAND_PART;
        memcpy( path+len, ".tmp", sizeof( ".tmp" ) );

        #if _DEBUG
            if( path == stackPath )
                ASSERT( path+len+ sizeof( ".tmp" ) <= stackPath + sizeof( stackPath ) );
        #endif

        FileStream file;
        if( !file.Open( path, FileMode::Create, FileAccess::ReadWrite ) )
        {
            Log::Error( "GetTmpPathsBlockSizes: Failed to open temp file '%s'.", path );
            goto EXIT;
        }

        blockSizes[i] = file.BlockSize();
        file.Close();

        remove( path );
    }

    tmpPath1Size = blockSizes[0];
    tmpPath2Size = blockSizes[1];
    success = true;

EXIT:
    if( path && path != stackPath )
        free( path );

    return success;
}

//-----------------------------------------------------------
size_t DiskPlotter::GetRequiredSizeForBuckets( const uint32 numBuckets, const char* tmpPath1, const char* tmpPath2 )
{
    size_t blockSizes[2] = { 0 };

    if( !GetTmpPathsBlockSizes( tmpPath1, tmpPath2, blockSizes[0], blockSizes[1] ) )
        return 0;

    return GetRequiredSizeForBuckets( numBuckets, blockSizes[0], blockSizes[1] );
}

//-----------------------------------------------------------
size_t DiskPlotter::GetRequiredSizeForBuckets( const uint32 numBuckets, const size_t fxBlockSize, const size_t pairsBlockSize )
{
    switch( numBuckets )
    {
        case 128 : return DiskFp<TableId::Table4, 128 >::GetRequiredHeapSize( fxBlockSize, pairsBlockSize );
        case 256 : return DiskFp<TableId::Table4, 256 >::GetRequiredHeapSize( fxBlockSize, pairsBlockSize );
        case 512 : return DiskFp<TableId::Table4, 512 >::GetRequiredHeapSize( fxBlockSize, pairsBlockSize );
        case 1024: return DiskFp<TableId::Table4, 1024>::GetRequiredHeapSize( fxBlockSize, pairsBlockSize );
    
    default:
        Fatal( "Invalid bucket size: %u.", numBuckets );
        break;
    }
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
    
    int r = snprintf( randFileName, pathLen + 32, "%s.%llx.blk", tmpPath, randNum );
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

Creates a plots by making use of a disk to temporarily store and read values.

<out_dir> : The output directory where the plot will be copied to after completion.

[OPTIONS]
 -b, --buckets <n>  : The number of buckets to use. The default is 256.
                      You may specify one of: 128, 256, 512, 1024.

 -t, --temp <dir>   : The temporary directory to use when plotting.
                      *REQUIRED*

 -t2, --temp2 <dir> : Specify a secondary temporary directory, which will be used for data
                      that needs to be read/written from constantly.
                      If nothing is specified, --temp will be used instead.

 -s, --sizes        : Output the memory requirements for a specific bucket count.
                      To change the bucket count from the default, pass a value to -b
                      before using this argument. You may also pass a value to --temp and --temp2
                      to get file system block-aligned values when using direct IO.

 --cache <n>        : Size of cache to reserve for IO. This is memory
                      reserved for files that incurr frequent I/O.
                      You need about 96GiB for high-performance Phase 1 calculations.

 --f1-threads <n>   : Override the thread count for F1 generation.

 --fp-threads <n>   : Override the thread count for forwrd propagation.

 --c-threads <n>    : Override the thread count for C table processing.
                      (Equivalent to Phase 4 in chiapos, but performed 
                      at the end of Phase 1.)

--p2-threads <n>    : Override the thread count for Phase 2.

--p3-threads <n>    : Override the thread count for Phase 3.

-h, --help          : Print this help text and exit.


[NOTES]
If you don't specify any thread count overrides, the default thread count
specified in the global options will be used.

Phases 2 and 3 are typically more I/O bound that Phase 1 as these
phases perform less computational work than Phase 1 and thus the CPU
finishes the currently loaded workload quicker and will proceed to
grab another buffer from disk with a shorter frequency. Because of this
you would typically lower the thread count for this threads if you are
incurring I/O waits.

[EXAMPLES]
bladebit -t 24 -f ... -c ... diskplot --f1 256MB --fx 256MB -t /my/temporary/plot/dir
 --f1-threads 3 --c-threads 8 --p2-threads 12 --p3-threads 8 /my/output/dir

bladebit -t 8 -f ... -c ... diskplot --f1 64MB --fx 128MB -t /my/temporary/plot/dir /my/output/dir
)";

//-----------------------------------------------------------
void DiskPlotter::PrintUsage()
{
    Log::Line( USAGE );
}