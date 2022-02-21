#include "DiskPlotter.h"
#include "util/Log.h"
#include "util/Util.h"
#include "util/CliParser.h"

#include "DiskPlotPhase1.h"
#include "DiskPlotPhase2.h"
#include "DiskPlotPhase3.h"
#include "SysHost.h"

// #TEST
// #TODO: Remove
#include "sandbox/Sandbox.h"


size_t ValidateTmpPathAndGetBlockSize( DiskPlotter::Config& cfg );


//-----------------------------------------------------------
// DiskPlotter::DiskPlotter()
// {
// }

//-----------------------------------------------------------
DiskPlotter::DiskPlotter( const Config cfg )
{
    // Initialize tables for matching
    LoadLTargets();
    
    ZeroMem( &_cx );

    ASSERT( cfg.tmpPath );

    GlobalPlotConfig& gCfg = *cfg.globalCfg;
    
    const size_t bucketsCountsSize = RoundUpToNextBoundaryT( BB_DP_BUCKET_COUNT * sizeof( uint32 ), cfg.expectedTmpDirBlockSize );
    const uint32 ioBufferCount     = cfg.ioBufferCount;
    const size_t ioHeapFullSize    = ( cfg.ioBufferSize + bucketsCountsSize ) * ioBufferCount;

    GetHeapRequiredSize( _fpBufferSizes, cfg.expectedTmpDirBlockSize, gCfg.threadCount );

    _cx.bufferSizes   = &_fpBufferSizes;
    _cx.tmpPath       = cfg.tmpPath;
    _cx.heapSize      = _fpBufferSizes.totalSize;
    _cx.ioBufferSize  = cfg.ioBufferSize;
    _cx.ioHeapSize    = ioHeapFullSize;
    _cx.ioBufferCount = ioBufferCount;
    _cx.useDirectIO   = cfg.enableDirectIO;
    _cx.totalHeapSize = _cx.heapSize + _cx.ioHeapSize;

    _cx.threadCount   = gCfg.threadCount;
    _cx.ioThreadCount = cfg.ioThreadCount;

    static_assert( sizeof( DiskPlotContext::writeIntervals ) == sizeof( Config::writeIntervals ), "Write interval array sizes do not match." );
    memcpy( _cx.writeIntervals, cfg.writeIntervals, sizeof( _cx.writeIntervals ) );

    Log::Line( "[Disk PLotter]" );
    Log::Line( " Work Heap size : %.2lf MiB", (double)_cx.heapSize BtoMB );
    Log::Line( " Work threads   : %u"       , _cx.threadCount   );
    Log::Line( " IO threads     : %u"       , _cx.ioThreadCount );
    Log::Line( " IO buffer size : %llu MiB (%llu MiB total)", _cx.ioBufferSize BtoMB, _cx.ioBufferSize * _cx.ioBufferCount BtoMB );
    Log::Line( " IO buffer count: %u"       , _cx.ioBufferCount );
    Log::Line( " Unbuffered IO  : %s"       , _cx.useDirectIO ? "true" : "false" );

    // Log::Line( "Allocating heap of %llu MiB.", _cx.heapSize BtoMB );
    Log::Line( " Allocating a heap of %llu MiB.", _cx.totalHeapSize BtoMB );
    _cx.heapBuffer = (byte*)SysHost::VirtualAlloc( _cx.totalHeapSize );
    FatalIf( !_cx.heapBuffer, "Failed to allocated heap buffer. Make sure you have enough free memory." );
    
    _cx.ioHeap = _cx.heapBuffer + _cx.heapSize;

    // #TODO: Check for warm start

    // Initialize our Thread Pool and IO Queue
    _cx.threadPool = new ThreadPool( _cx.threadCount, ThreadPool::Mode::Fixed, gCfg.disableCpuAffinity );
    _cx.ioQueue    = new DiskBufferQueue( _cx.tmpPath, _cx.heapBuffer, _cx.totalHeapSize, _cx.ioThreadCount, _cx.useDirectIO );
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
    Phase3Data p3Data;
    ZeroMem( &p3Data );

    {
        Log::Line( "Running Phase 1" );
        const auto timer = TimerBegin();

        DiskPlotPhase1 phase1( _cx );
        phase1.Run();

        const double elapsed = TimerEnd( timer );
        Log::Line( "Finished Phase 1 in %.2lf seconds ( %.2lf minutes ).", elapsed, elapsed / 60 );
    }

    {
        Log::Line( "Running Phase 2" );
        const auto timer = TimerBegin();

        DiskPlotPhase2 phase2( _cx );
        phase2.Run();

        const double elapsed = TimerEnd( timer );
        Log::Line( "Finished Phase 2 in %.2lf seconds ( %.2lf minutes ).", elapsed, elapsed / 60 );

        p3Data = phase2.GetPhase3Data();
    }

    {
        Log::Line( "Running Phase 3" );
        const auto timer = TimerBegin();

        DiskPlotPhase3 phase3( _cx, p3Data );
        phase3.Run();

        const double elapsed = TimerEnd( timer );
        Log::Line( "Finished Phase 3 in %.2lf seconds ( %.2lf minutes ).", elapsed, elapsed / 60 );
    }

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
        Fence fence;
        ioQueue.SignalFence( fence );
        ioQueue.CommitCommands();
        fence.Wait();
        
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
    }

    double plotElapsed = TimerEnd( plotTimer );
    Log::Line( "Finished plotting in %.2lf seconds ( %.2lf minutes ).", plotElapsed, plotElapsed / 60 );
}

//-----------------------------------------------------------
void DiskPlotter::ParseCommandLine( CliParser& cli, Config& cfg )
{
    // #TODO: Have these defined in the config ehader
    const size_t f1DefaultWriteInterval    = 128ull MB;
    size_t       fxDefaultWriteInterval    = 64ull  MB;
    size_t       matchDefaultWriteInterval = 64ull  MB;
    const uint   minBufferCount            = 3;

    // Parse fx and match per-table
    auto checkFx = [&]() {
        
        const char*  a       = cli.Arg();
        const size_t minSize = sizeof( "--fx" ) - 1;
        const size_t len     = strlen( a );

        if( len >= minSize && memcmp( "--fx", a, minSize ) == 0 )
        {
            if( len == minSize )
            {
                // Set the default value
                cli.NextArg();
                fxDefaultWriteInterval = cli.ReadSize( "--fx" );
                return true;
            }
            else
            {
                // Set the value for a table (--fx2, --fx4...)
                const char intChar = a[minSize];
                
                // Expect an integer from 2-7 (inclusive) to come immediately after --fx
                if( intChar >= '2' && intChar <= '7' )
                {
                    const int tableId = (int)intChar - '0';
                    cfg.writeIntervals[tableId].fxGen = cli.ReadSize( "--f[n]" );
                    
                    return true;
                }
            }
        }

        return false;
    };

    while( cli.HasArgs() )
    {
        if( cli.ReadValue( cfg.writeIntervals[0].fxGen, "--f1" ) ) 
            continue;
        if( cli.ReadValue( cfg.ioBufferCount, "-b", "--buffer-count" ) ) 
            continue;
        if( cli.ReadValue( cfg.tmpPath, "-t", "--temp" ) )
            continue;
        if( checkFx() )
            continue;
        else
            break;
    }

    // Validate some parameters
    cfg.ioBufferCount = std::max( minBufferCount, cfg.ioBufferCount );

    const size_t diskBlockSize = ValidateTmpPathAndGetBlockSize( cfg );

    const size_t minBucketSize = BB_DP_MAX_ENTRIES_PER_BUCKET * sizeof( uint32 );

    size_t maxWriteInterval = 0;

    for( TableId table = TableId::Table1; table < TableId::_Count; table++ )
    {
        auto& writeInterval = cfg.writeIntervals[(int)table];

        if( writeInterval.fxGen == 0 )
            writeInterval.fxGen = table == TableId::Table1 ? f1DefaultWriteInterval : fxDefaultWriteInterval;

        if( writeInterval.matching == 0 )
            writeInterval.matching = matchDefaultWriteInterval;

        // Ensure the intervals are <= than the minimum write size of a bucket (table 7)
        // and >= disk block size of the temporary directory.
        FatalIf( writeInterval.fxGen    > minBucketSize, "f%d write interval must be less or equal than %llu bytes.", (int)table+1, minBucketSize );
        FatalIf( writeInterval.matching > minBucketSize, "Table %d match write interval must be less or equal than %llu bytes.", (int)table+1, minBucketSize );
        FatalIf( writeInterval.fxGen    < diskBlockSize, "f%d write interval must be greater or equal than the tmp directory block size of %llu bytes.", (int)table+1, diskBlockSize );
        FatalIf( writeInterval.matching < diskBlockSize, "Table %d match write interval must be greater or equal than the tmp directory block size of %llu bytes.", (int)table + 1, minBucketSize );

        // Round up the size to the block size
        writeInterval.fxGen    = RoundUpToNextBoundaryT( writeInterval.fxGen   , diskBlockSize );
        writeInterval.matching = RoundUpToNextBoundaryT( writeInterval.matching, diskBlockSize );

        maxWriteInterval = std::max( maxWriteInterval, writeInterval.fxGen    );
        maxWriteInterval = std::max( maxWriteInterval, writeInterval.matching );
    }

    cfg.ioBufferSize = maxWriteInterval;

    FatalIf( cfg.ioBufferCount < 3, "IO buffer (write interval buffers) cont must be 3 or more." );
    

    const uint sysLogicalCoreCount = SysHost::GetLogicalCPUCount();

    if( cfg.ioThreadCount == 0 )
        cfg.ioThreadCount = 1;        // #TODO: figure out a reasonable default. Probably 1 or 2 for current consumer NVMes running on PCIe3...
    else if( cfg.ioThreadCount > sysLogicalCoreCount )
    {
        Log::Line( "Warning: Limiting disk queue threads to %u, which is the system's logical CPU count.", sysLogicalCoreCount );
        cfg.ioThreadCount = sysLogicalCoreCount;
    }
}

//-----------------------------------------------------------
void DiskPlotter::GetHeapRequiredSize( DiskFPBufferSizes& sizes, const size_t fileBlockSize, const uint threadCount )
{
    ZeroMem( &sizes );

    const uint maxBucketEntries = BB_DP_MAX_ENTRIES_PER_BUCKET;

    sizes.fileBlockSize = fileBlockSize;

    const size_t ySize       = RoundUpToNextBoundaryT( maxBucketEntries * sizeof( uint32 ), fileBlockSize );
    const size_t sortKeySize = RoundUpToNextBoundaryT( maxBucketEntries * sizeof( uint32 ), fileBlockSize );
    const size_t mapSize     = RoundUpToNextBoundaryT( maxBucketEntries * sizeof( uint64 ), fileBlockSize );
    const size_t metaSize    = RoundUpToNextBoundaryT( maxBucketEntries * sizeof( uint64 ), fileBlockSize );
    const size_t pairsLSize  = RoundUpToNextBoundaryT( maxBucketEntries * sizeof( uint32 ), fileBlockSize );
    const size_t pairsRSize  = RoundUpToNextBoundaryT( maxBucketEntries * sizeof( uint16 ), fileBlockSize );

    const size_t blockAlignedOverflowSize = fileBlockSize * BB_DP_BUCKET_COUNT * 2;

    sizes.yIO              = ySize       * 2;
    sizes.sortKeyIO        = sortKeySize * 2;
    sizes.mapIO            = mapSize;
    sizes.metaAIO          = metaSize    * 2;
    sizes.metaBIO          = metaSize    * 2;
    sizes.pairsLeftIO      = pairsLSize;
    sizes.pairsRightIO     = pairsRSize;

    sizes.groupsSize       = sizeof( uint32 ) * BB_DP_MAX_BC_GROUP_PER_BUCKET;
    sizes.yTemp            = ySize;
    sizes.metaATmp         = metaSize;
    sizes.metaBTmp         = metaSize;

    sizes.yOverflow        = blockAlignedOverflowSize;
    sizes.mapOverflow      = blockAlignedOverflowSize;
    sizes.pairOverflow     = blockAlignedOverflowSize * 2;
    sizes.metaAOverflow    = blockAlignedOverflowSize;
    sizes.metaBOverflow    = blockAlignedOverflowSize;

    sizes.crossBucketY          = sizeof(uint32) * ( kBC * 6 );
    sizes.crossBucketMetaA      = sizeof(uint64) * ( kBC * 6 );
    sizes.crossBucketMetaB      = sizeof(uint64) * ( kBC * 6 );
    sizes.crossBucketPairsLeft  = RoundUpToNextBoundaryT( sizeof(uint32) * (size_t)kBC, fileBlockSize );
    sizes.crossBucketPairsRight = RoundUpToNextBoundaryT( sizeof(uint16) * (size_t)kBC, fileBlockSize );
    sizes.crossBucketTotal      =
        sizes.crossBucketY          +
        sizes.crossBucketMetaA      +
        sizes.crossBucketMetaB      +
        sizes.crossBucketPairsLeft  +
        sizes.crossBucketPairsRight;

    sizes.totalSize = 
        sizes.yIO           +
        sizes.sortKeyIO     +
        sizes.mapIO         +
        sizes.metaAIO       +
        sizes.metaBIO       +
        sizes.pairsLeftIO   +
        sizes.pairsRightIO  +
        sizes.groupsSize    +
        sizes.yTemp         +
        sizes.metaATmp      +
        sizes.metaBTmp      +
        sizes.yOverflow     +
        sizes.mapOverflow   +
        sizes.pairOverflow  +
        sizes.metaAOverflow +
        sizes.metaBOverflow +
        sizes.crossBucketTotal;
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


