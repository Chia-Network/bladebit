#include "DiskPlotter.h"
#include "util/Log.h"
#include "Util.h"

#include "DiskPlotPhase1.h"
#include "DiskPlotPhase2.h"
#include "DiskPlotPhase3.h"
#include "SysHost.h"

// #TEST
// #TODO: Remove
#include "sandbox/Sandbox.h"

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
    
    const size_t bucketsCountsSize = RoundUpToNextBoundaryT( BB_DP_BUCKET_COUNT * sizeof( uint32 ), cfg.expectedTmpDirBlockSize );
    const uint32 ioBufferCount     = cfg.ioBufferCount;
    const size_t ioHeapFullSize    = ( cfg.ioBufferSize + bucketsCountsSize ) * ioBufferCount;

    GetHeapRequiredSize( _fpBufferSizes, cfg.expectedTmpDirBlockSize, cfg.workThreadCount );

    _cx.bufferSizes   = &_fpBufferSizes;
    _cx.tmpPath       = cfg.tmpPath;
    _cx.heapSize      = _fpBufferSizes.totalSize;
    _cx.ioBufferSize  = cfg.ioBufferSize;
    _cx.ioHeapSize    = ioHeapFullSize;
    _cx.ioBufferCount = ioBufferCount;
    _cx.useDirectIO   = cfg.enableDirectIO;

    _cx.threadCount   = cfg.workThreadCount;
    _cx.ioThreadCount = cfg.ioThreadCount;

    static_assert( sizeof( DiskPlotContext::writeIntervals ) == sizeof( Config::writeIntervals ), "Write interval array sizes do not match." );
    memcpy( _cx.writeIntervals, cfg.writeIntervals, sizeof( _cx.writeIntervals ) );

    Log::Line( "Work Heap size : %.2lf MiB", (double)_cx.heapSize BtoMB );
    Log::Line( "Work threads   : %u"       , _cx.threadCount   );
    Log::Line( "IO threads     : %u"       , _cx.ioThreadCount );
    Log::Line( "IO buffer size : %llu MiB (%llu MiB total)", _cx.ioBufferSize BtoMB, _cx.ioBufferSize * _cx.ioBufferCount BtoMB );
    Log::Line( "IO buffer count: %u"       , _cx.ioBufferCount );
    Log::Line( "Unbuffered IO  : %s"       , _cx.useDirectIO ? "true" : "false" );

    const size_t allocationSize = _cx.heapSize + _cx.ioHeapSize;

    // Log::Line( "Allocating heap of %llu MiB.", _cx.heapSize BtoMB );
    Log::Line( "Allocating a heap of %llu MiB.", allocationSize BtoMB );
    _cx.heapBuffer = (byte*)SysHost::VirtualAlloc( allocationSize );
    FatalIf( !_cx.heapBuffer, "Failed to allocated heap buffer. Make sure you have enough free memory." );
    
    _cx.ioHeap = _cx.heapBuffer + _cx.heapSize;
    // Log::Line( "Allocating IO buffers." );
    // _cx.ioHeap = (byte*)SysHost::VirtualAlloc( ioHeapFullSize );
    // FatalIf( !_cx.ioHeap, "Failed to allocated work buffer. Make sure you have enough free memory." );

    // Initialize our Thread Pool and IO Queue
    _cx.threadPool = new ThreadPool( _cx.threadCount, ThreadPool::Mode::Fixed, false );
    _cx.ioQueue    = new DiskBufferQueue( _cx.tmpPath, _cx.heapBuffer, allocationSize, _cx.ioThreadCount, _cx.useDirectIO );

    
    // #TODO: Remove this (test)
    static byte plotId[32] = {
        22, 24, 11, 3, 1, 15, 11, 6, 
        23, 22, 22, 24, 11, 3, 1, 15,
        11, 6, 23, 22, 22, 24, 11, 3,
        1, 15, 11, 6, 23, 22, 5, 28
    };

    static const uint plotMemoSize     = 128;
    static byte plotMemo[plotMemoSize] = { 0 };

    {
        const char refPlotId  [] = "c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835";
        const char refPlotMemo[] = "80a836a74b077cabaca7a76d1c3c9f269f7f3a8f2fa196a65ee8953eb81274eb8b7328d474982617af5a0fe71b47e9b8ade0cc43610ce7540ab96a524d0ab17f5df7866ef13d1221a7203e5d10ad2a4ae37f7b73f6cdfd6ddf4122e8a1c2f8ef01b7bf8a22a9ac82a003e07b551c851ea683839f3e1beb8ac9ede57d2c020669";

        memset( plotId  , 0, sizeof( plotId   ) );
        memset( plotMemo, 0, sizeof( plotMemo ) );

        HexStrToBytes( refPlotId  , sizeof( refPlotId   )-1, plotId  , sizeof( plotId   ) );
        HexStrToBytes( refPlotMemo, sizeof( refPlotMemo )-1, plotMemo, sizeof( plotMemo ) );
    }

    _cx.plotId       = plotId;
    _cx.plotMemoSize = plotMemoSize;
    _cx.plotMemo     = plotMemo;
}

//-----------------------------------------------------------
void DiskPlotter::Plot( const PlotRequest& req )
{
    // TestF1Buckets( *_cx.threadPool, _cx.plotId, _cx.plotMemo ); return;

    Log::Line( "Started plot." );
    auto plotTimer = TimerBegin();

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

    double plotElapsed = TimerEnd( plotTimer );
    Log::Line( "Finished plotting in %.2lf seconds ( %.2lf minutes ).", plotElapsed, plotElapsed / 60 );
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

    sizes.crossBucketTotal = 
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
