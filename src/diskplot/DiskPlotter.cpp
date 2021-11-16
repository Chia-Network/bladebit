#include "DiskPlotter.h"
#include "util/Log.h"
#include "Util.h"

#include "DiskPlotPhase1.h"
#include "SysHost.h"

//-----------------------------------------------------------
DiskPlotter::DiskPlotter()
{

}

//-----------------------------------------------------------
DiskPlotter::DiskPlotter( const Config cfg )
{
    ZeroMem( &_cx );

    ASSERT( cfg.tmpPath );
    
    const size_t bucketsCountsSize = RoundUpToNextBoundaryT( BB_DP_BUCKET_COUNT * sizeof( uint32 ), cfg.expectedTmpDirBlockSize );
    const uint32 ioBufferCount     = 3;    // Test with triple-buffering for now
    const size_t ioHeapFullSize    = ( cfg.ioBufferSize + bucketsCountsSize ) * ioBufferCount;

    _cx.tmpPath       = cfg.tmpPath;
    
    _cx.heapSize      = GetHeapRequiredSize( cfg.expectedTmpDirBlockSize, cfg.workThreadCount );
    _cx.ioBufferSize  = cfg.ioBufferSize;
    _cx.ioHeapSize    = ioHeapFullSize;
    _cx.ioBufferCount = ioBufferCount;

    _cx.threadCount   = cfg.workThreadCount;
    _cx.ioThreadCount = cfg.ioThreadCount;

    static_assert( sizeof( DiskPlotContext::writeIntervals ) == sizeof( Config::writeIntervals ), "Write interval array sizes do not match." );
    memcpy( _cx.writeIntervals, cfg.writeIntervals, sizeof( _cx.writeIntervals ) );

    Log::Line( "Work Heap size : %.2lf MiB", (double)_cx.heapSize BtoMB );
    Log::Line( "Work threads   : %u"       , _cx.threadCount   );
    Log::Line( "IO buffer size : %llu MiB (%llu MiB total)", _cx.ioBufferSize BtoMB, _cx.ioBufferSize * _cx.ioBufferCount BtoMB );
    Log::Line( "IO threads     : %u"       , _cx.ioThreadCount );
    Log::Line( "IO buffer count: %u"       , _cx.ioBufferCount );

    const size_t allocationSize = _cx.heapSize + _cx.ioHeapSize;

    // Log::Line( "Allocating heap of %llu MiB.", _cx.heapSize BtoMB );
    Log::Line( "Allocating a heap of %llu MiB.", allocationSize BtoMB );
    _cx.heapBuffer = (byte*)SysHost::VirtualAlloc( allocationSize );
    FatalIf( !_cx.heapBuffer, "Failed to allocated heap buffer. Make sure you have enough free memory." );
    
    _cx.ioHeap = _cx.heapBuffer + _cx.heapSize;
    // Log::Line( "Allocating IO buffers." );
    // _cx.ioHeap = (byte*)SysHost::VirtualAlloc( ioHeapFullSize );
    // FatalIf( !_cx.ioHeap, "Failed to allocated work buffer. Make sure you have enough free memory." );

    _cx.threadPool = new ThreadPool( _cx.threadCount, ThreadPool::Mode::Fixed, false );
    
    static const byte plotId[32] = {
        22, 24, 11, 3, 1, 15, 11, 6, 
        23, 22, 22, 24, 11, 3, 1, 15,
        11, 6, 23, 22, 22, 24, 11, 3,
        1, 15, 11, 6, 23, 22, 5, 28
    };

    static const uint plotMemoSize     = 128;
    static byte plotMemo[plotMemoSize] = { 0 };

    _cx.plotId       = plotId;
    _cx.plotMemoSize = plotMemoSize;
    _cx.plotMemo     = plotMemo;
}

//-----------------------------------------------------------
void DiskPlotter::Plot( const PlotRequest& req )
{
    Log::Line( "Started plot." );
    auto plotTimer = TimerBegin();

    {
        DiskPlotPhase1 phase1( _cx );
        phase1.Run();
    }

    double plotElapsed = TimerEnd( plotTimer );
    Log::Line( "Finished plotting in %.2lf seconds ( %.2lf minutes ).", plotElapsed, plotElapsed / 60 );
}

//-----------------------------------------------------------
size_t DiskPlotter::GetHeapRequiredSize( const size_t fileBlockSize, const uint threadCount )
{
    const uint maxBucketEntries = BB_DP_MAX_ENTRIES_PER_BUCKET;

    // We need to add extra space to retain 2 groups worth of y value as we need to retain the 
    // last 2 groups between bucket processing. This is because we may have to match the previous'
    // groups buckets against the new bucket.
    const size_t yGroupExtra = RoundUpToNextBoundaryT( (size_t)kBC * sizeof( uint32 ) * 2, fileBlockSize );

    const size_t ySize       = RoundUpToNextBoundaryT( maxBucketEntries * sizeof( uint32 ) * 2, fileBlockSize ) + yGroupExtra * 2;  // x  2 because we need the extra space in both buffers
    const size_t sortKeySize = RoundUpToNextBoundaryT( maxBucketEntries * sizeof( uint32 )    , fileBlockSize );
    const size_t metaSize    = RoundUpToNextBoundaryT( maxBucketEntries * sizeof( uint64 ) * 4, fileBlockSize );

    // Add tmp y and meta buffers for now too
    // 
    // #TODO: These need to be excluded and actually allocated whenever we are actually going to
    //        do matches so that we over commit but only use whatever pages are actually used when matching.
    //        Otherwise our requirements will increase substantially.
    const size_t pairsLSize  = RoundUpToNextBoundaryT( maxBucketEntries * sizeof( uint32 )    , fileBlockSize );
    const size_t pairsRSize  = RoundUpToNextBoundaryT( maxBucketEntries * sizeof( uint16 )    , fileBlockSize );
    const size_t groupsSize  = RoundUpToNextBoundaryT( ( maxBucketEntries + threadCount * 2 ) * sizeof( uint32), fileBlockSize );

    const size_t totalSize   = ySize + sortKeySize + metaSize + pairsLSize + pairsRSize + groupsSize;


    return totalSize;
}
