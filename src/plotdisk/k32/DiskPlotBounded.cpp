#include "DiskPlotBounded.h"
#include "util/Util.h"
#include "threading/MTJob.h"
#include "plotdisk/DiskPlotInfo.h"
#include "plotdisk/DiskPlotContext.h"
#include "plotdisk/DiskPlotConfig.h"
#include "plotdisk/DiskBufferQueue.h"
#include "plotting/PlotTools.h"

#include "F1Bounded.inl"
#include "FxBounded.inl"

//-----------------------------------------------------------
K32BoundedPhase1::K32BoundedPhase1( DiskPlotContext& context )
    : _context  ( context )
    , _ioQueue  ( *_context.ioQueue )
    , _allocator( context.heapBuffer, context.heapSize ) 
{
    const uint32 numBuckets = context.numBuckets;

    // Open files
    // Temp1
    {
        const FileSetOptions tmp1Options = context.cfg->noTmp1DirectIO ? FileSetOptions::None : FileSetOptions::DirectIO;

        _ioQueue.InitFileSet( FileId::T1, "t1", 1, tmp1Options, nullptr );  // X (sorted on Y)
        _ioQueue.InitFileSet( FileId::T2, "t2", 1, tmp1Options, nullptr );  // Back pointers
        _ioQueue.InitFileSet( FileId::T3, "t3", 1, tmp1Options, nullptr );
        _ioQueue.InitFileSet( FileId::T4, "t4", 1, tmp1Options, nullptr );
        _ioQueue.InitFileSet( FileId::T5, "t5", 1, tmp1Options, nullptr );
        _ioQueue.InitFileSet( FileId::T6, "t6", 1, tmp1Options, nullptr );
        _ioQueue.InitFileSet( FileId::T7, "t7", 1, tmp1Options, nullptr );

        _ioQueue.InitFileSet( FileId::MAP2, "map2", numBuckets, tmp1Options, nullptr );
        _ioQueue.InitFileSet( FileId::MAP3, "map3", numBuckets, tmp1Options, nullptr );
        _ioQueue.InitFileSet( FileId::MAP4, "map4", numBuckets, tmp1Options, nullptr );
        _ioQueue.InitFileSet( FileId::MAP5, "map5", numBuckets, tmp1Options, nullptr );
        _ioQueue.InitFileSet( FileId::MAP6, "map6", numBuckets, tmp1Options, nullptr );
        _ioQueue.InitFileSet( FileId::MAP7, "map7", numBuckets, tmp1Options, nullptr );
    }

    // Temp2
    {
        FileSetOptions opts = FileSetOptions::None | FileSetOptions::Interleaved;
        
        if( !context.cfg->noTmp2DirectIO )
            opts |= FileSetOptions::DirectIO;

        FileSetInitData data = {};
        opts |= FileSetOptions::UseTemp2;

        // if( context.cache )
        // {
        //     // #TODO: Divide cache evenly. For now just set it to fx0
        //     opts |= FileSetOptions::Cachable;
        //     data.cacheSize = context.cacheSize / 2;
        //     data.cache     = context.cache;
        // }

        _ioQueue.InitFileSet( FileId::FX0   , "y0"    , numBuckets, opts, &data );
        // UnSetFlag( opts, FileSetOptions::Cachable );
        _ioQueue.InitFileSet( FileId::FX1   , "y1"    , numBuckets, opts, &data );
        _ioQueue.InitFileSet( FileId::INDEX0, "index0", numBuckets, opts, &data );
        _ioQueue.InitFileSet( FileId::INDEX1, "index1", numBuckets, opts, &data );
        
        // opts |= FileSetOptions::Cachable;
        // data.cache = ((byte*)data.cache) + data.cacheSize;
        _ioQueue.InitFileSet( FileId::META0 , "meta0" , numBuckets, opts, &data );
        // UnSetFlag( opts, FileSetOptions::Cachable );
        
        _ioQueue.InitFileSet( FileId::META1 , "meta1" , numBuckets, opts, &data );
    }
}

//-----------------------------------------------------------
K32BoundedPhase1::~K32BoundedPhase1()
{}

//-----------------------------------------------------------
size_t K32BoundedPhase1::GetRequiredSize( const uint32 numBuckets, const size_t t1BlockSize, const size_t t2BlockSize, const uint32 threadCount )
{
    DummyAllocator allocator;

    #if BB_DP_FP_MATCH_X_BUCKET
        allocator.CAlloc<K32CrossBucketEntries>( numBuckets );
        allocator.CAlloc<K32CrossBucketEntries>( numBuckets );
    #endif

    switch( numBuckets )
    {
        case 64 : DiskPlotFxBounded<TableId::Table4,64 >::GetRequiredHeapSize( allocator, t1BlockSize, t2BlockSize, threadCount ); break;
        case 128: DiskPlotFxBounded<TableId::Table4,128>::GetRequiredHeapSize( allocator, t1BlockSize, t2BlockSize, threadCount ); break;
        case 256: DiskPlotFxBounded<TableId::Table4,256>::GetRequiredHeapSize( allocator, t1BlockSize, t2BlockSize, threadCount ); break;
        case 512: DiskPlotFxBounded<TableId::Table4,512>::GetRequiredHeapSize( allocator, t1BlockSize, t2BlockSize, threadCount ); break;

        default:
            Panic( "Invalid bucket count %u.", numBuckets );
            break;
    }

    return allocator.Size();
}

//-----------------------------------------------------------
void K32BoundedPhase1::Run()
{
    Log::Line( "Table 1: F1 generation" );

    switch( _context.numBuckets )
    {
        case 64 : RunWithBuckets<64 >(); break;
        case 128: RunWithBuckets<128>(); break;
        case 256: RunWithBuckets<256>(); break;
        case 512: RunWithBuckets<512>(); break;

        default:
            Fatal( "Invalid bucket count %u", _context.numBuckets );
    }
}

//-----------------------------------------------------------
template<uint32 _numBuckets>
void K32BoundedPhase1::RunWithBuckets()
{
    TableId startTable = TableId::Table2;

    #if defined( _DEBUG ) && defined( BB_DP_P1_SKIP_TO_TABLE ) 
    {
        ASSERT( _context.entryCounts[0] == 1ull << 32 );
        ASSERT( BB_DP_P1_START_TABLE > TableId::Table2 );
        startTable = BB_DP_P1_START_TABLE;

        {
            Fence fence;
            _ioQueue.DebugReadSliceSizes( startTable-1, (uint)(startTable-1) %2 == 0 ? FileId::FX0    : FileId::FX1    );
            _ioQueue.DebugReadSliceSizes( startTable-1, (uint)(startTable-1) %2 == 0 ? FileId::INDEX0 : FileId::INDEX1 );
            _ioQueue.DebugReadSliceSizes( startTable-1, (uint)(startTable-1) %2 == 0 ? FileId::META0  : FileId::META1  );
            _ioQueue.SignalFence( fence );
            _ioQueue.CommitCommands();
            fence.Wait();
        }
    }
    #else
        RunF1<_numBuckets>();
    #endif

    #if BB_DP_FP_MATCH_X_BUCKET
        _crossBucketEntries[0].values = _allocator.CAlloc<K32CrossBucketEntries>( _numBuckets );
        _crossBucketEntries[1].values = _allocator.CAlloc<K32CrossBucketEntries>( _numBuckets );
        _crossBucketEntries[0].length = _numBuckets;
        _crossBucketEntries[1].length = _numBuckets;

        _xBucketStackMarker = _allocator.Size();
    #endif

    for( TableId table = startTable; table <= TableId::Table7; table++ )
    {
        switch( table )
        {
            case TableId::Table2: RunFx<TableId::Table2, _numBuckets>(); break;
            case TableId::Table3: RunFx<TableId::Table3, _numBuckets>(); break;
            case TableId::Table4: RunFx<TableId::Table4, _numBuckets>(); break;
            case TableId::Table5: RunFx<TableId::Table5, _numBuckets>(); break;
            case TableId::Table6: RunFx<TableId::Table6, _numBuckets>(); break;
            case TableId::Table7: RunFx<TableId::Table7, _numBuckets>(); break;
        
            default:
                PanicExit();
                break;
        }
    }
}

//-----------------------------------------------------------
template<uint32 _numBuckets>
void K32BoundedPhase1::RunF1()
{
    Log::Line( "Generating f1..." );
    auto timer = TimerBegin();


    StackAllocator allocator( _context.heapBuffer, _context.heapSize );
    K32BoundedF1<_numBuckets> f1( _context, allocator );
    f1.Run();

    double elapsed = TimerEnd( timer );
    Log::Line( "Finished f1 generation in %.2lf seconds. ", elapsed );
    Log::Line( "Table 1 I/O wait time: %.2lf seconds.", _context.ioQueue->IOBufferWaitTime() );

    #if BB_IO_METRICS_ON
        const double writeThroughput = _context.ioQueue->GetAverageWriteThroughput();
        const auto&  writes          = _context.ioQueue->GetWriteMetrics();

        Log::Line( " Table 1 I/O Metrics:" );
        Log::Line( "  Average write throughput %.2lf MiB ( %.2lf MB ) or %.2lf GiB ( %.2lf GB ).", 
            writeThroughput BtoMB, writeThroughput / 1000000.0, writeThroughput BtoGB, writeThroughput / 1000000000.0 );
        Log::Line( "  Total size written: %.2lf MiB ( %.2lf MB ) or %.2lf GiB ( %.2lf GB ).",
            (double)writes.size BtoMB, (double)writes.size / 1000000.0, (double)writes.size BtoGB, (double)writes.size / 1000000000.0 );
        Log::Line( "  Total write commands: %llu.", (llu)writes.count );
        Log::Line( "" );

        _context.ioQueue->ClearWriteMetrics();
    #endif
}

//-----------------------------------------------------------
template<TableId table, uint32 _numBuckets>
void K32BoundedPhase1::RunFx()
{
    Log::Line( "Table %u", table+1 );
    auto timer = TimerBegin();

    #if BB_DP_FP_MATCH_X_BUCKET
        _allocator.PopToMarker( _xBucketStackMarker );
        
        const uint xBucketIdxIn  = (uint)(table-1) & 1;
        const uint xBucketIdxOut = (uint)table & 1;
        auto& crossBucketIn  = _crossBucketEntries[xBucketIdxIn];
        auto& crossBucketOut = _crossBucketEntries[xBucketIdxOut];

        for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
            crossBucketOut[bucket].length = 0;

        if( table == TableId::Table2 )
        {
            for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
                crossBucketIn[bucket].length = 0;
        }
    #else
        _allocator.PopToMarker( 0 );
    #endif

    DiskPlotFxBounded<table, _numBuckets> fx( _context );
    fx.Run( _allocator
        #if BB_DP_FP_MATCH_X_BUCKET
            , crossBucketIn
            , crossBucketOut
        #endif
    );

    Log::Line( "Completed table %u in %.2lf seconds with %.llu entries.", table+1, TimerEnd( timer ), _context.entryCounts[(int)table] );
    Log::Line( "Table %u I/O wait time: %.2lf seconds.",  table+1, TicksToSeconds( fx._tableIOWait ) );

    #if _DEBUG
        BB_DP_DBG_WriteTableCounts( _context );

        _ioQueue.DebugWriteSliceSizes( table, (uint)table %2 == 0 ? FileId::FX0    : FileId::FX1    );
        _ioQueue.DebugWriteSliceSizes( table, (uint)table %2 == 0 ? FileId::INDEX0 : FileId::INDEX1 );
        _ioQueue.DebugWriteSliceSizes( table, (uint)table %2 == 0 ? FileId::META0  : FileId::META1  );
        _ioQueue.CommitCommands();
    #endif
}

