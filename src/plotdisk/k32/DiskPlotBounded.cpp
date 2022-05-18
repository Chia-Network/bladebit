#include "DiskPlotBounded.h"
#include "util/Util.h"
#include "util/StackAllocator.h"
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
    : _context( context )
    , _ioQueue( *_context.ioQueue )
{
    const uint32 numBuckets = context.numBuckets;

    // Open files
    FileSetOptions opts = FileSetOptions::None | FileSetOptions::Interleaved;
    
    if( !context.cfg->noTmp2DirectIO )
        opts |= FileSetOptions::DirectIO;

    FileSetInitData data = {};
    opts |= FileSetOptions::UseTemp2;

    _ioQueue.InitFileSet( FileId::FX0   , "y0"    , numBuckets, opts, &data );
    _ioQueue.InitFileSet( FileId::FX1   , "y1"    , numBuckets, opts, &data );
    _ioQueue.InitFileSet( FileId::INDEX0, "index0", numBuckets, opts, &data );
    _ioQueue.InitFileSet( FileId::INDEX1, "index1", numBuckets, opts, &data );
    _ioQueue.InitFileSet( FileId::META0 , "meta0" , numBuckets, opts, &data );
    _ioQueue.InitFileSet( FileId::META1 , "meta1" , numBuckets, opts, &data );
}

//-----------------------------------------------------------
K32BoundedPhase1::~K32BoundedPhase1()
{}

//-----------------------------------------------------------
size_t K32BoundedPhase1::GetRequiredSize( const uint32 numBuckets, const size_t t1BlockSize, const size_t t2BlockSize )
{
    switch( numBuckets )
    {
        case 64 : return DiskPlotFxBounded<TableId::Table4,64 >::GetRequiredHeapSize( t1BlockSize, t2BlockSize );
        case 128: return DiskPlotFxBounded<TableId::Table4,128>::GetRequiredHeapSize( t1BlockSize, t2BlockSize );
        case 256: return DiskPlotFxBounded<TableId::Table4,256>::GetRequiredHeapSize( t1BlockSize, t2BlockSize );
        case 512: return DiskPlotFxBounded<TableId::Table4,512>::GetRequiredHeapSize( t1BlockSize, t2BlockSize );
        default:
            break;
    }

    Panic( "Invalid bucket count %u.", numBuckets );
    return 0;
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
    RunF1<_numBuckets>();

    for( TableId table = TableId::Table2; table <= TableId::Table7; table++ )
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

    _context.entryCounts[(int)TableId::Table1] = 1ull << 32;

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
    DiskPlotFxBounded<table, _numBuckets> fx( _context );
    fx.Run();
}

