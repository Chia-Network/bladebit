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
    , _ioQueue( _ioQueue )
{
    const uint32 numBuckets = context.numBuckets;

    // Open files
    FileSetOptions  opts = FileSetOptions::None;
    FileSetInitData data = {};

    opts = FileSetOptions::UseTemp2;
    _ioQueue.InitFileSet( FileId::FX0  , "y0"   , numBuckets, opts, &data );
    _ioQueue.InitFileSet( FileId::FX1  , "y1"   , numBuckets, opts, &data );
    _ioQueue.InitFileSet( FileId::META0, "meta0", numBuckets, opts, &data );
    _ioQueue.InitFileSet( FileId::META1, "meta1", numBuckets, opts, &data );
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
        RunFx<_numBuckets>( table );
    }
}

//-----------------------------------------------------------
template<uint32 _numBuckets>
void K32BoundedPhase1::RunF1()
{
    StackAllocator allocator( _context.heapBuffer, _context.heapSize );
    K32BoundedF1<_numBuckets> f1( _context, allocator );
    f1.Run();
}

//-----------------------------------------------------------
template<uint32 _numBuckets>
void K32BoundedPhase1::RunFx( const TableId table )
{
}


