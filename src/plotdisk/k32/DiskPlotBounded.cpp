#include "DiskPlotBounded.h"
#include "util/Util.h"
#include "threading/MTJob.h"
#include "plotdisk/DiskPlotInfo.h"
#include "plotdisk/DiskPlotContext.h"
#include "plotdisk/DiskPlotConfig.h"
#include "plotdisk/DiskBufferQueue.h"
#include "CTableWriterBounded.h"
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
        FileSetInitData data = {};
        FileSetOptions  opts = context.cfg->alternateBuckets ? FileSetOptions::Alternating : FileSetOptions::Interleaved;
        
        if( !context.cfg->noTmp2DirectIO )
            opts |= FileSetOptions::DirectIO;

        opts |= FileSetOptions::UseTemp2;

        size_t metaCacheSize = 0;

        if( context.cache )
        {
            // In fully interleaved mode (bigger writes chunks), we need 192GiB for k=32
            // In alternating mode, we need 96 GiB (around 99GiB if we account for disk block-alignment requirements).
            
            opts |= FileSetOptions::Cachable;
            data.cache = context.cache;

            // Proportion out the size required per file:
            //  A single y or index file requires at maximum 16GiB
            //  A asingle meta file at its maximum will require 64GiB
            // Divide the whole cache into 12 (6 for alternating mode) equal parts, where each meta file represent 4 parts.
            // Ex: 192 / 12 = 16.  This gives us 4 files of 16GiB and 2 files of 64GiB
            size_t singleFileCacheSize = context.cacheSize / ( context.cfg->alternateBuckets ? 6 : 12 );

            // Align to block size
            singleFileCacheSize = numBuckets * RoundUpToNextBoundaryT( singleFileCacheSize / numBuckets - context.tmp2BlockSize, context.tmp2BlockSize );

            data.cacheSize = singleFileCacheSize;
            metaCacheSize  = data.cacheSize * 4;     // Meta needs 4 times as much as y and index

            ASSERT( data.cacheSize && metaCacheSize );
        }

        auto InitCachableFileSet = [=]( FileId fileId, const char* fileName, uint32 numBuckets, FileSetOptions opts, FileSetInitData& data ) {
                
            _ioQueue.InitFileSet( fileId, fileName, numBuckets, opts, &data );
            data.cache = (byte*)data.cache + data.cacheSize;
        };

        if( _context.cfg->alternateBuckets )
        {
            const uint64 blockSize       = context.tmp2BlockSize;
            const uint64 tableEntries    = 1ull << 32;
            const uint64 bucketEntries   = tableEntries / numBuckets;
            const uint64 sliceEntries    = bucketEntries / numBuckets;

            const uint64 ysPerBlock      = blockSize / sizeof( uint32 );
            const uint64 metasPerBlock   = blockSize / (sizeof( uint32 ) * 4);

            const uint64 sliceSizeY    = RoundUpToNextBoundaryT( (uint64)(sliceEntries * BB_DP_ENTRY_SLICE_MULTIPLIER), ysPerBlock    ) * sizeof( uint32 );
            const uint64 sliceSizeMeta = RoundUpToNextBoundaryT( (uint64)(sliceEntries * BB_DP_ENTRY_SLICE_MULTIPLIER), metasPerBlock ) * sizeof( uint32 ) * 4;
            
            data.maxSliceSize = sliceSizeY;
            InitCachableFileSet( FileId::FX0   , "y0"    , numBuckets, opts, data );
            InitCachableFileSet( FileId::INDEX0, "index0", numBuckets, opts, data );

            data.maxSliceSize = sliceSizeMeta;
            data.cacheSize        = metaCacheSize;
            InitCachableFileSet( FileId::META0, "meta0", numBuckets, opts, data );
        }
        else
        {
            InitCachableFileSet( FileId::FX0   , "y0"    , numBuckets, opts, data );
            InitCachableFileSet( FileId::FX1   , "y1"    , numBuckets, opts, data );
            InitCachableFileSet( FileId::INDEX0, "index0", numBuckets, opts, data );
            InitCachableFileSet( FileId::INDEX1, "index1", numBuckets, opts, data );

            data.cacheSize = metaCacheSize;
            InitCachableFileSet( FileId::META0, "meta0", numBuckets, opts, data );
            InitCachableFileSet( FileId::META1, "meta1", numBuckets, opts, data );
        }
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

    #if defined( _DEBUG ) && ( defined( BB_DP_P1_SKIP_TO_TABLE ) || defined( BB_DP_DBG_SKIP_TO_C_TABLES ) )
    {
        ASSERT( _context.entryCounts[0] == 1ull << 32 );

        #if BB_DP_P1_SKIP_TO_TABLE
            ASSERT( BB_DP_P1_START_TABLE > TableId::Table2 );
            startTable = BB_DP_P1_START_TABLE;
        #endif

        {
            #if BB_DP_DBG_SKIP_TO_C_TABLES
                const FileId fxId   = FileId::FX0;
                const FileId idxId  = FileId::INDEX0;
                const FileId metaId = FileId::META0;
                startTable = TableId::Table7;
            #else
                const FileId fxId   = (uint)(startTable-1) %2 == 0 ? FileId::FX0    : FileId::FX1;
                const FileId idxId  = (uint)(startTable-1) %2 == 0 ? FileId::INDEX0 : FileId::INDEX1;
                const FileId metaId = (uint)(startTable-1) %2 == 0 ? FileId::META0  : FileId::META1;
            #endif

            Fence fence;
            _ioQueue.DebugReadSliceSizes( startTable, fxId   );
            _ioQueue.DebugReadSliceSizes( startTable, idxId  );
            _ioQueue.DebugReadSliceSizes( startTable, metaId );
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

#if !( defined( _DEBUG ) && defined( BB_DP_DBG_SKIP_TO_C_TABLES ) )
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

        #if !BB_DP_P1_KEEP_FILES
            if( table == TableId::Table6 )
            {
                if( !_context.cfg->alternateBuckets )
                {
                    _ioQueue.DeleteBucket( FileId::META0 );
                    _ioQueue.CommitCommands();
                }
            }
            else if( table == TableId::Table7 )
            {
                if( _context.cfg->alternateBuckets )
                    _ioQueue.DeleteBucket( FileId::META0 );
                else
                {
                    _ioQueue.DeleteBucket( FileId::FX1    );
                    _ioQueue.DeleteBucket( FileId::INDEX1 );
                    _ioQueue.DeleteBucket( FileId::META1  );
                }
                _ioQueue.CommitCommands();
            }
        #endif
    }
#endif

    // Process F7 and write C tables to plot
    {
        _allocator.PopToMarker( 0 );

        Log::Line( "Sorting F7 & Writing C Tables" );
        auto timer = TimerBegin();
        CTableWriterBounded<_numBuckets> cWriter( _context );

        cWriter.Run( _allocator );

        Log::Line( "Completed F7 tables in %.2lf seconds.", TimerEnd( timer ) );
        Log::Line( "F7/C Tables I/O wait time: %.2lf seconds.",  TicksToSeconds( cWriter.IOWait() ) );
        _context.ioWaitTime += cWriter.IOWait();
    }

     #if !BB_DP_P1_KEEP_FILES
        _ioQueue.DeleteBucket( FileId::FX0    );
        _ioQueue.DeleteBucket( FileId::INDEX0 );
        _ioQueue.CommitCommands();
        // # TODO: Wait for deletion?
    #endif
}

//-----------------------------------------------------------
template<uint32 _numBuckets>
void K32BoundedPhase1::RunF1()
{
    _context.ioWaitTime = Duration::zero();

    Log::Line( "Table 1: F1 generation" );
    Log::Line( "Generating f1..." );

    const auto timer = TimerBegin();
    StackAllocator allocator( _context.heapBuffer, _context.heapSize );
    K32BoundedF1<_numBuckets> f1( _context, allocator );
    f1.Run();
    const double elapsed = TimerEnd( timer );

    Log::Line( "Finished f1 generation in %.2lf seconds. ", elapsed );
    Log::Line( "Table 1 I/O wait time: %.2lf seconds.", _context.p1TableWaitTime[(int)TableId::Table1] );
    
    _context.ioWaitTime += _context.p1TableWaitTime[(int)TableId::Table1];
    _context.ioQueue->DumpWriteMetrics( TableId::Table1 );
}

//-----------------------------------------------------------
template<TableId table, uint32 _numBuckets>
void K32BoundedPhase1::RunFx()
{
    Log::Line( "Table %u", table+1 );
    auto timer = TimerBegin();

    #if BB_DP_FP_MATCH_X_BUCKET
        _allocator.PopToMarker( _xBucketStackMarker );
        
        const uint  xBucketIdxIn   = (uint)(table-1) & 1;
        const uint  xBucketIdxOut  = (uint)table & 1;
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
    
    _context.ioQueue->DumpDiskMetrics( table );
    _context.p1TableWaitTime[(int)table] = fx._tableIOWait;
    _context.ioWaitTime += fx._tableIOWait;

    #if _DEBUG
        BB_DP_DBG_WriteTableCounts( _context );
         // #TODO: Update this for alternating mode
        _ioQueue.DebugWriteSliceSizes( table, (uint)table %2 == 0 ? FileId::FX0    : FileId::FX1    );
        _ioQueue.DebugWriteSliceSizes( table, (uint)table %2 == 0 ? FileId::INDEX0 : FileId::INDEX1 );
        _ioQueue.DebugWriteSliceSizes( table, (uint)table %2 == 0 ? FileId::META0  : FileId::META1  );
        _ioQueue.CommitCommands();
    #endif
}

