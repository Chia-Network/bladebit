#include "DiskPlotPhase1.h"
#include "util/Util.h"
#include "util/Log.h"
#include "b3/blake3.h"
#include "algorithm/RadixSort.h"
#include "plotting/GenSortKey.h"
#include "jobs/LookupMapJob.h"
#include "plotting/TableWriter.h"
#include "util/StackAllocator.h"
#include "DiskF1.h"
#include "DiskFp.h"


// Test
#if _DEBUG
    #include "../plotmem/DbgHelper.h"
    #include "plotdisk/DiskPlotDebug.h"
#endif

//-----------------------------------------------------------
DiskPlotPhase1::DiskPlotPhase1( DiskPlotContext& cx )
    : _cx( cx )
    , _diskQueue( cx.ioQueue )
{
    ASSERT( cx.tmpPath );

    _diskQueue->InitFileSet( FileId::T1, "t1", 1, FileSetOptions::DirectIO, nullptr );  // X (sorted on Y)
    _diskQueue->InitFileSet( FileId::T2, "t2", 1, FileSetOptions::DirectIO, nullptr );  // Back pointers
    _diskQueue->InitFileSet( FileId::T3, "t3", 1, FileSetOptions::DirectIO, nullptr );
    _diskQueue->InitFileSet( FileId::T4, "t4", 1, FileSetOptions::DirectIO, nullptr );
    _diskQueue->InitFileSet( FileId::T5, "t5", 1, FileSetOptions::DirectIO, nullptr );
    _diskQueue->InitFileSet( FileId::T6, "t6", 1, FileSetOptions::DirectIO, nullptr );
    _diskQueue->InitFileSet( FileId::T7, "t7", 1, FileSetOptions::DirectIO, nullptr );

    _diskQueue->InitFileSet( FileId::MAP2, "map2", _cx.numBuckets+1, FileSetOptions::DirectIO, nullptr );
    _diskQueue->InitFileSet( FileId::MAP3, "map3", _cx.numBuckets+1, FileSetOptions::DirectIO, nullptr );
    _diskQueue->InitFileSet( FileId::MAP4, "map4", _cx.numBuckets+1, FileSetOptions::DirectIO, nullptr );
    _diskQueue->InitFileSet( FileId::MAP5, "map5", _cx.numBuckets+1, FileSetOptions::DirectIO, nullptr );
    _diskQueue->InitFileSet( FileId::MAP6, "map6", _cx.numBuckets+1, FileSetOptions::DirectIO, nullptr );
    _diskQueue->InitFileSet( FileId::MAP7, "map7", _cx.numBuckets+1, FileSetOptions::DirectIO, nullptr );
}

//-----------------------------------------------------------
void DiskPlotPhase1::Run()
{
    DiskPlotContext& cx = _cx;

    #if _DEBUG && BB_DP_DBG_SKIP_PHASE_1
    {
        FileStream bucketCounts, tableCounts, backPtrBucketCounts;

        if( bucketCounts.Open( BB_DP_DBG_TEST_DIR BB_DP_DBG_READ_BUCKET_COUNT_FNAME, FileMode::Open, FileAccess::Read ) )
        {
            if( bucketCounts.Read( cx.bucketCounts, sizeof( cx.bucketCounts ) ) != sizeof( cx.bucketCounts ) )
            {
                Log::Error( "Failed to read from bucket counts file." );
                goto CONTINUE;
            }
        }
        else
        {
            Log::Error( "Failed to open bucket counts file." );
            goto CONTINUE;
        }

        if( tableCounts.Open( BB_DP_DBG_TEST_DIR BB_DP_TABLE_COUNTS_FNAME, FileMode::Open, FileAccess::Read ) )
        {
            if( tableCounts.Read( cx.entryCounts, sizeof( cx.entryCounts ) ) != sizeof( cx.entryCounts ) )
            {
                Log::Error( "Failed to read from table counts file." );
                goto CONTINUE;
            }
        }
        else
        {
            Log::Error( "Failed to open table counts file." );
            goto CONTINUE;
        }

        if( backPtrBucketCounts.Open( BB_DP_DBG_TEST_DIR BB_DP_DBG_PTR_BUCKET_COUNT_FNAME, FileMode::Open, FileAccess::Read ) )
        {
            if( backPtrBucketCounts.Read( cx.ptrTableBucketCounts, sizeof( cx.ptrTableBucketCounts ) ) != sizeof( cx.ptrTableBucketCounts ) )
            {
                Fatal( "Failed to read from pointer bucket counts file." );
            }
        }
        else
        {
            Fatal( "Failed to open pointer bucket counts file." );
        }

        #if BB_DP_DBG_SKIP_TO_C_TABLES
            SortAndCompressTable7();
        #endif

        return;

    CONTINUE:;
    }
    #endif

    {
        const size_t cacheSize = _cx.cacheSize / 2;

        FileSetOptions opts = FileSetOptions::DirectIO;

        if( _cx.cache )
            opts |= FileSetOptions::Cachable;

        FileSetInitData fdata = {
            .cache     = _cx.cache,
            .cacheSize = cacheSize
        };

        _diskQueue->InitFileSet( FileId::FX0, "fx_0", _cx.numBuckets, opts, &fdata );
        
        fdata.cache = ((byte*)fdata.cache) + cacheSize;
        _diskQueue->InitFileSet( FileId::FX1, "fx_1", _cx.numBuckets, opts, &fdata );
    }

#if !BB_DP_DBG_READ_EXISTING_F1
    GenF1();
#else
    {
        size_t pathLen = strlen( cx.tmpPath );
        pathLen += sizeof( BB_DP_DBG_READ_BUCKET_COUNT_FNAME );

        std::string bucketsPath = cx.tmpPath;
        if( bucketsPath[bucketsPath.length() - 1] != '/' && bucketsPath[bucketsPath.length() - 1] != '\\' )
            bucketsPath += "/";

        bucketsPath += BB_DP_DBG_READ_BUCKET_COUNT_FNAME;

        const size_t bucketsCountSize = sizeof( uint32 ) * BB_DP_BUCKET_COUNT;

        FileStream fBucketCounts;
        if( fBucketCounts.Open( bucketsPath.c_str(), FileMode::Open, FileAccess::Read ) )
        {

            size_t sizeRead = fBucketCounts.Read( cx.bucketCounts[0], bucketsCountSize );
            FatalIf( sizeRead != bucketsCountSize, "Invalid bucket counts." );
        }
        else
        {
            GenF1();

            fBucketCounts.Close();
            FatalIf( !fBucketCounts.Open( bucketsPath.c_str(), FileMode::Create, FileAccess::Write ), "File to open bucket counts file" );
            FatalIf( fBucketCounts.Write( cx.bucketCounts[0], bucketsCountSize ) != bucketsCountSize, "Failed to write bucket counts.");
        }
    }
#endif

    #if _DEBUG && BB_DP_DBG_VALIDATE_F1
    // if( 0 )
    {
        const uint32* bucketCounts = cx.bucketCounts[0];
        uint64 totalEntries = 0;
        for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
            totalEntries += bucketCounts[i];
            
        ASSERT( totalEntries == 1ull << _K );

        Debug::ValidateYFileFromBuckets( FileId::Y0, *cx.threadPool, *_diskQueue, TableId::Table1, cx.bucketCounts[0] );
    }
    #endif

    ForwardPropagate();

    // Check all table counts
    #if _DEBUG
    for( int table = (int)TableId::Table1; table <= (int)TableId::Table7; table++ )
    {
        uint64 entryCount = 0;

        for( int bucket = 0; bucket < (int)_cx.numBuckets; bucket++ )
            entryCount += cx.bucketCounts[table][bucket];

        ASSERT( entryCount == cx.entryCounts[table] );
    }
    #endif

    #if _DEBUG
    {
        // Write bucket counts
        FileStream bucketCounts, tableCounts, backPtrBucketCounts;

        if( bucketCounts.Open( BB_DP_DBG_TEST_DIR BB_DP_DBG_READ_BUCKET_COUNT_FNAME, FileMode::Create, FileAccess::Write ) )
        {
            if( bucketCounts.Write( cx.bucketCounts, sizeof( cx.bucketCounts ) ) != sizeof( cx.bucketCounts ) )
                Log::Error( "Failed to write to bucket counts file." );
        }
        else
            Log::Error( "Failed to open bucket counts file." );

        if( tableCounts.Open( BB_DP_DBG_TEST_DIR BB_DP_TABLE_COUNTS_FNAME, FileMode::Create, FileAccess::Write ) )
        {
            if( tableCounts.Write( cx.entryCounts, sizeof( cx.entryCounts ) ) != sizeof( cx.entryCounts ) )
                Log::Error( "Failed to write to table counts file." );
        }
        else
            Log::Error( "Failed to open table counts file." );

        if( backPtrBucketCounts.Open( BB_DP_DBG_TEST_DIR BB_DP_DBG_PTR_BUCKET_COUNT_FNAME, FileMode::Create, FileAccess::Write ) )
        {
            if( backPtrBucketCounts.Write( cx.ptrTableBucketCounts, sizeof( cx.ptrTableBucketCounts ) ) != sizeof( cx.ptrTableBucketCounts ) )
                Log::Error( "Failed to write to back pointer bucket counts file." );
        }
        else
            Log::Error( "Failed to open back pointer bucket counts file." );
    }
    #endif

    // SortAndCompressTable7();

    Log::Line( " Phase 1 Total IO Aggregate Wait Time | READ: %.4lf | WRITE: %.4lf | BUFFERS: %.4lf", 
            TicksToSeconds( _cx.readWaitTime ), TicksToSeconds( _cx.writeWaitTime ), _cx.ioQueue->IOBufferWaitTime() );
}

///
/// F1 Generation
///
//-----------------------------------------------------------
void DiskPlotPhase1::GenF1()
{
    Log::Line( "Generating f1..." );
    auto timer = TimerBegin();
    
    switch( _cx.numBuckets )
    {
        case 128 : GenF1Buckets<128 >(); break;
        case 256 : GenF1Buckets<256 >(); break;
        case 512 : GenF1Buckets<512 >(); break;
        case 1024: GenF1Buckets<1024>(); break;
    default:
        ASSERT( 0 );
        break;
    }

    _cx.entryCounts[0] = 1ull << _K;
    
    double elapsed = TimerEnd( timer );
    Log::Line( "Finished f1 generation in %.2lf seconds. ", elapsed );
    Log::Line( "Table 1 IO wait time: Write: %.2lf.", _cx.ioQueue->IOBufferWaitTime() );
}

//-----------------------------------------------------------
template <uint32 _numBuckets>
void DiskPlotPhase1::GenF1Buckets()
{
    DiskF1<_numBuckets> f1( _cx, FileId::FX0 );
    f1.GenF1();
}

//-----------------------------------------------------------
void DiskPlotPhase1::ForwardPropagate()
{
    for( TableId table = TableId::Table2; table <= TableId::Table7; table++ )
    {
        Log::Line( "Table %u", table+1 );
        auto timer = TimerBegin();

        switch( table )
        {
            case TableId::Table2: ForwardPropagateTable<TableId::Table2>(); break;
            case TableId::Table3: ForwardPropagateTable<TableId::Table3>(); break;
            case TableId::Table4: ForwardPropagateTable<TableId::Table4>(); break;
            case TableId::Table5: ForwardPropagateTable<TableId::Table5>(); break;
            case TableId::Table6: ForwardPropagateTable<TableId::Table6>(); break;
            case TableId::Table7: ForwardPropagateTable<TableId::Table7>(); break;
        
            default:
                Fatal( "Invalid table." );
                break;
        }

        Log::Line( "Completed table %u in %.2lf seconds.", table+1, TimerEnd( timer ) );
        Log::Line( "Table IO wait time: Read: %.2lf s | Write: %.2lf.", 
                    TicksToSeconds( _tableReadWaitTime ), TicksToSeconds( _tableWriteWaitTime ) );

        std::swap( _fxIn, _fxOut );
    }
}

//-----------------------------------------------------------
template<TableId table>
void DiskPlotPhase1::ForwardPropagateTable()
{
    _tableReadWaitTime  = Duration::zero();
    _tableWriteWaitTime = Duration::zero();

    const uint32 numBuckets = _cx.numBuckets;
    
    switch ( numBuckets )
    {
        case 128 : ForwardPropagateBuckets<table, 128 >(); break;
        case 256 : ForwardPropagateBuckets<table, 256 >(); break;
        case 512 : ForwardPropagateBuckets<table, 512 >(); break;
        case 1024: ForwardPropagateBuckets<table, 1024>(); break;
    
        default:
            Fatal( "Invalid bucket count." );
            break;
    }
}

//-----------------------------------------------------------
template<TableId table, uint32 _numBuckets>
void DiskPlotPhase1::ForwardPropagateBuckets()
{
    DiskFp<table, _numBuckets> fp( _cx, _fxIn, _fxOut );
    fp.Run();

    _tableReadWaitTime  = fp.ReadWaitTime();
    _tableWriteWaitTime = fp.WriteWaitTime();
    _cx.readWaitTime  += _tableReadWaitTime;
    _cx.writeWaitTime += _tableWriteWaitTime;

    #if BB_DP_DBG_VALIDATE_FX
        #if !_DEBUG
            Log::Line( "Warning: Table validation enabled in release mode." );
        #endif
        
        using TYOut = typename DiskFp<table, _numBuckets>::TYOut;
        Debug::ValidateYForTable<table, _numBuckets, TYOut>( _fxOut, *_cx.ioQueue, *_cx.threadPool, _cx.bucketCounts[(int)table] );
    #endif
}

