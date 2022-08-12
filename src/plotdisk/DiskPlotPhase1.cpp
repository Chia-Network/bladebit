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

    const FileSetOptions tmp1Options = cx.cfg->noTmp1DirectIO ? FileSetOptions::None : FileSetOptions::DirectIO;

    _diskQueue->InitFileSet( FileId::T1, "t1", 1, tmp1Options, nullptr );  // X (sorted on Y)
    _diskQueue->InitFileSet( FileId::T2, "t2", 1, tmp1Options, nullptr );  // Back pointers
    _diskQueue->InitFileSet( FileId::T3, "t3", 1, tmp1Options, nullptr );
    _diskQueue->InitFileSet( FileId::T4, "t4", 1, tmp1Options, nullptr );
    _diskQueue->InitFileSet( FileId::T5, "t5", 1, tmp1Options, nullptr );
    _diskQueue->InitFileSet( FileId::T6, "t6", 1, tmp1Options, nullptr );
    _diskQueue->InitFileSet( FileId::T7, "t7", 1, tmp1Options, nullptr );

    _diskQueue->InitFileSet( FileId::MAP2, "map2", _cx.numBuckets+1, tmp1Options, nullptr );
    _diskQueue->InitFileSet( FileId::MAP3, "map3", _cx.numBuckets+1, tmp1Options, nullptr );
    _diskQueue->InitFileSet( FileId::MAP4, "map4", _cx.numBuckets+1, tmp1Options, nullptr );
    _diskQueue->InitFileSet( FileId::MAP5, "map5", _cx.numBuckets+1, tmp1Options, nullptr );
    _diskQueue->InitFileSet( FileId::MAP6, "map6", _cx.numBuckets+1, tmp1Options, nullptr );
    _diskQueue->InitFileSet( FileId::MAP7, "map7", _cx.numBuckets+1, tmp1Options, nullptr );

    {
        const size_t cacheSize = _cx.cacheSize / 2;

        FileSetOptions opts = FileSetOptions::UseTemp2;

        if( !_cx.cfg->noTmp2DirectIO )
            opts |= FileSetOptions::DirectIO;

        if( _cx.cache )
            opts |= FileSetOptions::Cachable;

        FileSetInitData fdata;
        fdata.cache     = _cx.cache;
        fdata.cacheSize = cacheSize;

        _diskQueue->InitFileSet( FileId::FX0, "fx_0", _cx.numBuckets, opts, &fdata );
        
        fdata.cache = ((byte*)fdata.cache) + cacheSize;
        _diskQueue->InitFileSet( FileId::FX1, "fx_1", _cx.numBuckets, opts, &fdata );
    }
}

//-----------------------------------------------------------
void DiskPlotPhase1::Run()
{
    #if _DEBUG && ( BB_DP_DBG_SKIP_PHASE_1 || BB_DP_P1_SKIP_TO_TABLE )
    {
        FileStream bucketCounts, tableCounts, backPtrBucketCounts;

        if( bucketCounts.Open( BB_DP_DBG_TEST_DIR BB_DP_DBG_READ_BUCKET_COUNT_FNAME, FileMode::Open, FileAccess::Read ) )
        {
            if( bucketCounts.Read( _cx.bucketCounts, sizeof( _cx.bucketCounts ) ) != sizeof( _cx.bucketCounts ) )
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
            if( tableCounts.Read( _cx.entryCounts, sizeof( _cx.entryCounts ) ) != sizeof( _cx.entryCounts ) )
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
            if( backPtrBucketCounts.Read( _cx.ptrTableBucketCounts, sizeof( _cx.ptrTableBucketCounts ) ) != sizeof( _cx.ptrTableBucketCounts ) )
            {
                Fatal( "Failed to read from pointer bucket counts file." );
            }
        }
        else
        {
            Fatal( "Failed to open pointer bucket counts file." );
        }

        #if BB_DP_P1_SKIP_TO_TABLE
            goto FP;
        #endif

        #if BB_DP_DBG_SKIP_TO_C_TABLES
            goto CTables;
        #endif

        return;

    CONTINUE:;
    }
    #endif

    

#if !BB_DP_DBG_READ_EXISTING_F1
    GenF1();
#else
    {
        size_t pathLen = strlen( _cx.tmpPath );
        pathLen += sizeof( BB_DP_DBG_READ_BUCKET_COUNT_FNAME );

        std::string bucketsPath = _cx.tmpPath;
        if( bucketsPath[bucketsPath.length() - 1] != '/' && bucketsPath[bucketsPath.length() - 1] != '\\' )
            bucketsPath += "/";

        bucketsPath += BB_DP_DBG_READ_BUCKET_COUNT_FNAME;

        const size_t bucketsCountSize = sizeof( uint32 ) * BB_DP_BUCKET_COUNT;

        FileStream fBucketCounts;
        if( fBucketCounts.Open( bucketsPath.c_str(), FileMode::Open, FileAccess::Read ) )
        {

            size_t sizeRead = fBucketCounts.Read( _cx.bucketCounts[0], bucketsCountSize );
            FatalIf( sizeRead != bucketsCountSize, "Invalid bucket counts." );
        }
        else
        {
            GenF1();

            fBucketCounts.Close();
            FatalIf( !fBucketCounts.Open( bucketsPath.c_str(), FileMode::Create, FileAccess::Write ), "File to open bucket counts file" );
            FatalIf( fBucketCounts.Write( _cx.bucketCounts[0], bucketsCountSize ) != bucketsCountSize, "Failed to write bucket counts.");
        }
    }
#endif

    #if _DEBUG && BB_DP_DBG_VALIDATE_F1
    // if constexpr ( 0 )
    // {
    //     const uint32* bucketCounts = _cx.bucketCounts[0];
    //     uint64 totalEntries = 0;
    //     for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
    //         totalEntries += bucketCounts[i];
            
    //     ASSERT( totalEntries == 1ull << _K );

    //     Debug::ValidateYFileFromBuckets( FileId::Y0, *_cx.threadPool, *_diskQueue, TableId::Table1, _cx.bucketCounts[0] );
    // }
    #endif

#if _DEBUG && BB_DP_P1_SKIP_TO_TABLE
    FP:
#endif

    ForwardPropagate();

    // Check all table counts
    #if _DEBUG
    for( int table = (int)TableId::Table1; table <= (int)TableId::Table7; table++ )
    {
        uint64 entryCount = 0;

        for( int bucket = 0; bucket < (int)_cx.numBuckets; bucket++ )
            entryCount += _cx.bucketCounts[table][bucket];

        ASSERT( entryCount == _cx.entryCounts[table] );
    }
    #endif

    #if _DEBUG
    {
        // Write bucket counts
        FileStream bucketCounts, tableCounts, backPtrBucketCounts;

        if( bucketCounts.Open( BB_DP_DBG_TEST_DIR BB_DP_DBG_READ_BUCKET_COUNT_FNAME, FileMode::Create, FileAccess::Write ) )
        {
            if( bucketCounts.Write( _cx.bucketCounts, sizeof( _cx.bucketCounts ) ) != sizeof( _cx.bucketCounts ) )
                Log::Error( "Failed to write to bucket counts file." );
        }
        else
            Log::Error( "Failed to open bucket counts file." );

        if( tableCounts.Open( BB_DP_DBG_TEST_DIR BB_DP_TABLE_COUNTS_FNAME, FileMode::Create, FileAccess::Write ) )
        {
            if( tableCounts.Write( _cx.entryCounts, sizeof( _cx.entryCounts ) ) != sizeof( _cx.entryCounts ) )
                Log::Error( "Failed to write to table counts file." );
        }
        else
            Log::Error( "Failed to open table counts file." );

        if( backPtrBucketCounts.Open( BB_DP_DBG_TEST_DIR BB_DP_DBG_PTR_BUCKET_COUNT_FNAME, FileMode::Create, FileAccess::Write ) )
        {
            if( backPtrBucketCounts.Write( _cx.ptrTableBucketCounts, sizeof( _cx.ptrTableBucketCounts ) ) != sizeof( _cx.ptrTableBucketCounts ) )
                Log::Error( "Failed to write to back pointer bucket counts file." );
        }
        else
            Log::Error( "Failed to open back pointer bucket counts file." );
    }
    #endif

    Log::Line( " Phase 1 Total I/O wait time: %.2lf", TicksToSeconds( _cx.ioWaitTime ) + _cx.ioQueue->IOBufferWaitTime() );

#if BB_DP_DBG_SKIP_TO_C_TABLES
    CTables:
#endif
    WriteCTables();

    #if !BB_DP_P1_KEEP_FILES
        _cx.ioQueue->DeleteBucket( _fxIn );
        _cx.ioQueue->CommitCommands();
    #endif
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
    Log::Line( "Table 1 I/O wait time: %.2lf seconds.", _cx.ioQueue->IOBufferWaitTime() );

    #if BB_IO_METRICS_ON
        const double writeThroughput = _cx.ioQueue->GetAverageWriteThroughput();
        const auto&  writes          = _cx.ioQueue->GetWriteMetrics();

        Log::Line( " Table 1 I/O Metrics:" );
        Log::Line( "  Average write throughput %.2lf MiB ( %.2lf MB ) or %.2lf GiB ( %.2lf GB ).", 
            writeThroughput BtoMB, writeThroughput / 1000000.0, writeThroughput BtoGB, writeThroughput / 1000000000.0 );
        Log::Line( "  Total size written: %.2lf MiB ( %.2lf MB ) or %.2lf GiB ( %.2lf GB ).",
            (double)writes.size BtoMB, (double)writes.size / 1000000.0, (double)writes.size BtoGB, (double)writes.size / 1000000000.0 );
        Log::Line( "  Total write commands: %llu.", (llu)writes.count );
        Log::Line( "" );

        _cx.ioQueue->ClearWriteMetrics();
    #endif
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
    TableId startTable = TableId::Table2;

    #if BB_DP_P1_SKIP_TO_TABLE
        startTable = BB_DP_P1_START_TABLE;
        if( (int)startTable ^ 1 )
            std::swap( _fxIn, _fxOut );
    #endif

    for( TableId table = startTable; table <= TableId::Table7; table++ )
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
        Log::Line( "Completed table %u in %.2lf seconds with %.llu entries.", table+1, TimerEnd( timer ), _cx.entryCounts[(int)table] );
        Log::Line( "Table %u I/O wait time: %.2lf seconds.",  table+1, TicksToSeconds( _tableIOWaitTime ) );

        std::swap( _fxIn, _fxOut );

        // No longer need fxout. Delete it
        if( table == TableId::Table7 )
        {
            #if !BB_DP_P1_KEEP_FILES
            _cx.ioQueue->DeleteBucket( _fxOut );
            _cx.ioQueue->CommitCommands();
            #endif
        }
        
        #if BB_IO_METRICS_ON
            const double readThroughput  = _cx.ioQueue->GetAverageReadThroughput();
            const auto&  reads           = _cx.ioQueue->GetReadMetrics();
            const double writeThroughput = _cx.ioQueue->GetAverageWriteThroughput();
            const auto&  writes          = _cx.ioQueue->GetWriteMetrics();

            Log::Line( " Table %u I/O Metrics:", (uint32)table+1 );
            
            Log::Line( "  Average read throughput %.2lf MiB ( %.2lf MB ) or %.2lf GiB ( %.2lf GB ).", 
                readThroughput BtoMB, readThroughput / 1000000.0, readThroughput BtoGB, readThroughput / 1000000000.0 );
            Log::Line( "  Total size read: %.2lf MiB ( %.2lf MB ) or %.2lf GiB ( %.2lf GB ).",
                (double)reads.size BtoMB, (double)reads.size / 1000000.0, (double)reads.size BtoGB, (double)reads.size / 1000000000.0 );
            Log::Line( "  Total read commands: %llu.", (llu)reads.count );

            Log::Line( "  Average write throughput %.2lf MiB ( %.2lf MB ) or %.2lf GiB ( %.2lf GB ).", 
                writeThroughput BtoMB, writeThroughput / 1000000.0, writeThroughput BtoGB, writeThroughput / 1000000000.0 );
            Log::Line( "  Total size written: %.2lf MiB ( %.2lf MB ) or %.2lf GiB ( %.2lf GB ).",
                (double)writes.size BtoMB, (double)writes.size / 1000000.0, (double)writes.size BtoGB, (double)writes.size / 1000000000.0 );
            Log::Line( "  Total write commands: %llu.", (llu)writes.count );
            Log::Line( "" );

            _cx.ioQueue->ClearReadMetrics();
            _cx.ioQueue->ClearWriteMetrics();
        #endif
    }
}

//-----------------------------------------------------------
template<TableId table>
void DiskPlotPhase1::ForwardPropagateTable()
{
    _tableIOWaitTime = Duration::zero();

    // Above table 6 the metadata is < x4, let's truncate the output file
    // so that we can recover some space from the x4 metadata from the previous tables
    if( table >= TableId::Table6 )
    {
        _cx.ioQueue->TruncateBucket( _fxOut, 0 );
        _cx.ioQueue->CommitCommands();
    }

    switch( _cx.numBuckets )
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

    _tableIOWaitTime = fp.IOWaitTime();
    _cx.ioWaitTime   += _tableIOWaitTime;

    #if BB_DP_DBG_VALIDATE_FX
        #if !_DEBUG
            Log::Line( "Warning: Table validation enabled in release mode." );
        #endif
        
        using TYOut = typename DiskFp<table, _numBuckets>::TYOut;
        Debug::ValidateYForTable<table, _numBuckets, TYOut>( _fxOut, *_cx.ioQueue, *_cx.threadPool, _cx.bucketCounts[(int)table] );
        Debug::ValidatePairs<256>( _cx, table );
    #endif

    #if BB_DP_DBG_DUMP_PAIRS
        if( table > TableId::Table2 )
            BB_DBG_DumpPairs( _numBuckets, table-1, _cx );

        if( table == TableId::Table7 )
            BB_DBG_DumpPairs( _numBuckets, table, _cx );
    #endif
}

//-----------------------------------------------------------
void DiskPlotPhase1::WriteCTables()
{
    switch( _cx.numBuckets )
    {
        case 128 : WriteCTablesBuckets<128 >(); break;
        case 256 : WriteCTablesBuckets<256 >(); break;
        case 512 : WriteCTablesBuckets<512 >(); break;
        case 1024: WriteCTablesBuckets<1024>(); break;
    
        default:
            Fatal( "Invalid bucket count." );
            break;
    }
}

//-----------------------------------------------------------
template<uint32 _numBuckets>
void DiskPlotPhase1::WriteCTablesBuckets()
{
    #if BB_DP_DBG_SKIP_TO_C_TABLES
        _fxIn  = FileId::FX0;
        _fxOut = FileId::FX1;

        #if BB_DP_DBG_VALIDATE_FX 
            #if !_DEBUG
                Log::Line( "Warning: Table validation enabled in release mode." );
            #endif
            
            using TYOut = typename DiskFp<TableId::Table7, _numBuckets>::TYOut;
            Debug::ValidateYForTable<TableId::Table7, _numBuckets, TYOut>( _fxIn, *_cx.ioQueue, *_cx.threadPool, _cx.bucketCounts[(int)TableId::Table7] );
        #endif
    #endif

    Log::Line( "Processing f7s and writing C tables to plot file." );

    const auto timer = TimerBegin();

    DiskFp<TableId::Table7, _numBuckets, true> fp( _cx, _fxIn, _fxOut );
    fp.RunF7();

    const double elapsed = TimerEnd( timer );
    Log::Line( "Completed C processing tables in %.2lf seconds.", elapsed );
    Log::Line( "C Tables I/O wait time: %.2lf.", TicksToSeconds( fp.IOWaitTime() ) );
}

