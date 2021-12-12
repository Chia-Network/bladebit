#include "DiskPlotPhase1.h"
#include "Util.h"
#include "util/Log.h"
#include "algorithm/RadixSort.h"
#include "pos/chacha8.h"
#include "b3/blake3.h"
#include "plotshared/GenSortKey.h"
#include "jobs/F1GenBucketized.h"
#include "jobs/FxGenBucketized.h"
#include "jobs/LookupMapJob.h"

// Test
#include "io/FileStream.h"
#include "SysHost.h"
#include "DiskPlotDebug.h"

#if _DEBUG
    #include "../memplot/DbgHelper.h"
#endif

// #TODO: Move there outside of here into a header
//        so that we can perform tests to determine best IO intervals

uint32 MatchEntries( const uint32* yBuffer, const uint32* groupBoundaries, Pairs pairs,
                     const uint32  groupCount, const uint32 startIndex, const uint32 maxPairs,
                     const uint64  bucketL, const uint64 bucketR );

FileId GetBackPointersFileIdForTable( TableId table );

//-----------------------------------------------------------
void DbgValidateFxBucket( const uint64 bucketIdx, uint32* entries, const uint64 entryCount )
{
    uint64* refEntries    = nullptr;
    uint64  refEntryCount = 0;

    {
        const char* refFilePath = "/mnt/p5510a/reference/t2.y";
        FileStream file;

        FatalIf( !file.Open( refFilePath, FileMode::Open, FileAccess::Read ), "Failed to open reference table file." );

        file.Read( &refEntryCount, sizeof( refEntryCount ) );
        file.Seek( (int64)file.BlockSize(), SeekOrigin::Begin );
        ASSERT( refEntryCount <= 1ull << _K );

        refEntries = bbcvirtalloc<uint64>( refEntryCount );

        size_t sizeToRead = sizeof( uint64 ) * refEntryCount;

        byte* dst = (byte*)refEntries;
        while( sizeToRead )
        {
            ssize_t sizeRead = file.Read( dst, sizeToRead );
            FatalIf( sizeRead < 1, "Read failed." );

            dst += sizeRead;
            sizeToRead -= (size_t)sizeRead;
        }
    }

    uint64 bucket = bucketIdx << 32;

    const uint64* refReader = refEntries;

    for( uint64 i = 0; i < entryCount; i++ )
    {
        const uint64 y    = bucket | entries[i];
        const uint64 refY = *refReader++;

        if( y != refY )
            Fatal( "Invalid y value @ %llu", i );
    }

    Log::Line( "Bucket verified successfully." );
}



//-----------------------------------------------------------
DiskPlotPhase1::DiskPlotPhase1( DiskPlotContext& cx )
    : _cx( cx )
    , _diskQueue( cx.ioQueue )
{
    ASSERT( cx.tmpPath );

    // Use the whole allocation for F1
    byte*  heap     = cx.heapBuffer;
    size_t heapSize = cx.heapSize + cx.ioHeapSize;

    _diskQueue->ResetHeap( heapSize, heap );
}

//-----------------------------------------------------------
void DiskPlotPhase1::Run()
{
    DiskPlotContext& cx = _cx;

    #if BB_DP_DBG_SKIP_PHASE_1
    {
        FileStream bucketCounts, tableCounts;

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

        return;

    CONTINUE:;
    }
    #endif

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

    // Re-create the disk queue with the io buffer only (remove working heap section)
    // #TODO: Remove this, this is for now while testing.
    _diskQueue->ResetHeap( cx.ioHeapSize, cx.ioHeap );

    ForwardPropagate();

    // Calculate all table counts
    for( int table = (int)TableId::Table1; table <= (int)TableId::Table7; table++ )
    {
        uint64 entryCount = 0;

        for( int bucket = 0; bucket < (int)BB_DP_BUCKET_COUNT; bucket++ )
            entryCount += cx.bucketCounts[table][bucket];

        cx.entryCounts[table] = entryCount;
    }

    #if _DEBUG
    {
        // Write bucket counts
        FileStream bucketCounts, tableCounts;

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
    }
    #endif
}


///
/// F1 Generation
///
//-----------------------------------------------------------
void DiskPlotPhase1::GenF1()
{
    DiskPlotContext& cx   = _cx;
    ThreadPool&      pool = *cx.threadPool;

    Log::Line( "Generating f1..." );
    auto timer = TimerBegin();
    
    F1GenBucketized::GenerateF1Disk( 
        cx.plotId, pool, cx.threadCount, *_diskQueue, 
        cx.writeIntervals[(int)TableId::Table1].fxGen, 
        cx.bucketCounts[0] );
    
    double elapsed = TimerEnd( timer );
    Log::Line( "Finished f1 generation in %.2lf seconds. ", elapsed );
}


///
/// Forward Propagate Tables
///
//-----------------------------------------------------------
inline void DiskPlotPhase1::GetWriteFileIdsForBucket( 
    TableId table, FileId& outYId,
    FileId& outMetaAId, FileId& outMetaBId )
{
    const bool isEven = static_cast<uint>( table ) & 1;

    outYId     = isEven ? FileId::Y1       : FileId::Y0;
    outMetaAId = isEven ? FileId::META_A_0 : FileId::META_A_1;
    outMetaBId = isEven ? FileId::META_B_0 : FileId::META_B_1;
}

//-----------------------------------------------------------
void DiskPlotPhase1::ForwardPropagate()
{
    DiskBufferQueue& ioQueue     = *_diskQueue;
    ThreadPool&      threadPool  = *_cx.threadPool;
    const uint       threadCount = _cx.threadCount;

    uint   maxBucketCount = 0;
    size_t maxBucketSize  = 0;
   
    // Find the largest bucket so that we can reserve buffers of its size
    for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
        maxBucketCount = std::max( maxBucketCount, _cx.bucketCounts[0][i] );

    ASSERT( maxBucketCount <= BB_DP_MAX_ENTRIES_PER_BUCKET );

    maxBucketCount = BB_DP_MAX_ENTRIES_PER_BUCKET;
    maxBucketSize   = maxBucketCount * sizeof( uint32 );
    _maxBucketCount =  maxBucketCount;

    // #TODO: We need to have a maximum size here, and just allocate that.
    //        we don't want to allocate per-table as we might not be able to do
    //        so due to fragmentation and the other buffers allocated after this big chunk.
    //        Therefore, it's better to have a set size
    // 
    // #TODO: Create a separate heap... Grab the section out of the working heap we need, and then create
    //        a new heap strictly with the space that will go to IO. No need to have these show up in the 
    //        heap as it will slow down deallocation.
    // 
    // Allocate buffers we need for forward propagation
//     DoubleBuffer bucketBuffers;

    Bucket bucket;
    _bucket = &bucket;

    {
        const size_t fileBlockSize = ioQueue.BlockSize();

        // Extra space used to store the last 2 groups of y entries.
        // const size_t yGroupExtra = RoundUpToNextBoundary( kBC * sizeof( uint32 ) * 2ull, (int)fileBlockSize );

        const size_t ySize       = RoundUpToNextBoundary( maxBucketCount * sizeof( uint32 ) * 2, (int)fileBlockSize );
        const size_t sortKeySize = RoundUpToNextBoundary( maxBucketCount * sizeof( uint32 )    , (int)fileBlockSize );
        const size_t mapSize     = RoundUpToNextBoundary( maxBucketCount * sizeof( uint64 )    , (int)fileBlockSize );
        const size_t metaSize    = RoundUpToNextBoundary( maxBucketCount * sizeof( uint64 ) * 4, (int)fileBlockSize );
        const size_t pairsLSize  = RoundUpToNextBoundary( maxBucketCount * sizeof( uint32 )    , (int)fileBlockSize );
        const size_t pairsRSize  = RoundUpToNextBoundary( maxBucketCount * sizeof( uint16 )    , (int)fileBlockSize );
        const size_t groupsSize  = RoundUpToNextBoundary( ( maxBucketCount + threadCount * 2 ) * sizeof( uint32), (int)fileBlockSize );

        const size_t totalSize = ySize + sortKeySize + mapSize + metaSize + pairsLSize + pairsRSize + groupsSize;

        Log::Line( "Reserving %.2lf MiB for forward propagation.", (double)totalSize BtoMB );


        // #TODO: Remove this. Allocating here temporarily for testing.
        // Temp test:
        bucket.yTmp     = bbvirtalloc<uint32>( ySize    );
        bucket.metaATmp = bbvirtalloc<uint64>( metaSize );
        bucket.metaBTmp = bucket.metaATmp + maxBucketCount;

        // #TODO: Remove this as well. Testing allocating here for now. It should be allocated as part of the IO heap.
        bucket.yOverflow    .Init( bbvirtalloc<void>( fileBlockSize * BB_DP_BUCKET_COUNT * 2 ), fileBlockSize );
        bucket.metaAOverflow.Init( bbvirtalloc<void>( fileBlockSize * BB_DP_BUCKET_COUNT * 2 ), fileBlockSize );
        bucket.metaBOverflow.Init( bbvirtalloc<void>( fileBlockSize * BB_DP_BUCKET_COUNT * 2 ), fileBlockSize );

        // #TODO: Remove these also. Testing the allocation here for now
        bucket.crossBucketInfo.y           = bbcvirtalloc<uint32>( kBC * 6 );
        bucket.crossBucketInfo.metaA       = bbcvirtalloc<uint64>( kBC * 6 );
        bucket.crossBucketInfo.metaB       = bbcvirtalloc<uint64>( kBC * 6 );
        bucket.crossBucketInfo.pairs.left  = bbcvirtalloc<uint32>( kBC );
        bucket.crossBucketInfo.pairs.right = bbcvirtalloc<uint16>( kBC );

        bucket.fpBuffer = _cx.heapBuffer;

        byte* ptr = bucket.fpBuffer;

        // Offset by yGroupExtra so that we're aligned to file block size for reading,
        // but can store the previous bucket's 2 groups worth of y values just before that.

        bucket.y0 = (uint32*)ptr; 
        bucket.y1 = bucket.y0 + maxBucketCount;
        ptr += ySize;

        bucket.sortKey = (uint32*)ptr;
        ptr += sortKeySize;

        bucket.map = (uint32*)ptr;
        ptr += mapSize;

        bucket.metaA0 = (uint64*)ptr;
        bucket.metaA1 = bucket.metaA0 + maxBucketCount;
        bucket.metaB0 = bucket.metaA1 + maxBucketCount;
        bucket.metaB1 = bucket.metaB0 + maxBucketCount;
        ptr += metaSize;
        ASSERT( ptr == (byte*)( bucket.metaB1 + maxBucketCount ) );

        bucket.pairs.left = (uint32*)ptr;
        ptr += pairsLSize;

        bucket.pairs.right = (uint16*)ptr;
        ptr += pairsRSize;

        // bucket.groupBoundaries =  (uint32*)ptr;
        // #TODO: Remove this, testing only
        {
            const size_t pageSize = SysHost::GetPageSize();

            byte* groupBytes =  bbvirtalloc<byte>( groupsSize + pageSize * 2 );
            bucket.groupBoundaries = (uint32*)( groupBytes + pageSize );

            SysHost::VirtualProtect( groupBytes, pageSize );
            SysHost::VirtualProtect( groupBytes + pageSize + groupsSize, pageSize );
        }

        // The remainder for the work heap is used to write as fx disk write buffers 
    }

    // Set back pointers fence initially
    bucket.backPointersFence.Signal();


    /// Propagate to each table
    for( TableId table = TableId::Table2; table <= TableId::Table7; table++ )
    {
        const bool isEven = ( (uint)table ) & 1;

        bucket.yFileId     = isEven ? FileId::Y0       : FileId::Y1;
        bucket.metaAFileId = isEven ? FileId::META_A_1 : FileId::META_A_0;
        bucket.metaBFileId = isEven ? FileId::META_B_1 : FileId::META_B_0;
        
        // Reset the entry count for the table
        bucket.tableEntryCount = 0;

        // Seek buckets to the start and load the first y bucket
        Log::Line( "Forward propagating to table %d...", (int)table + 1 );
        const auto tableTimer = TimerBegin();

        switch( table )
        {
            case TableId::Table2: ForwardPropagateTable<TableId::Table2>(); break;
            case TableId::Table3: ForwardPropagateTable<TableId::Table3>(); break;
            case TableId::Table4: ForwardPropagateTable<TableId::Table4>(); break;
            case TableId::Table5: ForwardPropagateTable<TableId::Table5>(); break;
            case TableId::Table6: ForwardPropagateTable<TableId::Table6>(); break;
            case TableId::Table7: ForwardPropagateTable<TableId::Table7>(); break;
        }

        const double tableElapsed = TimerEnd( tableTimer );
        Log::Line( "Finished forward propagating table %d in %.2lf seconds.", (int)table + 1, tableElapsed );

        #if _DEBUG && BB_DP_DBG_VALIDATE_FX
        // if( 0 )
        // if( table >= TableId::Table6 )
        {
            Log::Line( "Validating table %d...", (int)table + 1 );
            const FileId fileId = isEven ? FileId::Y1 : FileId::Y0;
            Debug::ValidateYFileFromBuckets( fileId, *_cx.threadPool, *_diskQueue, table, _cx.bucketCounts[(int)table] );
        }
        #endif
    }

    // #TODO: Ensure last writes completed.
}


//-----------------------------------------------------------
template<TableId tableId>
void DiskPlotPhase1::ForwardPropagateTable()
{
    constexpr size_t MetaInASize = TableMetaIn<tableId>::SizeA;
    constexpr size_t MetaInBSize = TableMetaIn<tableId>::SizeB;

    DiskPlotContext& cx                = _cx;
    DiskBufferQueue& ioQueue           = *_diskQueue;
    Bucket&          bucket            = *_bucket;
    const uint       threadCount       = _cx.threadCount;
    const uint32*    inputBucketCounts = cx.bucketCounts[(uint)tableId - 1];

    // const FileId sortKeyFileId = TableIdToSortKeyId( (TableId)((int)tableId - 1) );

    // Set the correct file id, given the table (we swap between them for each table)
    {
        const bool isEven  = ( (uint)tableId ) & 1;

        bucket.yFileId     = isEven ? FileId::Y0       : FileId::Y1;
        bucket.metaAFileId = isEven ? FileId::META_A_1 : FileId::META_A_0;
        bucket.metaBFileId = isEven ? FileId::META_B_1 : FileId::META_B_0;

        if constexpr ( tableId == TableId::Table2 )
        {
            bucket.metaAFileId = FileId::X;
        }
    }

    // Seek all buckets to the start
    ioQueue.SeekBucket( FileId::Y0, 0, SeekOrigin::Begin );
    ioQueue.SeekBucket( FileId::Y1, 0, SeekOrigin::Begin );

    if constexpr( tableId == TableId::Table2 )
    {
        ioQueue.SeekBucket( FileId::X, 0, SeekOrigin::Begin );
    }
    else
    {
        ioQueue.SeekBucket( FileId::META_A_0, 0, SeekOrigin::Begin );
        ioQueue.SeekBucket( FileId::META_A_1, 0, SeekOrigin::Begin );
        ioQueue.SeekBucket( FileId::META_B_0, 0, SeekOrigin::Begin );
        ioQueue.SeekBucket( FileId::META_B_1, 0, SeekOrigin::Begin );
    }
    ioQueue.CommitCommands();

    // Read first bucket y & metadata values
    bucket.fence.Reset( FPFenceId::Start );

    ioQueue.ReadFile( bucket.yFileId, 0, bucket.y0, inputBucketCounts[0] * sizeof( uint32 ) );
    ioQueue.SignalFence( bucket.fence, FPFenceId::YLoaded );
    ioQueue.CommitCommands();

    // if constexpr ( tableId > TableId::Table2 )
    // {
    //     ioQueue.ReadFile( sortKeyFileId, 0, bucket.sortKey, inputBucketCounts[0] * sizeof( uint32 ) );
    //     ioQueue.SignalFence( bucket.fence, FPFenceId::FirstSortKeyLoaded );
    // }

    ioQueue.ReadFile( bucket.metaAFileId, 0, bucket.metaA0, inputBucketCounts[0] * MetaInASize );
    ioQueue.SignalFence( bucket.fence, FPFenceId::MetaALoaded );

    if constexpr ( MetaInBSize > 0 )
    {
        ioQueue.ReadFile( bucket.metaBFileId, 0, bucket.metaB0, inputBucketCounts[0] * MetaInBSize );
        ioQueue.SignalFence( bucket.fence, FPFenceId::MetaBLoaded );
    }

    ioQueue.CommitCommands();

    for( uint bucketIdx = 0; bucketIdx < BB_DP_BUCKET_COUNT; bucketIdx++ )
    {
        Log::Line( " Processing bucket %-2u", bucketIdx );

        const uint entryCount = inputBucketCounts[bucketIdx];
        ASSERT( entryCount < _maxBucketCount );

        // Read the next bucket in the background if we're not at the last bucket
        const uint nextBucketIdx = bucketIdx + 1;

        if( nextBucketIdx < BB_DP_BUCKET_COUNT )
        {
            const size_t nextBufferCount = inputBucketCounts[nextBucketIdx];

            ioQueue.ReadFile( bucket.yFileId, nextBucketIdx, bucket.y1, nextBufferCount * sizeof( uint32 ) );
            ioQueue.SignalFence( bucket.fence, FPFenceId::YLoaded + nextBucketIdx );
            ioQueue.ReadFile( bucket.metaAFileId, nextBucketIdx, bucket.metaA1, nextBufferCount * MetaInASize );
            ioQueue.SignalFence( bucket.fence, FPFenceId::MetaALoaded + nextBucketIdx );

            if constexpr ( MetaInBSize > 0 )
            {
                // Don't load the metadata B yet, we will use the metadata B back buffer as our temporary buffer for sorting
                ioQueue.WaitForFence( bucket.ioFence );
                ioQueue.ReadFile( bucket.metaBFileId, nextBucketIdx, bucket.metaB1, nextBufferCount * MetaInBSize );
                ioQueue.SignalFence( bucket.fence, FPFenceId::MetaBLoaded + nextBucketIdx );

                // #TODO: Maybe we should just allocate .5 GiB more for the temp buffers?
                //        for now, try it this way.
            }

            ioQueue.CommitCommands();
        }

        // Forward propagate this bucket
        const uint32 bucketEntryCount = ForwardPropagateBucket<tableId>( bucketIdx, bucket, entryCount );
        bucket.tableEntryCount += bucketEntryCount;


        // Swap are front/back buffers
        std::swap( bucket.y0    , bucket.y1     );
        std::swap( bucket.metaA0, bucket.metaA1 );
        std::swap( bucket.metaB0, bucket.metaB1 );
    }
}

//-----------------------------------------------------------
template<TableId tableId>
uint32 DiskPlotPhase1::ForwardPropagateBucket( uint32 bucketIdx, Bucket& bucket, uint32 entryCount )
{
    constexpr size_t MetaInASize = TableMetaIn<tableId>::SizeA;
    constexpr size_t MetaInBSize = TableMetaIn<tableId>::SizeB;

    DiskPlotContext& cx          = _cx;
    DiskBufferQueue& ioQueue     = *_diskQueue;
    ThreadPool&      threadPool  = *cx.threadPool;
    const uint32     threadCount = cx.threadCount;

    uint crossBucketMatches = 0;

    // To avoid confusion as we sort into metaTmp, and then use bucket.meta0 as tmp for fx
    // we explicitly set them to appropriately-named variables here
    uint64* fxMetaInA  = bucket.metaATmp;
    uint64* fxMetaInB  = bucket.metaBTmp;
    uint64* fxMetaOutA = bucket.metaA0;
    uint64* fxMetaOutB = bucket.metaB0;

    ///
    /// Sort our current bucket
    ///
    uint32* sortKey = bucket.sortKey;
    {
        Log::Line( "  Sorting bucket y." );
        auto timer = TimerBegin();

        if constexpr ( tableId == TableId::Table2 )
        {
            // No sort key needed for table 1, just sort x along with y
            sortKey    = (uint32*)bucket.metaA0;

            fxMetaInA  = bucket.metaA0;
            fxMetaInB  = bucket.metaB0;
            fxMetaOutA = bucket.metaATmp;
            fxMetaOutB = bucket.metaBTmp;

            // Ensure Meta A has been loaded (which for table to is just x)
            bucket.fence.Wait( FPFenceId::MetaALoaded );
        }
        else
        {
            // Generate a sort key
            SortKeyGen::Generate<BB_MAX_JOBS>( threadPool, entryCount, sortKey );

            // Ensure Y as been loaded
            bucket.fence.Wait( FPFenceId::YLoaded + bucketIdx );
        }

        uint32* yTemp       = (uint32*)bucket.metaB1;
        uint32* sortKeyTemp = yTemp + entryCount;

        RadixSort256::SortWithKey<BB_MAX_JOBS>( threadPool, bucket.y0, yTemp, sortKey, sortKeyTemp, entryCount );

        // #TODO: Write sort key to disk as the previous table's sort key, so that we can do a quick sort of L/R later.

        // OK to load next (back) metadata B buffer now (see comment above in ForwardPropagateTable)
        if constexpr ( MetaInBSize > 0 )
            bucket.ioFence.Signal();

        #if _DEBUG
            //ASSERT( DbgVerifyGreater( entryCount, bucket.y0 ) );
        #endif

        double elapsed = TimerEnd( timer );
        Log::Line( "  Sorted bucket y in %.2lf seconds.", elapsed );
    }

    ///
    /// Matching
    ///
    GroupInfo groupInfos[BB_MAX_JOBS];
    uint32 matchCount = MatchBucket( tableId, bucketIdx, bucket, entryCount, groupInfos );


    ///
    /// Sort metadata with the key
    /// NOTE: MatchBucket make use of the metaATmp buffer, so we have to sort this after.
    ///       Plus it gives us extra time for the metadata to load since matching doesn't require it.
    if constexpr ( tableId > TableId::Table2 )
    {
        using TMetaA = typename TableMetaIn<tableId>::MetaA;

        bucket.fence.Wait( FPFenceId::MetaALoaded + bucketIdx );
        SortKeyGen::Sort<BB_MAX_JOBS, TMetaA>( threadPool, (int64)entryCount, sortKey, (const TMetaA*)bucket.metaA0, (TMetaA*)fxMetaInA );

        if constexpr ( MetaInBSize > 0 )
        {
            using TMetaB = typename TableMetaIn<tableId>::MetaB;

            bucket.fence.Wait( FPFenceId::MetaBLoaded + bucketIdx );
            SortKeyGen::Sort<BB_MAX_JOBS, TMetaB>( threadPool, (int64)entryCount, sortKey, (const TMetaB*)bucket.metaB0, (TMetaB*)fxMetaInB );
        }

        // #NOTE: For Debugging/Validation
        #if _DEBUG && BB_DP_DBG_VALIDATE_META
        if( tableId > TableId::Table2 )
        {
            Debug::ValidateMetaFileFromBuckets( fxMetaInA, fxMetaInB, tableId-(TableId)1, entryCount, bucketIdx, 
                                                _cx.bucketCounts[(int)tableId-1] );
        }
        #endif
    }

    ///
    /// Adjacent-bucket matching.
    //   Now that the current bucket y & meta is sorted, we can do matches
    //   and generate entries with matches crossing bucket boundaries
    /// 
    if( bucketIdx > 0 )
    {
        // Generate matches that were 
        crossBucketMatches = ProcessAdjoiningBuckets<tableId>( bucketIdx, bucket, entryCount, bucket.y0, fxMetaInA, fxMetaInB );
    }

    ///
    /// FX
    ///
    {
        Log::Line( "  Generating fx..." );
        const auto timer = TimerBegin();

        const size_t chunkSize     = cx.writeIntervals[(int)tableId].fxGen;
        const uint32 sortKeyOffset = bucket.tableEntryCount + crossBucketMatches;

        FxGenBucketized<tableId>::GenerateFxBucketizedToDisk(
            ioQueue,
            chunkSize,
            threadPool,
            cx.threadCount,
            bucketIdx,
            matchCount,
            sortKeyOffset,
            bucket.pairs,
            (byte*)bucket.sortKey,                   // #TODO: Change this, for now use sort key buffer
            bucket.y0  , fxMetaInA , fxMetaInB,
            bucket.yTmp, fxMetaOutA, fxMetaOutB,
            cx.bucketCounts[(uint)tableId]
        );

        const double elapsed = TimerEnd( timer );
        Log::Line( "  Finished generating fx in %.2lf seconds.", elapsed );
    }

    ///
    /// Save the last 2 groups worth of data for this bucket.
    ///
     // If not the last group, save info to match adjacent bucket groups (cross-bucket matches)
    if( bucketIdx + 1 < BB_DP_BUCKET_COUNT )
    {
        // Copy over the last 2 groups worth of y values to 
        // our new bucket's y buffer. There's space reserved before the start
        // of our y buffers that allow for it.
        const GroupInfo& lastThreadGrp = groupInfos[threadCount-1];

        AdjacentBucketInfo& crossBucketInfo = bucket.crossBucketInfo;

        const uint32 penultimateGroupIdx = lastThreadGrp.groupBoundaries[lastThreadGrp.groupCount-2];
        const uint32 lastGroupIdx        = lastThreadGrp.groupBoundaries[lastThreadGrp.groupCount-1];

        crossBucketInfo.groupCounts[0]   = lastGroupIdx - penultimateGroupIdx;
        crossBucketInfo.groupCounts[1]   = entryCount - lastGroupIdx;

        // // Copy over the last 2 groups worth of entries to the reserved area of our current bucket
        const uint32 last2GroupEntryCount = crossBucketInfo.groupCounts[0] + crossBucketInfo.groupCounts[1];
        ASSERT( last2GroupEntryCount <= kBC * 2 );

        const byte* bytesMetaA = (byte*)fxMetaInA;
        const byte* bytesMetaB = (byte*)fxMetaInB;

        bbmemcpy_t( crossBucketInfo.y, bucket.y0 + penultimateGroupIdx, last2GroupEntryCount );

        memcpy( crossBucketInfo.metaA, bytesMetaA + penultimateGroupIdx * MetaInASize, last2GroupEntryCount * MetaInASize );

        if( MetaInBSize )
            memcpy( crossBucketInfo.metaB, bytesMetaB + penultimateGroupIdx * MetaInBSize, last2GroupEntryCount * MetaInBSize );
    }

    return matchCount + crossBucketMatches;
}


///
/// Write map from the sort key
///
//-----------------------------------------------------------
void DiskPlotPhase1::WriteReverseMap( const uint32 count, const uint32* sortKey )
{
    

    
}


///
/// Adjacent Bucket Groups
///
//-----------------------------------------------------------
template<TableId tableId>
uint32 DiskPlotPhase1::ProcessAdjoiningBuckets( uint32 bucketIdx, Bucket& bucket, uint32 entryCount,
                                                const uint32* curY, const uint64* curMetaA, const uint64* curMetaB )
{
    using TMetaA = typename TableMetaIn<tableId>::MetaA;
    using TMetaB = typename TableMetaIn<tableId>::MetaB;

    constexpr size_t MetaInASize = TableMetaIn<tableId>::SizeA;
    constexpr size_t MetaInBSize = TableMetaIn<tableId>::SizeB;

    AdjacentBucketInfo& crossBucketInfo = bucket.crossBucketInfo;

    // #TODO: Get the current maximum entries of the previous bucket?
    uint32 maxEntries = BB_DP_MAX_ENTRIES_PER_BUCKET;

    const uint32  prevBucket     = bucketIdx - 1;

    uint32        prevGroupCount = crossBucketInfo.groupCounts[0];
    const uint32* prevGroupY     = crossBucketInfo.y;
    const TMetaA* prevGroupMetaA = (TMetaA*)crossBucketInfo.metaA;
    const TMetaB* prevGroupMetaB = (TMetaB*)crossBucketInfo.metaB;
    const uint32* curGroupY      = curY;
    const TMetaA* curGroupMetaA  = (TMetaA*)curMetaA;
    const TMetaB* curGroupMetaB  = (TMetaB*)curMetaB;

    byte*         bucketIndices  = (byte*)bucket.sortKey;   // #TODO: Use its own buffer, instead of sort key
    Pairs         pairs          = bucket.crossBucketInfo.pairs;
    uint32*       yTmp           = crossBucketInfo.y + kBC * 2;
    TMetaA*       metaATmp       = (TMetaA*)( crossBucketInfo.metaA + kBC * 2 );
    TMetaB*       metaBTmp       = (TMetaB*)( crossBucketInfo.metaB + kBC * 2 );
    uint32*       yOut           = crossBucketInfo.y + kBC * 4;
    uint64*       metaAOut       = (uint64*)( crossBucketInfo.metaA + kBC * 4 );
    uint64*       metaBOut       = (uint64*)( crossBucketInfo.metaB + kBC * 4 );

    // Get matches between the adjoining groups
    uint32 curGroupCount     = 0;
    uint32 matchesSecondLast = 0;
    uint32 matchesLast       = 0;
    
    uint32 sortKeyOffset     = bucket.tableEntryCount;

    // Process penultimate group from the previous bucket & 
    // the first group from the current bucket
    matchesSecondLast = ProcessCrossBucketGroups<tableId>(
        prevGroupY, 
        prevGroupMetaA, 
        prevGroupMetaB,
        curGroupY, 
        curGroupMetaA, 
        curGroupMetaB,
        yTmp, 
        metaATmp, 
        metaBTmp,
        yOut,
        metaAOut,
        metaBOut,
        prevGroupCount, 
        entryCount,
        prevBucket, 
        bucketIdx,
        pairs, 
        maxEntries,
        sortKeyOffset,
        curGroupCount );

    // Process the last group from the previous bucket &
    // the second group from the current bucket

    prevGroupY     += prevGroupCount;
    prevGroupMetaA += prevGroupCount;
    prevGroupMetaB += prevGroupCount;
    curGroupY      += curGroupCount;
    curGroupMetaA  += curGroupCount;
    curGroupMetaB  += curGroupCount;
    
    maxEntries     -= matchesSecondLast;
    entryCount     -= curGroupCount;
    
    sortKeyOffset  += matchesSecondLast;

    prevGroupCount = crossBucketInfo.groupCounts[1];

    matchesLast = ProcessCrossBucketGroups<tableId>(
        prevGroupY,
        prevGroupMetaA,
        prevGroupMetaB,
        curGroupY,
        curGroupMetaA,
        curGroupMetaB,
        yTmp,
        metaATmp,
        metaBTmp,
        yOut,
        metaAOut,
        metaBOut,
        prevGroupCount,
        entryCount,
        prevBucket,
        bucketIdx,
        pairs,
        maxEntries,
        sortKeyOffset,
        curGroupCount );


    const uint32 crossMatches = matchesSecondLast + matchesLast;
    return crossMatches;
}


//-----------------------------------------------------------
template<TableId tableId, typename TMetaA, typename TMetaB>
uint32 DiskPlotPhase1::ProcessCrossBucketGroups(
    const uint32* prevBucketY,
    const TMetaA* prevBucketMetaA,
    const TMetaB* prevBucketMetaB,
    const uint32* curBucketY,
    const TMetaA* curBucketMetaA,
    const TMetaB* curBucketMetaB,
    uint32*       tmpY,
    TMetaA*       tmpMetaA,
    TMetaB*       tmpMetaB,
    uint32*       outY,
    uint64*       outMetaA,
    uint64*       outMetaB,
    uint32        prevBucketGroupCount,
    uint32        curBucketEntryCount,
    uint32        prevBucketIndex,
    uint32        curBucketIndex,
    Pairs         pairs,
    uint32        maxPairs,
    uint32        sortKeyOffset,
    uint32&       outCurGroupCount
)
{
    const size_t MetaInASize  = TableMetaIn<tableId>::SizeA;
    const size_t MetaInBSize  = TableMetaIn<tableId>::SizeB;
    const size_t MetaOutASize = TableMetaOut<tableId>::SizeA;
    const size_t MetaOutBSize = TableMetaOut<tableId>::SizeB;

    using TMetaOutA = typename TableMetaOut<tableId>::MetaA;
    using TMetaOutB = typename TableMetaOut<tableId>::MetaB;

    ASSERT( MetaInASize );

    const uint64 lBucket    = ((uint64)prevBucketIndex) << 32;
    const uint64 rBucket    = ((uint64)curBucketIndex ) << 32;

    const uint64 lBucketGrp = ( lBucket | *prevBucketY ) / kBC;
    const uint64 rBucketGrp = ( rBucket | *curBucketY  ) / kBC;

    ASSERT( rBucketGrp > lBucketGrp );
    outCurGroupCount = 0;

    // Ensure the groups are adjacent
    if( rBucketGrp - lBucketGrp > 1 )
        return 0;

    byte* bucketIndices = (byte*)_bucket->sortKey;  // #TODO: Use a proper buffer for this

    // Find the length of the group
    uint32 curBucketGroupCount = 0;

    for( ; curBucketGroupCount < curBucketEntryCount; curBucketGroupCount++ )
    {
        const uint64 grp = ( rBucket | curBucketY[curBucketGroupCount] ) / kBC;

        if( grp != rBucketGrp )
            break;
    }

    ASSERT( curBucketGroupCount < curBucketEntryCount );
    ASSERT( curBucketGroupCount );

    // Copy the entries as adjacent entries into the tmp buffer
    bbmemcpy_t( tmpY, prevBucketY, prevBucketGroupCount );
    bbmemcpy_t( tmpY + prevBucketGroupCount, curBucketY, curBucketGroupCount );

    // Get matches
    const uint32 boundaries[2] = { prevBucketGroupCount, prevBucketGroupCount + curBucketGroupCount };
    const uint32 matches       = MatchEntries( tmpY, boundaries, pairs, 1, 0, maxPairs, lBucket, rBucket );

    // If we got matches, then complete processing them
    if( matches )
    {
        // Write pairs for table
        {
            uint32* lPairs = (uint32*)_diskQueue->GetBuffer( sizeof( uint32 ) * matches );
            uint16* rPairs = (uint16*)_diskQueue->GetBuffer( sizeof( uint16 ) * matches );

            // Copy pairs, fixing L pairs with the offset in the previous bucket
            for( uint i = 0; i < matches; i++ )
                lPairs[i] = pairs.left[i] + sortKeyOffset;

            bbmemcpy_t( rPairs, pairs.right, matches );

            // Write to disk
            const FileId leftId  = TableIdToBackPointerFileId( tableId );
            const FileId rightId = (FileId)( (int)leftId + 1 );
            
            _diskQueue->WriteFile( leftId , 0, lPairs, sizeof( uint32 ) * matches );
            _diskQueue->WriteFile( rightId, 0, rPairs, sizeof( uint16 ) * matches );
            _diskQueue->ReleaseBuffer( lPairs );
            _diskQueue->ReleaseBuffer( rPairs );
            _diskQueue->CommitCommands();
        }


        // Copy metadata
        bbmemcpy_t( tmpMetaA, prevBucketMetaA, prevBucketGroupCount );
        bbmemcpy_t( tmpMetaA + prevBucketGroupCount, curBucketMetaA, curBucketGroupCount );
        
        if constexpr ( MetaInBSize )
        {
            bbmemcpy_t( tmpMetaB, prevBucketMetaB, prevBucketGroupCount );
            bbmemcpy_t( tmpMetaB + prevBucketGroupCount, curBucketMetaB, curBucketGroupCount );
        }

        // Process Fx
        ComputeFxForTable<tableId>( 
            lBucket, matches, pairs, 
            tmpY, (uint64*)tmpMetaA, (uint64*)tmpMetaB,
            outY, bucketIndices, outMetaA, outMetaB, 
            0 );    // #TODO: Remove job Id. It's for testing

        // Count how many entries we have per buckets
        uint32 counts[BB_DP_BUCKET_COUNT];
        uint32 pfxSum[BB_DP_BUCKET_COUNT];

        memset( counts, 0, sizeof( uint32 ) * BB_DP_BUCKET_COUNT );
        for( const byte* ptr = bucketIndices, *end = ptr + matches; ptr < end; ptr++ )
        {
            ASSERT( *ptr <= ( 0b111111u ) );
            counts[*ptr] ++;
        }

        // Calculate prefix sum
        memcpy( pfxSum, counts, sizeof( uint32 ) * BB_DP_BUCKET_COUNT );
        for( uint i = 1; i < BB_DP_BUCKET_COUNT; i++ )
            pfxSum[i] += pfxSum[i-1];

        // Grab IO buffers
        uint32*    yBuckets     = nullptr;
        uint32*    sortKey      = nullptr;
        TMetaOutA* metaABuckets = nullptr;
        TMetaOutB* metaBBuckets = nullptr;

        yBuckets = (uint32*)_diskQueue->GetBuffer( sizeof( uint32 ) * matches );
        sortKey  = (uint32*)_diskQueue->GetBuffer( sizeof( uint32 ) * matches );

        if constexpr ( MetaOutASize )
            metaABuckets = (TMetaOutA*)_diskQueue->GetBuffer( MetaOutASize * matches );

        if constexpr ( MetaOutBSize )
            metaBBuckets = (TMetaOutB*)_diskQueue->GetBuffer( MetaOutBSize * matches );

        // Distribute entries into their respective buckets
        for( uint32 i = 0, key = sortKeyOffset; i < matches; i++, key++ )
        {
            const uint32 dstIdx = --pfxSum[bucketIndices[i]];

            yBuckets[dstIdx] = outY[i];
            sortKey [dstIdx] = key;

            if constexpr ( MetaInASize > 0 )
                metaABuckets[dstIdx] = reinterpret_cast<TMetaOutA*>( outMetaA )[i];

            if constexpr ( MetaOutBSize > 0 )
                metaBBuckets[dstIdx] = reinterpret_cast<TMetaOutB*>( outMetaB )[i];
        }

        // Write to disk
        FileId yId, keyFileId, metaAId, metaBId;
        GetWriteFileIdsForBucket( tableId, yId, metaAId, metaBId );
        keyFileId = TableIdToSortKeyId( tableId );

        {
            uint32* sizes = (uint32*)_diskQueue->GetBuffer( sizeof( uint32 ) * BB_DP_BUCKET_COUNT );
            for( uint32 i = 0; i < BB_DP_BUCKET_COUNT; i++ )
                sizes[i] = counts[i] * sizeof( uint32 );

            _diskQueue->WriteBuckets( yId      , yBuckets, sizes );
            _diskQueue->WriteBuckets( keyFileId, sortKey,  sizes );
            _diskQueue->ReleaseBuffer( yBuckets );
            _diskQueue->ReleaseBuffer( sortKey  );
            _diskQueue->ReleaseBuffer( sizes    );
            _diskQueue->CommitCommands();
        }

        if constexpr ( MetaOutASize > 0 )
        {
            uint32* sizes = (uint32*)_diskQueue->GetBuffer( sizeof( uint32 ) * BB_DP_BUCKET_COUNT );
            for( uint32 i = 0; i < BB_DP_BUCKET_COUNT; i++ )
                sizes[i] = counts[i] * MetaOutASize;

            _diskQueue->WriteBuckets( metaAId, metaABuckets, sizes );
            _diskQueue->ReleaseBuffer( metaABuckets );
            _diskQueue->ReleaseBuffer( sizes );
            _diskQueue->CommitCommands();
        }

        if constexpr ( MetaOutBSize > 0 )
        {
            uint32* sizes = (uint32*)_diskQueue->GetBuffer( sizeof( uint32 ) * BB_DP_BUCKET_COUNT );
            for( uint32 i = 0; i < BB_DP_BUCKET_COUNT; i++ )
                sizes[i] = counts[i] * MetaOutBSize;

            _diskQueue->WriteBuckets( metaBId, metaBBuckets, sizes );
            _diskQueue->ReleaseBuffer( metaBBuckets );
            _diskQueue->ReleaseBuffer( sizes );
            _diskQueue->CommitCommands();
        }

        // Save total bucket conts
        uint32* totalBucketCounts = _cx.bucketCounts[(int)tableId];
        for( uint32 i = 0; i < BB_DP_BUCKET_COUNT; i++ )
            totalBucketCounts[i] += counts[i];
    }

    outCurGroupCount = curBucketGroupCount;
    return matches;
}


///
/// Group Matching
///
//-----------------------------------------------------------
uint32 DiskPlotPhase1::MatchBucket( TableId table, uint32 bucketIdx, Bucket& bucket, uint32 entryCount, GroupInfo groupInfos[BB_MAX_JOBS] )
{
    const uint32 threadCount = _cx.threadCount;

    // Scan for group boundaries
    const uint32 groupCount = ScanGroups( bucketIdx, bucket.y0, entryCount, bucket.groupBoundaries, BB_DP_MAX_BC_GROUP_PER_BUCKET, groupInfos );
    
    // Produce per-thread matches in meta tmp. It has enough space to hold them.
    // Then move them over to a contiguous buffer.
    uint32* lPairs = (uint32*)bucket.metaATmp;
    uint16* rPairs = (uint16*)( lPairs + BB_DP_MAX_ENTRIES_PER_BUCKET );

    // Match pairs
    const uint32 entriesPerBucket  = (uint32)BB_DP_MAX_ENTRIES_PER_BUCKET;
    const uint32 maxPairsPerThread = entriesPerBucket / threadCount;

    for( uint i = 0; i < threadCount; i++ )
    {
        groupInfos[i].pairs.left  = lPairs + i * maxPairsPerThread;
        groupInfos[i].pairs.right = rPairs + i * maxPairsPerThread;
    }
    
    uint32 matchCount = Match( bucketIdx, maxPairsPerThread, bucket.y0, groupInfos );

    // #TODO: Make this multi-threaded... Testing for now
    // Copy matches to a contiguous buffer
    Pairs& pairs = bucket.pairs;

    uint32* lPtr = pairs.left;
    uint16* rPtr = pairs.right;

    for( uint i = 0; i < threadCount; i++ )
    {
        GroupInfo& group = groupInfos[i];
        bbmemcpy_t( lPtr, group.pairs.left, group.entryCount );
        lPtr += group.entryCount;
    }

    for( uint i = 0; i < threadCount; i++ )
    {
        GroupInfo& group = groupInfos[i];
        bbmemcpy_t( rPtr, group.pairs.right, group.entryCount );
        rPtr += group.entryCount;
    }
    
    // #TODO: Should we write these at intervals?
    const FileId idLeft  = GetBackPointersFileIdForTable( table );
    const FileId idRight = (FileId)( (int)idLeft + 1 );

    // Ensure the previous's bucket writes finished
    _bucket->backPointersFence.Wait();

    _diskQueue->WriteFile( idLeft , 0, pairs.left , matchCount * sizeof( uint32 ) );
    _diskQueue->WriteFile( idRight, 0, pairs.right, matchCount * sizeof( uint16 ) );
    _diskQueue->SignalFence( _bucket->backPointersFence );
    _diskQueue->CommitCommands();

    return matchCount;
}

//-----------------------------------------------------------
inline FileId GetBackPointersFileIdForTable( TableId table )
{
    switch( table )
    {
        case TableId::Table2: return FileId::T2_L;
        case TableId::Table3: return FileId::T3_L;
        case TableId::Table4: return FileId::T4_L;
        case TableId::Table5: return FileId::T5_L;
        case TableId::Table6: return FileId::T6_L;
        case TableId::Table7: return FileId::T7_L;

        default:
            ASSERT( 0 );
            return FileId::None;
    }
}

//-----------------------------------------------------------
uint32 DiskPlotPhase1::ScanGroups( uint bucketIdx, const uint32* yBuffer, uint32 entryCount, uint32* groups, uint32 maxGroups, GroupInfo groupInfos[BB_MAX_JOBS] )
{
    Log::Line( "  Scanning for groups." );

    auto& cx = _cx;

    ThreadPool& pool               = *cx.threadPool;
    const uint  threadCount        = _cx.threadCount;
    const uint  maxGroupsPerThread = maxGroups / threadCount - 1;   // -1 because we need to add an extra end index to check R group 
                                                                    // without adding an extra 'if'
    MTJobRunner<ScanGroupJob> jobs( pool );

    const uint64 bucket = ((uint64)bucketIdx) << 32;

    jobs[0].yBuffer         = yBuffer;
    jobs[0].groupBoundaries = groups;
    jobs[0].bucketIdx       = bucketIdx;
    jobs[0].startIndex      = 0;
    jobs[0].endIndex        = entryCount;
    jobs[0].maxGroups       = maxGroupsPerThread;
    jobs[0].groupCount      = 0;
    
    for( uint i = 1; i < threadCount; i++ )
    {
        ScanGroupJob& job = jobs[i];

        job.yBuffer         = yBuffer;
        job.groupBoundaries = groups + maxGroupsPerThread * i;
        job.bucketIdx       = bucketIdx;
        job.maxGroups       = maxGroupsPerThread;
        job.groupCount      = 0;

        const uint32 idx           = entryCount / threadCount * i;
        const uint64 y             = bucket | yBuffer[idx];
        const uint64 curGroup      = y / kBC;
        const uint64 groupLocalIdx = y - curGroup * kBC;

        uint64 targetGroup;

        // If we are already at the start of a group, just use this index
        if( groupLocalIdx == 0 )
        {
            job.startIndex = idx;
        }
        else
        {
            // Choose if we should find the upper boundary or the lower boundary
            const uint64 remainder = kBC - groupLocalIdx;
            
            #if _DEBUG
                bool foundBoundary = false;
            #endif
            if( remainder <= kBC / 2 )
            {
                // Look for the upper boundary
                for( uint32 j = idx+1; j < entryCount; j++ )
                {
                    targetGroup = (bucket | yBuffer[j] ) / kBC;
                    if( targetGroup != curGroup )
                    {
                        #if _DEBUG
                            foundBoundary = true;
                        #endif
                        job.startIndex = j; break;
                    }   
                }
            }
            else
            {
                // Look for the lower boundary
                for( uint32 j = idx-1; j >= 0; j-- )
                {
                    targetGroup = ( bucket | yBuffer[j] ) / kBC;
                    if( targetGroup != curGroup )
                    {
                        #if _DEBUG
                            foundBoundary = true;
                        #endif
                        job.startIndex = j+1; break;
                    }  
                }
            }

            ASSERT( foundBoundary );
        }

        auto& lastJob = jobs[i-1];
        ASSERT( job.startIndex > lastJob.startIndex );  // #TODO: This should not happen but there should
                                                        //        be a pre-check in the off chance that the thread count is really high.
                                                        //        Again, should not happen with the hard-coded thread limit,
                                                        //        but we can check if entryCount / threadCount <= kBC 


        // We add +1 so that the next group boundary is added to the list, and we can tell where the R group ends.
        lastJob.endIndex = job.startIndex + 1;

        ASSERT( ( bucket | yBuffer[job.startIndex-1] ) / kBC != 
                ( bucket | yBuffer[job.startIndex] ) / kBC );

        job.groupBoundaries = groups + maxGroupsPerThread * i;
    }

    // Fill in missing data for the last job
    jobs[threadCount-1].endIndex = entryCount;

    // Run the scan job
    const double elapsed = jobs.Run();
    Log::Line( "  Finished group scan in %.2lf seconds." );

    // Get the total group count
    uint groupCount = 0;

    for( uint i = 0; i < threadCount-1; i++ )
    {
        auto& job = jobs[i];

        // Add a trailing end index (but don't count it) so that we can test against it
        job.groupBoundaries[job.groupCount] = jobs[i+1].groupBoundaries[0];

        groupInfos[i].groupBoundaries = job.groupBoundaries;
        groupInfos[i].groupCount      = job.groupCount;
        groupInfos[i].startIndex      = job.startIndex;

        groupCount += job.groupCount;
    }
    
    // Let the last job know where its R group is
    auto& lastJob = jobs[threadCount-1];
    lastJob.groupBoundaries[lastJob.groupCount] = entryCount;

    groupInfos[threadCount-1].groupBoundaries = lastJob.groupBoundaries;
    groupInfos[threadCount-1].groupCount      = lastJob.groupCount;
    groupInfos[threadCount-1].startIndex      = lastJob.startIndex;

    Log::Line( "  Found %u groups.", groupCount );

    return groupCount;
}

//-----------------------------------------------------------
uint32 DiskPlotPhase1::Match( uint bucketIdx, uint maxPairsPerThread, const uint32* yBuffer, GroupInfo groupInfos[BB_MAX_JOBS] )
{
    Log::Line( "  Matching groups." );

    auto&      cx          = _cx;
    const uint threadCount = cx.threadCount;

    MTJobRunner<MatchJob> jobs( *cx.threadPool );

    for( uint i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];

        job.yBuffer         = yBuffer;
        job.bucketIdx       = bucketIdx;
        job.maxPairCount    = maxPairsPerThread;
        job.pairCount       = 0;
        job.groupInfo       = &groupInfos[i];
        job.copyLDst        = nullptr;
        job.copyRDst        = nullptr;
    }

    const double elapsed = jobs.Run();

    uint32 matches = jobs[0].pairCount;
    for( uint i = 1; i < threadCount; i++ )
        matches += jobs[i].pairCount;

    Log::Line( "  Found %u matches in %.2lf seconds.", matches, elapsed );
    return matches;
}

//-----------------------------------------------------------
void ScanGroupJob::Run()
{
    const uint32 maxGroups = this->maxGroups;
    
    uint32* groupBoundaries = this->groupBoundaries;
    uint32  groupCount      = 0;

    const uint32* yBuffer = this->yBuffer;
    const uint32  start   = this->startIndex;
    const uint32  end     = this->endIndex;

    const uint64  bucket  = ( (uint64)this->bucketIdx ) << 32;

    uint64 lastGroup = ( bucket | yBuffer[start] ) / kBC;

    for( uint32 i = start+1; i < end; i++ )
    {
        const uint64 group = ( bucket | yBuffer[i] ) / kBC;

        if( group != lastGroup )
        {
            ASSERT( group > lastGroup );

            groupBoundaries[groupCount++] = i;
            lastGroup = group;

            if( groupCount == maxGroups )
            {
                ASSERT( 0 );    // We ought to always have enough space
                                // So this should be an error
                break;
            }
        }
    }

    this->groupCount = groupCount;
}

//-----------------------------------------------------------
uint32 MatchEntries( const uint32* yBuffer, const uint32* groupBoundaries, Pairs pairs,
                     const uint32  groupCount, const uint32 startIndex, const uint32 maxPairs,
                     const uint64  bucketL, const uint64 bucketR )
{
    uint32 pairCount = 0;

    uint8  rMapCounts [kBC];
    uint16 rMapIndices[kBC];

    uint64 groupLStart = startIndex;
    uint64 groupL      = ( bucketL | yBuffer[groupLStart] ) / kBC;

    #if _DEBUG
        uint32 groupPairs = 0;
    #endif

    for( uint32 i = 0; i < groupCount; i++ )
    {
        const uint64 groupRStart = groupBoundaries[i];
        const uint64 groupR      = ( bucketR | yBuffer[groupRStart] ) / kBC;

        if( groupR - groupL == 1 )
        {
            // Groups are adjacent, calculate matches
            const uint16 parity           = groupL & 1;
            const uint64 groupREnd        = groupBoundaries[i+1];

            const uint64 groupLRangeStart = groupL * kBC;
            const uint64 groupRRangeStart = groupR * kBC;
            
            ASSERT( groupREnd - groupRStart <= 350 );
            ASSERT( groupLRangeStart == groupRRangeStart - kBC );

            // Prepare a map of range kBC to store which indices from groupR are used
            // For now just iterate our bGroup to find the pairs
           
            // #NOTE: memset(0) works faster on average than keeping a separate a clearing buffer
            memset( rMapCounts, 0, sizeof( rMapCounts ) );
            
            for( uint64 iR = groupRStart; iR < groupREnd; iR++ )
            {
                uint64 localRY = ( bucketR | yBuffer[iR] ) - groupRRangeStart;
                ASSERT( ( bucketR | yBuffer[iR] ) / kBC == groupR );

                if( rMapCounts[localRY] == 0 )
                    rMapIndices[localRY] = (uint16)( iR - groupRStart );

                rMapCounts[localRY] ++;
            }
            
            // For each group L entry
            for( uint64 iL = groupLStart; iL < groupRStart; iL++ )
            {
                const uint64 yL     = bucketL | yBuffer[iL];
                const uint64 localL = yL - groupLRangeStart;

                // Iterate kExtraBitsPow = 1 << kExtraBits = 1 << 6 == 64
                // So iterate 64 times for each L entry.
                for( int iK = 0; iK < kExtraBitsPow; iK++ )
                {
                    const uint64 targetR = L_targets[parity][localL][iK];

                    for( uint j = 0; j < rMapCounts[targetR]; j++ )
                    {
                        const uint64 iR = groupRStart + rMapIndices[targetR] + j;

                        ASSERT( iL < iR );

                        // Add a new pair
                        ASSERT( ( iR - iL ) <= 0xFFFF );

                        pairs.left [pairCount] = (uint32)iL;
                        pairs.right[pairCount] = (uint16)(iR - iL);
                        pairCount++;

                        #if _DEBUG
                            groupPairs++;
                        #endif

                        ASSERT( pairCount <= maxPairs );
                        if( pairCount == maxPairs )
                            return pairCount;
                    }
                }
            }
        }
        // Else: Not an adjacent group, skip to next one.

        // Go to next group
        groupL      = groupR;
        groupLStart = groupRStart;

        #if _DEBUG
            groupPairs = 0;
        #endif
    }

    return pairCount;
}

//-----------------------------------------------------------
void MatchJob::Run()
{
    const uint32* yBuffer         = this->yBuffer;
    const uint32* groupBoundaries = this->groupInfo->groupBoundaries;
    const uint32  groupCount      = this->groupInfo->groupCount;
    const uint32  maxPairs        = this->maxPairCount;
    const uint64  bucket          = ((uint64)this->bucketIdx) << 32;

    Pairs  pairs     = groupInfo->pairs;
    uint32 pairCount = 0;

    uint8  rMapCounts [kBC];
    uint16 rMapIndices[kBC];

    uint64 groupLStart = this->groupInfo->startIndex;
    uint64 groupL      = ( bucket | yBuffer[groupLStart] ) / kBC;

    #if _DEBUG
        uint32 groupPairs = 0;
    #endif

    for( uint32 i = 0; i < groupCount; i++ )
    {
        const uint64 groupRStart = groupBoundaries[i];
        const uint64 groupR      = ( bucket | yBuffer[groupRStart] ) / kBC;

        if( groupR - groupL == 1 )
        {
            // Groups are adjacent, calculate matches
            const uint16 parity           = groupL & 1;
            const uint64 groupREnd        = groupBoundaries[i+1];

            const uint64 groupLRangeStart = groupL * kBC;
            const uint64 groupRRangeStart = groupR * kBC;

            ASSERT( groupREnd - groupRStart <= 350 );
            ASSERT( groupLRangeStart == groupRRangeStart - kBC );

            // Prepare a map of range kBC to store which indices from groupR are used
            // For now just iterate our bGroup to find the pairs
           
            // #NOTE: memset(0) works faster on average than keeping a separate a clearing buffer
            memset( rMapCounts, 0, sizeof( rMapCounts ) );

            for( uint64 iR = groupRStart; iR < groupREnd; iR++ )
            {
                uint64 localRY = ( bucket | yBuffer[iR] ) - groupRRangeStart;
                ASSERT( ( bucket | yBuffer[iR] ) / kBC == groupR );

                if( rMapCounts[localRY] == 0 )
                    rMapIndices[localRY] = (uint16)( iR - groupRStart );

                rMapCounts[localRY] ++;
            }

            // For each group L entry
            for( uint64 iL = groupLStart; iL < groupRStart; iL++ )
            {
                const uint64 yL     = bucket | yBuffer[iL];
                const uint64 localL = yL - groupLRangeStart;

                // Iterate kExtraBitsPow = 1 << kExtraBits = 1 << 6 == 64
                // So iterate 64 times for each L entry.
                for( int iK = 0; iK < kExtraBitsPow; iK++ )
                {
                    const uint64 targetR = L_targets[parity][localL][iK];

                    for( uint j = 0; j < rMapCounts[targetR]; j++ )
                    {
                        const uint64 iR = groupRStart + rMapIndices[targetR] + j;

                        ASSERT( iL < iR );

                        // Add a new pair
                        ASSERT( ( iR - iL ) <= 0xFFFF );

                        pairs.left [pairCount] = (uint32)iL;
                        pairs.right[pairCount] = (uint16)(iR - iL);
                        pairCount++;

                        #if _DEBUG
                            groupPairs++;
                        #endif
                        // #TODO: Write to disk if there's a buffer available and we have enough entries to write

                        ASSERT( pairCount <= maxPairs );
                        if( pairCount == maxPairs )
                            goto RETURN;
                    }
                }
            }
        }
        // Else: Not an adjacent group, skip to next one.

        // Go to next group
        groupL      = groupR;
        groupLStart = groupRStart;

        #if _DEBUG
            groupPairs = 0;
        #endif
    }

RETURN:
    this->pairCount             = pairCount;
    this->groupInfo->entryCount = pairCount;
}





///
/// Fx Generation
///
//-----------------------------------------------------------
template<TableId tableId>
void DiskPlotPhase1::GenFxForTable( uint bucketIdx, uint entryCount, const Pairs pairs,
                                    const uint32* yIn, uint32* yOut, byte* bucketIdOut,
                                    const uint64* metaInA, const uint64* metaInB,
                                    uint64* metaOutA, uint64* metaOutB )
{
    Log::Line( "  Computing Fx..." );
    auto timer = TimerBegin();

    auto& cx = _cx;

    const size_t outMetaSizeA     = TableMetaOut<tableId>::SizeA;
    const size_t outMetaSizeB     = TableMetaOut<tableId>::SizeB;

    const size_t fileBlockSize    = _diskQueue->BlockSize();
    const size_t sizePerEntry     = sizeof( uint32 ) + outMetaSizeA + outMetaSizeB;

    const size_t writeInterval    = cx.writeIntervals[(uint)tableId].fxGen;

    const size_t entriesTotalSize = entryCount * sizePerEntry;
    ASSERT( writeInterval <= entriesTotalSize );

    uint32 entriesPerChunk        = (uint32)( writeInterval / sizePerEntry );
    uint32 sizePerChunk           = (uint32)( entriesPerChunk * sizePerEntry ); ASSERT( sizePerChunk <= writeInterval );
    uint32 chunkCount             = (uint32)( entriesTotalSize / sizePerChunk );

    const uint32 threadCount      = cx.threadCount;
    const uint32 entriesPerThread = entriesPerChunk / threadCount;

    entriesPerChunk = entriesPerThread * threadCount;
    uint32       trailingEntries  = entryCount - ( entriesPerChunk * chunkCount );

    while( trailingEntries >= entriesPerChunk )
    {
        chunkCount++;
        trailingEntries -= entriesPerChunk;
    }

    // Add trailing entries as a trailing chunk
    const uint32 lastChunkEntries = trailingEntries / threadCount;

    // Remove that from the trailing entries.
    // This guarantees that any trailing entries will be <= threadCount
    trailingEntries -= lastChunkEntries * threadCount;

    ASSERT( entriesPerThread * threadCount * chunkCount + lastChunkEntries * threadCount + trailingEntries == entryCount );

    uint32* bucketCounts = _cx.bucketCounts[(uint)tableId];
//     memset( bucketCounts, 0, sizeof( uint32 ) * BB_DP_BUCKET_COUNT );    // Already zeroed

    MTJobRunner<FxJob> jobs( *cx.threadPool );

    for( uint i = 0; i < threadCount; i++ )
    {
        FxJob& job = jobs[i];

        job.totalBucketCounts    = nullptr;        
        job.diskQueue            = _diskQueue;
        job.tableId              = tableId;
        job.bucketIdx            = bucketIdx;
        job.entryCount           = entriesPerThread;
        job.chunkCount           = chunkCount;
        job.entriesPerChunk      = entriesPerChunk;
        job.trailingChunkEntries = lastChunkEntries;

        const size_t offset = entriesPerThread * i;
        job.pairs       = pairs;
        job.pairs.left  += offset;
        job.pairs.right += offset;

        job.yIn            = yIn;
        job.metaInA        = metaInA;
        job.metaInB        = metaInB;
        job.yOut           = yOut        + offset;
        job.metaOutA       = metaOutA    + offset;
        job.metaOutB       = metaOutB    + offset;
        job.bucketIdOut    = bucketIdOut + offset;

        job.yOverflows     = &_bucket->yOverflow;
        job.metaAOverflows = &_bucket->metaAOverflow;
        job.metaBOverflows = &_bucket->metaBOverflow;
    }

    jobs[0].totalBucketCounts = bucketCounts;

    // #TODO We need to grab the trailing entries...
    //       This isn't working for some reason. Not sure why yet.
    jobs[threadCount - 1].trailingChunkEntries += trailingEntries;

    // Zero-out overflow buffers
    memset( _bucket->yOverflow.sizes    , 0, sizeof( uint32 ) * BB_DP_BUCKET_COUNT );
    memset( _bucket->metaAOverflow.sizes, 0, sizeof( uint32 ) * BB_DP_BUCKET_COUNT );
    memset( _bucket->metaBOverflow.sizes, 0, sizeof( uint32 ) * BB_DP_BUCKET_COUNT );

    // Calculate Fx
    jobs.Run();

    // #TODO: Calculate trailing entries here (they are less than the thread count)
    //        if we have any.
    if( trailingEntries )
    {
        // Call ComputeFxForTable
    }

    auto elapsed = TimerEnd( timer );
    Log::Line( "  Finished computing Fx in %.4lf seconds.", elapsed );
}

//-----------------------------------------------------------
void FxJob::Run()
{
    // ASSERT( this->entriesPerChunk == this->entryCount * this->_jobCount );
    // switch( tableId )
    // {
    //     case TableId::Table1: RunForTable<TableId::Table1>(); return;
    //     case TableId::Table2: RunForTable<TableId::Table2>(); return;
    //     case TableId::Table3: RunForTable<TableId::Table3>(); return;
    //     case TableId::Table4: RunForTable<TableId::Table4>(); return;
    //     case TableId::Table5: RunForTable<TableId::Table5>(); return;
    //     case TableId::Table6: RunForTable<TableId::Table6>(); return;
    //     case TableId::Table7: RunForTable<TableId::Table7>(); return;
        
    //     default:
    //         ASSERT( 0 );
    //     break;
    // }
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"

//-----------------------------------------------------------
template<typename T>
FORCE_INLINE
void FxJob::SaveBlockRemainders( FileId fileId, const uint32* bucketCounts, const T* buffer, 
                                 uint32* remainderSizes, DoubleBuffer* remainderBuffers )
{
    DiskBufferQueue& queue               = *this->diskQueue;
    const size_t     fileBlockSize       = queue.BlockSize();
    const size_t     remainderBufferSize = fileBlockSize * BB_DP_BUCKET_COUNT;

    ASSERT( fileBlockSize > sizeof( T) );
    
    byte* ptr = (byte*)buffer;

    for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
    {
        const size_t bucketSize       = bucketCounts[i] * sizeof( T );
        const size_t blockAlignedSize = bucketSize / fileBlockSize * fileBlockSize;
        
        size_t remainderSize = bucketSize - blockAlignedSize;
        ASSERT( remainderSize / sizeof( T ) * sizeof( T ) == remainderSize );

        if( remainderSize )
        {
            size_t curRemainderSize = remainderSizes[i];
                        
            const size_t copySize = std::min( remainderSize, fileBlockSize - curRemainderSize );

            DoubleBuffer& buf       = remainderBuffers[i];
            byte*         remainder = buf.front;

            bbmemcpy_t( remainder + curRemainderSize, ptr + blockAlignedSize, copySize );

            curRemainderSize += remainderSize;

            if( curRemainderSize >= fileBlockSize )
            {
                // This may block if the last buffer has not yet finished writing to disk
                buf.Flip();

                // Overflow buffer is full, submit it for writing
                queue.WriteFile( fileId, i, remainder, fileBlockSize );
                queue.SignalFence( buf.fence );
                queue.CommitCommands();

                // Update new remainder size, if we overflowed our buffer
                // and copy any overflow, if we have some.
                remainderSize = curRemainderSize - fileBlockSize;

                if( remainderSize )
                {
                    remainder = buf.front;
                    bbmemcpy_t( remainder, ptr + blockAlignedSize + copySize, remainderSize );
                }

                remainderSizes[i] = 0;
                remainderSize     = remainderSize;
            }

            // Update size
            remainderSizes[i] += (uint)remainderSize;
        }

        // Go to the next input bucket
        ptr += bucketSize;
    }
}

#pragma GCC diagnostic pop

//-----------------------------------------------------------
void OverflowBuffer::Init( void* bucketBuffers, const size_t fileBlockSize )
{
    ASSERT( bucketBuffers );
    ASSERT( fileBlockSize );

    const size_t fileBlockSizeX2 = fileBlockSize * 2;

    byte* ptr = (byte*)bucketBuffers;

    for( int i = 0; i < (int)BB_DP_BUCKET_COUNT; i++ )
    {
        buffers[i].front = ptr;
        buffers[i].back  = ptr + fileBlockSize;
        ptr += fileBlockSizeX2;
    }

    memset( sizes, 0, sizeof( sizes ) );
}

