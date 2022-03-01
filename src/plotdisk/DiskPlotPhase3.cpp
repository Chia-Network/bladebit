#include "DiskPlotPhase3.h"
#include "util/BitField.h"
#include "algorithm/RadixSort.h"
#include "DiskPlotDebug.h"
#include "plotmem/LPGen.h"
#include "plotmem/ParkWriter.h"
#include "jobs/UnpackMapJob.h"
#include "plotting/TableWriter.h"

#define P3_EXTRA_L_ENTRIES_TO_LOAD 1024     // Extra L entries to load per bucket to ensure we
                                            // have cross bucket entries accounted for


/**
 * Algorithm:
 * 
 * Let rTable be a table in a set {table2, table3, ..., table7}
 * Let lTable be rTable - 1. Such that if rTable is table2, then lTable is table1
 * 
 * For each rTable perform 2 passes:
 *
 * Pass 1. Process each bucket as follows]:
 * - Load L/R back pointers for rTable.
 * - Load y index map for rTable.
 * - Load marked entries from Phase 2 for rTable.
 * - Load lTable, which for rTable==1 is the x buckets, otherwise it is the output of map of 
 *      the previous iteration's rTable.
 * - If rTable > table2:
 *      - Sort the lTable map on its origin (y) index, and then discard the origin index,
 *          keeping only the destination index (final position of an entry after LP sort).
 * - Sort the rTable map on its origin index.
 * - Generate LinePoints (LPs) from the rTable pointers and the lTable x or map values,
 *      while excluding each entry that is not marked in the marked entries table.
 * - Distribute the LPs to their respective buckets along with the rTable (y) map 
 *      and write them to disk.
 *      (The r table (y) map represents the origin index before sorting.)
 * 
 * Pass 2. Process each LP bucket as follows:
 * - Load the rTable LP output and map.
 * - Sort the LP bucket and map on LP.
 * - Compress the LP bucket and write it to disk.
 * - Convert the sorted map into a reverse lookup by extending them with its origin index (its current value)
 *      and its destination index (its current index after sort). Then distribute
 *      them to buckets given its origin value. Write the buckets to disk.
 * 
 * Go to next table.
 */
struct P3FenceId
{
    enum 
    {
        Start = 0,

        RTableLoaded,
        RMapLoaded,

        FENCE_COUNT
    };
};

struct Step2FenceId
{
    enum 
    {
        Start = 0,

        LPLoaded,
        MapLoaded,

        FENCE_COUNT
    };
};


struct ConvertToLPJob : public PrefixSumJob<ConvertToLPJob>
{
    DiskPlotContext* context;
    TableId          rTable;

    uint64           rTableOffset;
    uint32           bucketEntryCount;
    const uint64*    markedEntries;
    const uint32*    lMap;
    const uint32*    rMap;
    Pairs            rTablePairs;

    uint64*          linePoints;        // Buffer for line points/pruned pairs
    uint32*          rMapPruned;        // Where we store our pruned R map

    int64            prunedEntryCount;      // Pruned entry count per thread.
    int64            totalPrunedEntryCount; // Pruned entry count accross all threads
    
    void Run() override;

    template<TableId rTable>
    void RunForTable();

    // For distributing
    uint32*          bucketCounts;          // Total count of entries per bucket (used by first thread)
    uint64*          lpOutBuffer;
    uint32*          keyOutBuffer;

    void DistributeToBuckets( const int64 enytryCount, const uint64* linePoints, const uint32* map );
};

/*
//-----------------------------------------------------------
DiskPlotPhase3::DiskPlotPhase3( DiskPlotContext& context, const Phase3Data& phase3Data )
    : _context   ( context    )
    , _phase3Data( phase3Data )
{
    memset( _tablePrunedEntryCount, 0, sizeof( _tablePrunedEntryCount ) );

    DiskBufferQueue& ioQueue = *context.ioQueue;

    // Open required files
    ioQueue.InitFileSet( FileId::LP_2, "lp_2", BB_DPP3_LP_BUCKET_COUNT );
    ioQueue.InitFileSet( FileId::LP_3, "lp_3", BB_DPP3_LP_BUCKET_COUNT );
    ioQueue.InitFileSet( FileId::LP_4, "lp_4", BB_DPP3_LP_BUCKET_COUNT );
    ioQueue.InitFileSet( FileId::LP_5, "lp_5", BB_DPP3_LP_BUCKET_COUNT );
    ioQueue.InitFileSet( FileId::LP_6, "lp_6", BB_DPP3_LP_BUCKET_COUNT );
    ioQueue.InitFileSet( FileId::LP_7, "lp_7", BB_DPP3_LP_BUCKET_COUNT );

    ioQueue.InitFileSet( FileId::LP_KEY_2, "lp_key_2", BB_DPP3_LP_BUCKET_COUNT );
    ioQueue.InitFileSet( FileId::LP_KEY_3, "lp_key_3", BB_DPP3_LP_BUCKET_COUNT );
    ioQueue.InitFileSet( FileId::LP_KEY_4, "lp_key_4", BB_DPP3_LP_BUCKET_COUNT );
    ioQueue.InitFileSet( FileId::LP_KEY_5, "lp_key_5", BB_DPP3_LP_BUCKET_COUNT );
    ioQueue.InitFileSet( FileId::LP_KEY_6, "lp_key_6", BB_DPP3_LP_BUCKET_COUNT );
    ioQueue.InitFileSet( FileId::LP_KEY_7, "lp_key_7", BB_DPP3_LP_BUCKET_COUNT );

    ioQueue.InitFileSet( FileId::LP_MAP_2, "lp_map_2", BB_DP_BUCKET_COUNT );
    ioQueue.InitFileSet( FileId::LP_MAP_3, "lp_map_3", BB_DP_BUCKET_COUNT );
    ioQueue.InitFileSet( FileId::LP_MAP_4, "lp_map_4", BB_DP_BUCKET_COUNT );
    ioQueue.InitFileSet( FileId::LP_MAP_5, "lp_map_5", BB_DP_BUCKET_COUNT );
    ioQueue.InitFileSet( FileId::LP_MAP_6, "lp_map_6", BB_DP_BUCKET_COUNT );
    ioQueue.InitFileSet( FileId::LP_MAP_7, "lp_map_7", BB_DP_BUCKET_COUNT );

    // Find largest bucket size accross all tables
    uint32 maxBucketLength = 0;
    for( TableId table = TableId::Table1; table <= TableId::Table7; table = table +1 )
    {
        if( table < TableId::Table2 )
        {
            for( uint32 i = 0; i < BB_DP_BUCKET_COUNT; i++ )
                maxBucketLength = std::max( context.bucketCounts[(int)table][i], maxBucketLength );
        }
        else
        {
            for( uint32 i = 0; i < BB_DP_BUCKET_COUNT; i++ )
            {
                maxBucketLength = std::max( context.bucketCounts[(int)table][i], 
                                    std::max( context.ptrTableBucketCounts[(int)table][i], maxBucketLength ) );
            }
        }
    }

    maxBucketLength += P3_EXTRA_L_ENTRIES_TO_LOAD;

    // Init our buffers
    // #TODO: Remove this as we're moving alignment on to the ioQueue to handle?
    const size_t fileBlockSize        = ioQueue.BlockSize();

    // #TODO: Only have marking table, lp bucket and pruned r map buckets as
    //        fixed buffers, the rest we can just grab from the heap.
    const size_t markedEntriesSize    = phase3Data.bitFieldSize;
    const size_t rTableMapBucketSize  = RoundUpToNextBoundary( maxBucketLength * sizeof( uint32 ), fileBlockSize );
    const size_t rTableLPtrBucketSize = RoundUpToNextBoundary( maxBucketLength * sizeof( uint32 ), fileBlockSize );
    const size_t rTableRPtrBucketSize = RoundUpToNextBoundary( maxBucketLength * sizeof( uint16 ), fileBlockSize );
    
    const size_t lTableBucketSize     = RoundUpToNextBoundary( maxBucketLength * sizeof( uint32 ), fileBlockSize );
    const size_t lpBucketSize         = RoundUpToNextBoundary( maxBucketLength * sizeof( uint64 ), fileBlockSize );

    byte* heap = context.heapBuffer;

    _markedEntries        = (uint64*)heap;
    heap += markedEntriesSize;

    _rMap[0]              = (uint32*)heap; heap += rTableMapBucketSize;
    _rMap[1]              = (uint32*)heap; heap += rTableMapBucketSize;

    _rTablePairs[0].left  = (uint32*)heap; heap += rTableLPtrBucketSize;
    _rTablePairs[1].left  = (uint32*)heap; heap += rTableLPtrBucketSize;

    _rTablePairs[0].right = (uint16*)heap; heap += rTableRPtrBucketSize;
    _rTablePairs[1].right = (uint16*)heap; heap += rTableRPtrBucketSize;

    _lMap[0]    = (uint32*)heap; heap += lTableBucketSize;
    _lMap[1]    = (uint32*)heap; heap += lTableBucketSize;

    _rPrunedMap = (uint32*)heap; heap += rTableMapBucketSize;
    _linePoints = (uint64*)heap; heap += lpBucketSize;

    size_t totalSize = 
        markedEntriesSize        + 
        rTableMapBucketSize  * 2 + 
        rTableLPtrBucketSize * 2 + 
        rTableRPtrBucketSize * 2 + 
        lTableBucketSize     * 2 +
        rTableMapBucketSize      +
        lpBucketSize;

    ASSERT( (size_t)(heap - context.heapBuffer) == totalSize );

    // Reset our heap to the remainder of what we're not using
    const size_t fullHeapSize  = context.heapSize + context.ioHeapSize;
    const size_t heapRemainder = fullHeapSize - totalSize;
    ASSERT( context.heapBuffer + fullHeapSize == heap + heapRemainder );

    ioQueue.ResetHeap( heapRemainder, heap );
}

//-----------------------------------------------------------
DiskPlotPhase3::~DiskPlotPhase3()
{}

//-----------------------------------------------------------
void DiskPlotPhase3::Run()
{
    TableId startTable = TableId::Table2;
    #ifdef BB_DP_DBG_P3_START_TABLE
        startTable = TableId::BB_DP_DBG_P3_START_TABLE;
    #endif

    for( TableId rTable = startTable; rTable <= TableId::Table7; rTable++ )
    {
        Log::Line( "Compressing Tables %u and %u...", rTable, rTable+1 );
        const auto timer = TimerBegin();

        ProcessTable( rTable );

        const auto elapsed = TimerEnd( timer );
        Log::Line( "Finished compression in %.2lf seconds.", elapsed );
    }

    Log::Line( "Phase3 Total IO Aggregate Wait Time | READ: %.4lf | WRITE: %.4lf | BUFFERS: %.4lf", 
            TicksToSeconds( _context.readWaitTime ), TicksToSeconds( _context.writeWaitTime ), _context.ioQueue->IOBufferWaitTime() );
}

//-----------------------------------------------------------
void DiskPlotPhase3::ProcessTable( const TableId rTable )
{
    DiskPlotContext& context = _context;
    // DiskBufferQueue& ioQueue = *context.ioQueue;

    // Reset table counts 
    _prunedEntryCount = 0;
    memset( _lpBucketCounts  , 0, sizeof( _lpBucketCounts ) );
    memset( _lMapBucketCounts, 0, sizeof( _lMapBucketCounts ) );

    // Reset Fence
    _readFence.Reset( P3FenceId::Start );

    #if _DEBUG && BB_DBG_SKIP_P3_S1

    // Skip first step
    // if( 0 )
    #endif
    // Prune the R table pairs and key,
    // convert pairs to LPs, then distribute
    // the LPs to buckets, along with the key.
    TableFirstStep( rTable );
    Log::Line( "  Step 1 IO Aggregate Wait Time | READ: %.4lf | WRITE: %.4lf | BUFFERS: %.4lf", 
            TicksToSeconds( context.readWaitTime ), TicksToSeconds( context.writeWaitTime ), context.ioQueue->IOBufferWaitTime() );

    // Validate linePoints
    // #if _DEBUG
    //     if( rTable > TableId::Table2 )
    //         Debug::ValidateLinePoints( context, rTable, _lpBucketCounts );
    // #endif

    // Load LP buckets and key, sort them, 
    // write a reverse lookup map given the sorted key,
    // then compress and write the rTable to disk.
    TableSecondStep( rTable );
    Log::Line( "  Step 2 IO Aggregate Wait Time | READ: %.4lf | WRITE: %.4lf | BUFFERS: %.4lf", 
            TicksToSeconds( context.readWaitTime ), TicksToSeconds( context.writeWaitTime ), context.ioQueue->IOBufferWaitTime() );

    // Unpack map to be used as the L table for the next table iteration
    TableThirdStep( rTable );

    // Update to our new bucket count and table entry count
    const uint64 oldEntryCount = context.entryCounts[(int)rTable];
    Log::Line( " Table %u now has %llu / %llu ( %.2lf%% ) entries.", rTable,
                _prunedEntryCount, oldEntryCount, (double)_prunedEntryCount / oldEntryCount * 100 );

    Log::Line( " Table %u IO Aggregate Wait Time | READ: %.4lf | WRITE: %.4lf | BUFFERS: %.4lf", rTable,
            TicksToSeconds( context.readWaitTime ), TicksToSeconds( context.writeWaitTime ), context.ioQueue->IOBufferWaitTime() );

    // context.entryCounts[(int)rTable] = _prunedEntryCount;
    _tablePrunedEntryCount[(int)rTable] = _prunedEntryCount;
}


///
/// First Step
///
//-----------------------------------------------------------
void DiskPlotPhase3::TableFirstStep( const TableId rTable )
{
    Log::Line( "  Step 1" );

    DiskPlotContext& context         = _context;
    DiskBufferQueue& ioQueue         = *context.ioQueue;
    Fence&           readFence       = _readFence;

    const TableId lTable             = rTable - 1;
    const uint64  maxEntries         = 1ull << _K;
    const uint64  lTableEntryCount   = context.entryCounts[(int)lTable];
    const uint64  rTableEntryCount   = context.entryCounts[(int)rTable];

    const FileId markedEntriesFileId = rTable < TableId::Table7 ? TableIdToMarkedEntriesFileId( rTable ) : FileId::None;
    const FileId lMapId              = rTable == TableId::Table2 ? FileId::X : TableIdToLinePointMapFileId( lTable );
    const FileId rMapId              = TableIdToMapFileId( rTable );
    const FileId rPtrsRId            = TableIdToBackPointerFileId( rTable ); 
    const FileId rPtrsLId            = rPtrsRId + 1;

    // Prepare our files for reading
    if( rTable < TableId::Table7 )
        ioQueue.SeekBucket( markedEntriesFileId, 0, SeekOrigin::Begin );

    ioQueue.SeekFile  ( lMapId             , 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile  ( rMapId             , 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile  ( rPtrsRId           , 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile  ( rPtrsLId           , 0, 0, SeekOrigin::Begin );
    ioQueue.CommitCommands();

    uint64 lEntriesLoaded = 0;

    // Read first bucket
    {
        const uint32 lBucketLength = context.bucketCounts[(int)lTable][0] + P3_EXTRA_L_ENTRIES_TO_LOAD;
        const uint32 rBucketLength = context.ptrTableBucketCounts[(int)rTable][0];

        lEntriesLoaded += lBucketLength;

        // Read L Table 1st bucket
        ioQueue.ReadFile( lMapId, 0, _lMap[0], lBucketLength * sizeof( uint32 ) );

        // Read R Table marks
        if( rTable < TableId::Table7 )
        {
            ioQueue.ReadFile( markedEntriesFileId, 0, _markedEntries, _phase3Data.bitFieldSize );
            DeleteFile( markedEntriesFileId, 0 );
        }

        // Read R Table 1st bucket
        ioQueue.ReadFile( rPtrsRId, 0, _rTablePairs[0].left , rBucketLength * sizeof( uint32 ) );
        ioQueue.ReadFile( rPtrsLId, 0, _rTablePairs[0].right, rBucketLength * sizeof( uint16 ) );

        ioQueue.ReadFile( rMapId, 0, _rMap[0], rBucketLength * sizeof( int32 ) );
        ioQueue.SignalFence( readFence, 1 );

        ioQueue.CommitCommands();
    }
    
    // Reset offsets
    _rTableOffset = 0;

    // Start processing buckets
    for( uint bucket = 0; bucket < BB_DP_BUCKET_COUNT; bucket++ )
    {
        const bool isCurrentBucketLastBucket = bucket == BB_DP_BUCKET_COUNT - 1;
        
        if( !isCurrentBucketLastBucket )
        {
            // Load the next bucket on the background
            const uint32 nextBucket             = bucket + 1;
            const bool   nextBucketIsLastBucket = nextBucket == BB_DP_BUCKET_COUNT - 1; 
           
            uint32 lBucketLength = context.bucketCounts[(int)lTable][nextBucket];
            uint32 rBucketLength = context.ptrTableBucketCounts[(int)rTable][nextBucket];

            // ASSERT( !nextBucketIsLastBucket || lEntriesLoaded+lBucketLength == context.entryCounts[(int)lTable] );
            if( nextBucketIsLastBucket )
                lBucketLength = (uint32)( context.entryCounts[(int)lTable] - lEntriesLoaded );

            lEntriesLoaded += lBucketLength;

            // Load L Table
            ioQueue.ReadFile( lMapId, 0, _lMap[1] + P3_EXTRA_L_ENTRIES_TO_LOAD, lBucketLength * sizeof( uint32 ) );

            // Load R Table
            ioQueue.ReadFile( rPtrsRId, 0, _rTablePairs[1].left , rBucketLength * sizeof( uint32 ) );
            ioQueue.ReadFile( rPtrsLId, 0, _rTablePairs[1].right, rBucketLength * sizeof( uint16 ) );
            
            ioQueue.ReadFile( rMapId, 0, _rMap[1], rBucketLength * sizeof( uint32 ) );
            ioQueue.SignalFence( readFence, nextBucket + 1 );

            ioQueue.CommitCommands();
        }

        // Process the bucket
        BucketFirstStep( rTable, bucket );

        // Copy last L entries from current bucket to next bucket's first entries
        memcpy( _lMap[1], _lMap[0] + context.bucketCounts[(int)lTable][bucket], P3_EXTRA_L_ENTRIES_TO_LOAD * sizeof( uint32 ) );

        // Swap buffers
        std::swap( _lMap[0]       , _lMap[1] );
        std::swap( _rMap[0]       , _rMap[1] );
        std::swap( _rTablePairs[0], _rTablePairs[1] );
    }

    DeleteFile( lMapId  , 0 );
    DeleteFile( rPtrsRId, 0 );
    DeleteFile( rPtrsLId, 0 );
    DeleteFile( rMapId  , 0 );
}

//-----------------------------------------------------------
void DiskPlotPhase3::BucketFirstStep( const TableId rTable, const uint32 bucket )
{
    DiskPlotContext& context        = _context;
    DiskBufferQueue& ioQueue        = *context.ioQueue;
    ThreadPool&      threadPool     = *context.threadPool;
    Fence&           readFence      = _readFence;

    const TableId lTable            = rTable - 1;
    const bool    isLastBucket      = bucket == BB_DP_BUCKET_COUNT - 1;
    const uint32  bucketEntryCountR = context.ptrTableBucketCounts[(int)rTable][bucket];

    // Wait for the bucket to be loaded
    readFence.Wait( bucket + 1, context.readWaitTime );

    #if _DEBUG
    {
        const uint32 r = _rTablePairs[0].left[bucketEntryCountR-1] + _rTablePairs[0].right[bucketEntryCountR-1];
        const uint32 lTableBucketLength = _context.bucketCounts[(int)lTable][bucket] + P3_EXTRA_L_ENTRIES_TO_LOAD;
        ASSERT( r < lTableBucketLength );
    }
    #endif

    // Convert to line points
    const uint32 prunedEntryCount = 
        PointersToLinePoints( 
            rTable, _rTableOffset,
            bucketEntryCountR, _markedEntries, 
            _lMap[0], 
            _rTablePairs[0], _rMap[0], 
            _rPrunedMap, _linePoints );

    _prunedEntryCount += prunedEntryCount;

    // Update our offset for the next bucket
    _rTableOffset += bucketEntryCountR;
}

//-----------------------------------------------------------
uint32 DiskPlotPhase3::PointersToLinePoints( 
    TableId rTable, uint64 entryOffset,
    const uint32 entryCount, const uint64* markedEntries, 
    const uint32* lTable, 
    const Pairs pairs, const uint32* rMapIn, 
    uint32* rMapOut, uint64* outLinePoints )
{
    const uint32 threadCount = _context.p3ThreadCount;

    uint32 bucketCounts[BB_DPP3_LP_BUCKET_COUNT];

    MTJobRunner<ConvertToLPJob> jobs( *_context.threadPool );

    for( uint32 i = 0; i < threadCount; i++ )
    {
        ConvertToLPJob& job = jobs[i];

        job.context = &_context;
        job.rTable  = rTable;

        job.rTableOffset     = entryOffset;
        job.bucketEntryCount = entryCount;
        job.markedEntries    = markedEntries;
        job.lMap             = lTable;
        job.rTablePairs      = pairs;
        job.rMap             = rMapIn;
        job.linePoints       = outLinePoints;
        job.rMapPruned       = rMapOut;

        job.bucketCounts     = bucketCounts;
    }

    jobs.Run( threadCount );

    for( uint32 i = 0; i < BB_DPP3_LP_BUCKET_COUNT; i++ )
        _lpBucketCounts[i] += bucketCounts[i];

    uint32 prunedEntryCount = (uint32)jobs[0].totalPrunedEntryCount;
    return prunedEntryCount;
}

//-----------------------------------------------------------
void ConvertToLPJob::Run()
{
    switch( this->rTable )
    {
        case TableId::Table2: this->RunForTable<TableId::Table2>(); break;
        case TableId::Table3: this->RunForTable<TableId::Table3>(); break;
        case TableId::Table4: this->RunForTable<TableId::Table4>(); break;
        case TableId::Table5: this->RunForTable<TableId::Table5>(); break;
        case TableId::Table6: this->RunForTable<TableId::Table6>(); break;
        case TableId::Table7: this->RunForTable<TableId::Table7>(); break;
    
    default:
        ASSERT( 0 );
        break;
    }
}

//-----------------------------------------------------------
template<TableId rTable>
void ConvertToLPJob::RunForTable()
{
    DiskPlotContext& context = *this->context;

    const int32 threadCount  = (int32)this->JobCount();
    int64       entryCount   = (int64)( this->bucketEntryCount / threadCount );

    const int64 bucketOffset = entryCount * (int32)this->JobId();   // Offset in the bucket
    const int64 rTableOffset = (int64)this->rTableOffset;           // Offset in the table overall (used for checking marks)
    const int64 marksOffset  = rTableOffset + bucketOffset;

    if( this->IsLastThread() )
        entryCount += (int64)this->bucketEntryCount - entryCount * threadCount;

    const int64 end = bucketOffset + entryCount;

    const BitField markedEntries( (uint64*)this->markedEntries );
    
    const uint32* rMap  = this->rMap;
    const Pairs   pairs = this->rTablePairs;

    // First, scan our entries in order to prune them
    int64 prunedLength = 0;

    if constexpr ( rTable < TableId::Table7 )
    {
        for( int64 i = bucketOffset; i < end; i++)
        {
            // #TODO: Try changing Phase 2 to write atomically to see
            //        (if we don't get a huge performance hit),
            //        if we can do reads without the rMap
            if( markedEntries.Get( rMap[i] ) )
                prunedLength ++;
        }
    }
    else
    {
        prunedLength = entryCount;
    }

    this->prunedEntryCount = prunedLength;
    this->SyncThreads();

    // #NOTE: Not necesarry for T7, but let's avoid code duplication for now.
    // Set our destination offset
    int64 dstOffset = 0;

    for( int32 i = 0; i < (int32)this->JobId(); i++ )
        dstOffset += GetJob( i ).prunedEntryCount;

    // Copy pruned entries into new buffer and expend R pointers to absolute address
    // #TODO: check if doing 1 pass per buffer performs better
    struct Pair
    {
        uint32 left;
        uint32 right;
    };

    Pair*   outPairsStart = (Pair*)(this->linePoints + dstOffset);
    Pair*   outPairs      = outPairsStart;
    uint32* outRMap       = this->rMapPruned + dstOffset;
    uint32* mapWriter     = outRMap;

    for( int64 i = bucketOffset; i < end; i++ )
    {
        const uint32 mapIdx = rMap[i];

        if constexpr ( rTable < TableId::Table7 )
        {
            if( !markedEntries.Get( mapIdx ) )
                continue;
        }

        outPairs->left  = pairs.left[i];
        outPairs->right = outPairs->left + pairs.right[i];

        *mapWriter      = mapIdx;

        outPairs++;
        mapWriter++;
    }

    // Now we can convert our pruned pairs to line points
    uint64* outLinePoints = this->linePoints + dstOffset;
    {
        const uint32* lTable = this->lMap;
        
        for( int64 i = 0; i < prunedLength; i++ )
        {
            Pair p = outPairsStart[i];
            const uint64 x = lTable[p.left ];
            const uint64 y = lTable[p.right];

            outLinePoints[i] = SquareToLinePoint( x, y );
            
            // TEST
            #if _DEBUG
            // if( rTable == TableId::Table7 && outLinePoints[i] == 6866525082325270466 ) BBDebugBreak();
            #endif
        }

        // const uint64* lpEnd = outLinePoints + prunedLength;
        // do
        // {
        //     Pair p = *((Pair*)outLinePoints);
        //     const uint64 x = lTable[p.left ];
        //     const uint64 y = lTable[p.right];
            
        //     *outLinePoints = SquareToLinePoint( x, y );

        // } while( ++outLinePoints < lpEnd );
    }

    this->DistributeToBuckets( prunedLength, outLinePoints, outRMap );
}

//-----------------------------------------------------------
void ConvertToLPJob::DistributeToBuckets( const int64 entryCount, const uint64* linePoints, const uint32* key )
{
    uint32 counts[BB_DPP3_LP_BUCKET_COUNT];
    uint32 pfxSum[BB_DPP3_LP_BUCKET_COUNT];

    memset( counts, 0, sizeof( counts ) );

    // Count entries per bucket
    for( const uint64* lp = linePoints, *end = lp + entryCount; lp < end; lp++ )
    {
        const uint64 bucket = (*lp) >> 56; ASSERT( bucket < BB_DPP3_LP_BUCKET_COUNT );
        counts[bucket]++;
    }
    
    this->CalculatePrefixSum( BB_DPP3_LP_BUCKET_COUNT, counts, pfxSum, this->bucketCounts );

    uint64* lpOutBuffer  = nullptr;
    uint32* keyOutBuffer = nullptr;

    // Grab write buffers for distribution
    if( this->IsControlThread() )
    {
        this->LockThreads();

        DiskBufferQueue& ioQueue = *context->ioQueue;

        const int64 threadCount           = (int64)this->JobCount();
        int64       totalEntryCountPruned = entryCount;

        for( int64 i = 1; i < threadCount; i++ )
            totalEntryCountPruned += this->GetJob( (int)i ).prunedEntryCount;

        this->totalPrunedEntryCount = totalEntryCountPruned;

        const size_t sizeLPs = (size_t)totalEntryCountPruned * sizeof( uint64 );
        const size_t sizeKey = (size_t)totalEntryCountPruned * sizeof( uint32 );

        lpOutBuffer  = (uint64*)ioQueue.GetBuffer( sizeLPs, true );
        keyOutBuffer = (uint32*)ioQueue.GetBuffer( sizeKey, true );

        this->lpOutBuffer  = lpOutBuffer;
        this->keyOutBuffer = keyOutBuffer;

        this->ReleaseThreads();
    }
    else
    {
        this->WaitForRelease();
        lpOutBuffer  = GetJob( 0 ).lpOutBuffer ;
        keyOutBuffer = GetJob( 0 ).keyOutBuffer;
    }

    // Distribute entries to their respective buckets
    for( int64 i = 0; i < entryCount; i++ )
    {
        const uint64 lp       = linePoints[i];
        const uint64 bucket   = lp >> 56;           ASSERT( bucket < BB_DPP3_LP_BUCKET_COUNT );
        const uint32 dstIndex = --pfxSum[bucket];

        ASSERT( dstIndex < this->bucketEntryCount );

        lpOutBuffer [dstIndex] = lp;
        keyOutBuffer[dstIndex] = key[i];
    }

    if( this->IsControlThread() )
    {
        DiskBufferQueue& ioQueue = *context->ioQueue;

        uint32* lpSizes  = (uint32*)ioQueue.GetBuffer( BB_DPP3_LP_BUCKET_COUNT * sizeof( uint32 ) );
        uint32* keySizes = (uint32*)ioQueue.GetBuffer( BB_DPP3_LP_BUCKET_COUNT * sizeof( uint32 ) );

        const uint32* bucketCounts = this->bucketCounts;

        for( int64 i = 0; i < (int)BB_DPP3_LP_BUCKET_COUNT; i++ )
            lpSizes[i] = bucketCounts[i] * sizeof( uint64 );

        for( int64 i = 0; i < (int)BB_DPP3_LP_BUCKET_COUNT; i++ )
            keySizes[i] = bucketCounts[i] * sizeof( uint32 );

        const FileId lpFileId   = TableIdToLinePointFileId   ( this->rTable );
        const FileId lpKeyFilId = TableIdToLinePointKeyFileId( this->rTable );

        // Wait for all threads to finish writing
        this->LockThreads();

        ioQueue.WriteBuckets( lpFileId, lpOutBuffer, lpSizes );
        ioQueue.ReleaseBuffer( lpOutBuffer );
        ioQueue.ReleaseBuffer( lpSizes );

        ioQueue.WriteBuckets( lpKeyFilId, keyOutBuffer, keySizes );
        ioQueue.ReleaseBuffer( keyOutBuffer );
        ioQueue.ReleaseBuffer( keySizes );

        ioQueue.CommitCommands();

        this->ReleaseThreads();
    }
    else
        this->WaitForRelease();

    // #NOTE: If we move the write from here, we still need to sync the 
    //        threads before existing to ensure the counts[] buffer
    //        doesn't go out of scope.
}



///
/// Seconds Step
///
//-----------------------------------------------------------
void DiskPlotPhase3::TableSecondStep( const TableId rTable )
{
    Log::Line( "  Step 2" );

    // #TODO: Organize buckets for a single bucket step here so that we can avoid waiting for buffers
    //        as much as we can.

    auto& context = _context;
    auto& ioQueue = *context.ioQueue;

    const FileId lpId  = TableIdToLinePointFileId   ( rTable );
    const FileId keyId = TableIdToLinePointKeyFileId( rTable );
    
    Fence& readFence = _readFence;
    readFence.Reset( Step2FenceId::Start );

    ioQueue.SeekBucket( lpId , 0, SeekOrigin::Begin );
    ioQueue.SeekBucket( keyId, 0, SeekOrigin::Begin );
    ioQueue.CommitCommands();

    struct BucketBuffers
    {
        uint64* linePoints;
        uint32* key;
    };

    // Clear per-bucket park left over LPs for parks
    _bucketParkLeftOversCount = 0;

    uint64 entryOffset   = 0;
    uint32 bucketsLoaded = 0;
    BucketBuffers buffers[BB_DPP3_LP_BUCKET_COUNT];

    // #TODO: Check this to optimize better
    // Since with BB_DPP3_LP_BUCKET_COUNT = 256, we get a lot of empty buckets,
    //  we need to determine which is the last bucket with entries to let the 
    //  park writer know when to write the final partial park.
    uint32 lastBucketWithEntries = 0;

    for( uint32 bucket = 0; bucket < BB_DPP3_LP_BUCKET_COUNT; bucket++ )
    {
        if( _lpBucketCounts[bucket] )
            lastBucketWithEntries = bucket;
    }
    ASSERT( lastBucketWithEntries > 0 );

    // Use a capture lambda for now, but change this to a non-capturing one later maybe
    auto LoadBucket = [&]( uint32 bucket, bool forceLoad ) -> BucketBuffers
    {
        const uint32 bucketLength = _lpBucketCounts[bucket];
        if( bucketLength < 1 )
        {
            return { nullptr, nullptr };
        }

        const size_t lpBucketSize  = sizeof( uint64 ) * bucketLength;
        const size_t mapBucketSize = sizeof( uint32 ) * bucketLength;

        uint64* linePoints = (uint64*)ioQueue.GetBuffer( lpBucketSize , forceLoad );
        uint32* key        = (uint32*)ioQueue.GetBuffer( mapBucketSize, forceLoad );

        const uint32 fenceIdx = bucket * Step2FenceId::FENCE_COUNT;

        ioQueue.ReadFile( lpId , bucket, linePoints, lpBucketSize  );
        DeleteFile( lpId, bucket );
        ioQueue.SignalFence( readFence, Step2FenceId::LPLoaded + fenceIdx );

        ioQueue.ReadFile( keyId, bucket, key, mapBucketSize );
        DeleteFile( keyId, bucket );
        ioQueue.SignalFence( readFence, Step2FenceId::MapLoaded + fenceIdx );
        
        ioQueue.CommitCommands();

        return {
            .linePoints = linePoints,
            .key        = key
        };
    };

    buffers[0] = LoadBucket( 0, true );
    bucketsLoaded++;

    for( uint32 bucket = 0; bucket <= lastBucketWithEntries; bucket++ )
    {
        const bool isLastBucket = bucket == lastBucketWithEntries;

        if( !isLastBucket )
        {
            const uint32 nextBucket   = bucket + 1;
            // #TODO: Make background loading optional if we have no buffers available,
            //        then force-load if we don't have the current bucket pre-loaded.
            buffers[nextBucket] = LoadBucket( nextBucket, true );
            bucketsLoaded++;
        }

        const uint32 bucketLength = _lpBucketCounts[bucket];
        if( bucketLength > 0 )
        {
            const uint32 fenceIdx = bucket * Step2FenceId::FENCE_COUNT;
            readFence.Wait( Step2FenceId::MapLoaded + fenceIdx, context.readWaitTime );

            uint64* linePoints = buffers[bucket].linePoints;
            uint32* key        = buffers[bucket].key;

            uint64* sortedLinePoints = _linePoints;
            uint32* sortedKey        = _rPrunedMap;

            // Sort line point w/ the key
            // Since we're skipping an iteration, the output will be 
            // stored in the temp buffers, instead on the input ones.
            RadixSort256::SortWithKey<BB_MAX_JOBS, uint64, uint32, 7>( 
                *context.threadPool, linePoints, sortedLinePoints,
                key, sortedKey, bucketLength );

            ioQueue.ReleaseBuffer( linePoints ); linePoints = nullptr;
            ioQueue.ReleaseBuffer( key        ); key        = nullptr;
            ioQueue.CommitCommands();

            // Write the map back to disk as a reverse lookup map
            WriteLPReverseLookup( rTable, sortedKey, bucket, bucketLength, entryOffset );
            
            // Deltafy, compress and write bucket to a park into the plot file
            WriteLinePointsToPark( rTable, isLastBucket, sortedLinePoints, bucketLength );
        }

        entryOffset += bucketLength;
    }

    // Delete the reset of the buckets
    for( uint32 bucket = lastBucketWithEntries; bucket < BB_DPP3_LP_BUCKET_COUNT; bucket++ )
    {
        ioQueue.DeleteFile( lpId , bucket );
        ioQueue.DeleteFile( keyId, bucket );
    }
    ioQueue.CommitCommands();

    // Set the table offset for the next table
    context.plotTablePointers[(int)rTable] = context.plotTablePointers[(int)rTable-1] + context.plotTableSizes[(int)rTable-1];
}

struct WriteLPMapJob : PrefixSumJob<WriteLPMapJob>
{
    uint32  bucket;
    uint32  entryCount;
    uint64  entryOffset;

    const uint32* inKey;
    uint64*       outMap;

    uint32* bucketCounts;

    void Run() override;
};

//-----------------------------------------------------------
void DiskPlotPhase3::WriteLPReverseLookup( 
    const TableId rTable, const uint32* key,
    const uint32 bucket , const uint32  entryCount,
    const uint64 entryOffset )
{
    constexpr uint32 BucketSize = BB_DP_BUCKET_COUNT;

    // Pack entries to a reverse lookup map and sort them
    // into their buckets of origin (before sorted to line point)
    ASSERT( entryOffset + entryCount <= 0xFFFFFFFFull );

    if( entryCount < 1 )
        return;

    auto& ioQueue = *_context.ioQueue;

    const size_t bufferSize =  sizeof( uint64 ) * entryCount;

    uint64* outMap       = (uint64*)ioQueue.GetBuffer( bufferSize );
    uint32* bucketCounts = (uint32*)ioQueue.GetBuffer( sizeof( uint32 ) * BucketSize );

    const uint32 threadCount = _context.p3ThreadCount;

    MTJobRunner<WriteLPMapJob> jobs( *_context.threadPool );

    for( uint32 i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];

        job.bucket       = bucket;
        job.entryCount   = entryCount;
        job.entryOffset  = entryOffset;

        job.inKey        = key;
        job.outMap       = outMap;
        job.bucketCounts = nullptr;
    }

    jobs[0].bucketCounts = bucketCounts;
    jobs.Run( threadCount );

    // Append to our overall bucket count
    for( uint32 i = 0; i < BucketSize; i++ )
        _lMapBucketCounts[i] += bucketCounts[i];

    #if _DEBUG
    // #TEST
    // // if( 0 )
    // {
    //     uint32 totalCount = 0;
    //     for( uint32 i = 0; i < BucketSize; i++ )
    //         totalCount += bucketCounts[i];

    //     ASSERT( totalCount == entryCount )
        
    //     const uint64* map = outMap;
    //     for( uint32 b = 0; b < BucketSize; b++ )
    //     {
    //         const uint32 count = bucketCounts[b];
            
    //         for( uint32 i = 0; i < count; i++ )
    //         {
    //             const uint64 e   = map[i];
    //             const uint32 idx = (uint32)e;
    //             ASSERT( ( idx >> 26 ) == b );
    //         }

    //         map += count;
    //     }
    // }
    #endif

    // Update count to sizes
    for( uint32 i = 0; i < BucketSize; i++ )
        bucketCounts[i] *= sizeof( uint64 );

    // Write to disk
    const FileId mapId = TableIdToLinePointMapFileId( rTable );

    ioQueue.WriteBuckets( mapId, outMap, bucketCounts );
    ioQueue.ReleaseBuffer( outMap );
    ioQueue.ReleaseBuffer( bucketCounts );
    ioQueue.CommitCommands();
}

//-----------------------------------------------------------
void WriteLPMapJob::Run()
{
    const int32 threadCount = (int32)this->JobCount();
    
    int64 entriesPerThread  = (int64)this->entryCount / threadCount;

    int64 offset = entriesPerThread * this->JobId();

    if( this->IsLastThread() )
        entriesPerThread += (int64)this->entryCount - entriesPerThread * threadCount;

    // Count how many entries we have per bucket
    // #TODO: Use arbirary bucket count here too (from 64-512) and bit-pack entries tightly
    //          then we can save at least 6 bits per entry, since we can infer it from its bucket.
    const uint32 bitShift = 32 - kExtraBits;
    constexpr uint32 BucketSize = BB_DP_BUCKET_COUNT;

    const uint32* inKey  = this->inKey + offset;
    uint64*       outMap = this->outMap;

    uint32 counts[BucketSize];
    uint32 pfxSum[BucketSize];

    memset( counts, 0, sizeof( counts ) );

    for( const uint32* key = inKey, *end = key + entriesPerThread; key < end; key++ )
    {
        const uint32 bucket = *key >> bitShift;
        counts[bucket]++;
    }

    this->CalculatePrefixSum( BucketSize, counts, pfxSum, this->bucketCounts );

    // Write into our buckets
    const uint64 entryOffset = (uint64)this->entryOffset + (uint64)offset;

    for( int64 i = 0; i < entriesPerThread; i++ )
    {
        const uint32 key    = inKey[i];
        const uint32 bucket = key >> bitShift;
        
        const uint32 writeIdx = --pfxSum[bucket];
        ASSERT( writeIdx < this->entryCount );

        uint64 map = (entryOffset + (uint64)i) << 32;
        
        // TEST
        #if _DEBUG
            // if( key == 2 ) BBDebugBreak();
        #endif

        outMap[writeIdx] = map | key;
    }

    // Wait for other thread sp that counts doesn't go out of scope
    this->SyncThreads();
}

//-----------------------------------------------------------
void DiskPlotPhase3::WriteLinePointsToPark( TableId rTable, bool isLastBucketWithEntries, const uint64* linePoints, uint32 bucketLength )
{
    ASSERT( bucketLength );

    DiskPlotContext& context = _context;
    DiskBufferQueue& ioQueue = *context.ioQueue;

    const TableId lTable     = rTable - 1;
    const size_t  parkSize   = CalculateParkSize( lTable );

    if( _bucketParkLeftOversCount )
    {
        ASSERT( _bucketParkLeftOversCount < kEntriesPerPark );

        const uint32 requiredEntriesToCompletePark = kEntriesPerPark - _bucketParkLeftOversCount;
        const uint32 entriesToCopy                 = std::min( requiredEntriesToCompletePark, bucketLength );

        // Write cross-bucket park
        byte* xBucketPark = (byte*)ioQueue.GetBuffer( parkSize );

        memcpy( _bucketParkLeftOvers + _bucketParkLeftOversCount, linePoints, entriesToCopy * sizeof( uint64 ) );
        _bucketParkLeftOversCount += entriesToCopy;

        // Don't write unless we filled the whole park, or it is the last bucket (non-filled park is allowed then)
        if( entriesToCopy < requiredEntriesToCompletePark )
        {
            ASSERT( bucketLength < requiredEntriesToCompletePark );
            return;
        }
        
        ASSERT( _bucketParkLeftOversCount == kEntriesPerPark );
        WritePark( parkSize, kEntriesPerPark, _bucketParkLeftOvers, xBucketPark, lTable );

        ioQueue.WriteFile( FileId::PLOT, 0, xBucketPark, parkSize );
        ioQueue.ReleaseBuffer( xBucketPark );
        ioQueue.CommitCommands();

        context.plotTableSizes[(int)rTable-1] += parkSize;

        // Offset our current bucket to account for the entries we just used
        linePoints   += entriesToCopy;
        bucketLength -= entriesToCopy;

        // Clear the left-overs
        _bucketParkLeftOversCount = 0;

        if( bucketLength < 1 )
            return;
    }

    uint32  parkCount       = bucketLength / kEntriesPerPark;
    uint32  leftOverEntries = bucketLength - parkCount * kEntriesPerPark;

    if( isLastBucketWithEntries && leftOverEntries )
    {
        leftOverEntries = 0;
        parkCount++;
    }
    else if( leftOverEntries )
    {
        // Save any entries that don't fill-up a full park for the next bucket
        memcpy( _bucketParkLeftOvers, linePoints + bucketLength - leftOverEntries, leftOverEntries * sizeof( uint64 ) );
        
        _bucketParkLeftOversCount = leftOverEntries;
        bucketLength -= leftOverEntries;
    }

    ASSERT( isLastBucketWithEntries || bucketLength / kEntriesPerPark * kEntriesPerPark == bucketLength );

    byte* parkBuffer = (byte*)ioQueue.GetBuffer( parkSize * parkCount );

    const size_t sizeWritten = WriteParks<BB_MAX_JOBS>( *context.threadPool, bucketLength, (uint64*)linePoints, parkBuffer, lTable );
    ASSERT( sizeWritten <= parkSize * parkCount  );

    ioQueue.WriteFile( FileId::PLOT, 0, parkBuffer, sizeWritten );
    ioQueue.ReleaseBuffer( parkBuffer );
    ioQueue.CommitCommands();

    context.plotTableSizes[(int)rTable-1] += sizeWritten;
}


///
/// Third Step
///
struct LPUnpackMapJob : MTJob<LPUnpackMapJob>
{
    uint32        bucket;
    uint32        entryCount;
    const uint64* mapSrc;
    uint32*       mapDst;

    //-----------------------------------------------------------
    static void RunJob( ThreadPool& pool, const uint32 threadCount, const uint32 bucket,
                        const uint32 entryCount, const uint64* mapSrc, uint32* mapDst )
    {
        MTJobRunner<LPUnpackMapJob> jobs( pool );

        for( uint32 i = 0; i < threadCount; i++ )
        {
            auto& job = jobs[i];
            job.bucket     = bucket;
            job.entryCount = entryCount;
            job.mapSrc     = mapSrc;
            job.mapDst     = mapDst;
        }

        jobs.Run( threadCount );
    }

    //-----------------------------------------------------------
    void Run() override
    {
        const uint64 maxEntries         = 1ull << _K ;
        const uint32 fixedBucketLength  = (uint32)( maxEntries / BB_DP_BUCKET_COUNT );
        const uint32 bucketOffset       = fixedBucketLength * this->bucket;

        const uint32 threadCount        = this->JobCount();
        uint32       entriesPerThread   = this->entryCount / threadCount;

        const uint32 offset = entriesPerThread * this->JobId();

        if( this->IsLastThread() )
            entriesPerThread += this->entryCount - entriesPerThread * threadCount;

        const uint64* mapSrc = this->mapSrc + offset;
        uint32*       mapDst = this->mapDst;

        // Unpack with the bucket id
        for( uint32 i = 0; i < entriesPerThread; i++ )
        {
            const uint64 m   = mapSrc[i];
            const uint32 idx = (uint32)m - bucketOffset;
            
            // #TODO: No need to keep track of bucketOffset, can just 
            //        mask out the bucket portion...
            ASSERT( idx < fixedBucketLength );

            mapDst[idx] = (uint32)(m >> 32);
        }
    }
};

//-----------------------------------------------------------
void DiskPlotPhase3::TableThirdStep( const TableId rTable )
{
    Log::Line( "  Step 3" );

    // Read back the packed map buffer from the current R table, then
    // write them back to disk as a single, contiguous file

    DiskPlotContext& context = _context;
    DiskBufferQueue& ioQueue = *context.ioQueue;

    constexpr uint32 BucketCount = BB_DP_BUCKET_COUNT;

    const FileId mapId = TableIdToLinePointMapFileId( rTable );

    const uint64 tableEntryCount = context.entryCounts[(int)rTable];

    const uint64 maxEntries      = 1ull << _K;
    const uint32 fixedBucketSize = (uint32)( maxEntries / BucketCount );
    const uint32 lastBucketSize  = (uint32)( tableEntryCount - fixedBucketSize * ( BucketCount - 1 ) );

    Fence& readFence = _readFence;
    readFence.Reset( 0 );

    ioQueue.SeekBucket( mapId, 0, SeekOrigin::Begin );
    ioQueue.CommitCommands();


    uint64* buffers[BucketCount] = { 0 };
    uint32  bucketsLoaded = 0;

    auto LoadBucket = [&]( const bool forceLoad ) -> void
    {
        const uint32 bucket = bucketsLoaded;

        const uint32 entryCount = _lMapBucketCounts[bucket];
        if( entryCount < 1 )
        {
            buffers[bucketsLoaded++] = nullptr;
            return;
        }

        // const size_t bucketSize = RoundUpToNextBoundaryT( entryCount * sizeof( uint64 ), 4096ul );
        const size_t bucketSize = entryCount * sizeof( uint64 );

        auto* buffer = (uint64*)ioQueue.GetBuffer( bucketSize, forceLoad );
        if( !buffer )
            return;

        ioQueue.ReadFile( mapId, bucket, buffer, bucketSize );
        ioQueue.SignalFence( readFence, bucket + 1 );
        ioQueue.CommitCommands();

        if( bucket == 0 && rTable < TableId::Table7 )
            ioQueue.SeekFile( mapId, 0, 0, SeekOrigin::Begin ); // Seek to the start to re-use this file for writing the unpacked map
        else
            ioQueue.DeleteFile( mapId, bucket );
        
        ioQueue.CommitCommands();

        buffers[bucketsLoaded++] = buffer;
    };

    LoadBucket( true );

    const uint32 maxBucketsToLoadPerIter = 2;

    for( uint32 bucket = 0; bucket < BucketCount; bucket++ )
    {
        const uint32 nextBucket   = bucket + 1;
        const bool   isLastBucket = nextBucket == BucketCount;

        // Reserve a buffer for writing
        const uint32 entryCount = _lMapBucketCounts[bucket];

        const uint32 writeEntryCount = isLastBucket ? lastBucketSize : fixedBucketSize;
        const size_t writeSize       = writeEntryCount * sizeof( uint32 );

        uint32* writeBuffer = nullptr;
        if( entryCount > 0 )
            writeBuffer = (uint32*)ioQueue.GetBuffer( writeSize, true );

        // Load next bucket
        if( !isLastBucket && bucketsLoaded < BucketCount )
        {
            uint32 maxBucketsToLoad = std::min( maxBucketsToLoadPerIter, BucketCount - bucketsLoaded );

            for( uint32 i = 0; i < maxBucketsToLoad; i++ )
            {
                const bool needNextBucket = bucketsLoaded == nextBucket;
                LoadBucket( needNextBucket );
            }
        }

        if( entryCount < 1 )
            continue;

        readFence.Wait( nextBucket, context.readWaitTime );

        // Unpack the map
        const uint64* inMap = buffers[bucket];

        // TEST
        #if _DEBUG
        // if( 0 )
        // {
        //     const uint64 maxEntries         = 1ull << _K ;
        //     const uint32 fixedBucketLength  = (uint32)( maxEntries / BB_DP_BUCKET_COUNT );
        //     const uint32 bucketOffset       = fixedBucketLength * bucket;

        //     for( int64 i = 0; i < (int64)entryCount; i++ )
        //     {
        //         const uint32 idx = ((uint32)inMap[i]);
        //         ASSERT( idx  - bucketOffset < fixedBucketLength );
        //         ASSERT( ( idx >> 26 ) == bucket );
        //     }
        // }

        // memset( writeBuffer, 0, 67108864 * sizeof( uint32 ) );
        #endif

        LPUnpackMapJob::RunJob( 
            *context.threadPool, context.p3ThreadCount, 
            bucket, entryCount, inMap, writeBuffer );

        ioQueue.ReleaseBuffer( (void*)inMap );
        ioQueue.CommitCommands();

        if( rTable < TableId::Table7 )
        {
            // Write the unpacked map back to disk
            ioQueue.WriteFile( mapId, 0, writeBuffer, writeSize );
        }
        else
        {
            // For table 7 we just write the parks to disk
            WritePark7( bucket, writeBuffer, writeEntryCount );
        }

        ioQueue.ReleaseBuffer( writeBuffer );
        ioQueue.CommitCommands();
    }
}

// Write the entries for table 7 as indices into 
// table 6 into a park in the plot file
//-----------------------------------------------------------
void DiskPlotPhase3::WritePark7( uint32 bucket, uint32* t6Indices, uint32 bucketLength )
{
    ASSERT( bucketLength );

    DiskPlotContext& context = _context;
    DiskBufferQueue& ioQueue = *context.ioQueue;

    const size_t parkSize = CDiv( (_K + 1) * kEntriesPerPark, 8 );  // #TODO: Move this to its own function

    if( _park7LeftOversCount )
    {
        ASSERT( _park7LeftOversCount < kEntriesPerPark );

        const uint32 requiredEntriesToCompletePark = kEntriesPerPark - _park7LeftOversCount;
        const uint32 entriesToCopy                 = std::min( requiredEntriesToCompletePark, bucketLength );

        // Write cross-bucket park
        byte* xBucketPark = (byte*)ioQueue.GetBuffer( parkSize );

        memcpy( _park7LeftOvers + _park7LeftOversCount, t6Indices, entriesToCopy * sizeof( uint32 ) );
        _park7LeftOversCount += entriesToCopy;

        // Don't write unless we filled the whole park, or it is the last bucket (non-filled park is allowed then)
        if( entriesToCopy < requiredEntriesToCompletePark )
        {
            ASSERT( bucketLength < requiredEntriesToCompletePark );
            return;
        }

        // #TODO: Have to zero-out any entries remaining (in case of the last park)
        
        ASSERT( _park7LeftOversCount == kEntriesPerPark );
        TableWriter::WriteP7Entries( kEntriesPerPark, _park7LeftOvers, xBucketPark );
        context.plotTableSizes[(int)TableId::Table7] += parkSize;

        ioQueue.WriteFile( FileId::PLOT, 0, xBucketPark, parkSize );
        ioQueue.ReleaseBuffer( xBucketPark );
        ioQueue.CommitCommands();

        // Offset our current bucket to account for the entries we just used
        t6Indices    += entriesToCopy;
        bucketLength -= entriesToCopy;

        // Clear the overflow entries
        _park7LeftOversCount = 0;

        if( bucketLength < 1 )
            return;
    }

    const uint32 BucketCount  = BB_DP_BUCKET_COUNT;
    const bool   isLastBucket = bucket + 1 == BucketCount;

    uint32 parkCount       = bucketLength / kEntriesPerPark;
    uint32 overflowEntries = bucketLength - parkCount * kEntriesPerPark;

    if( isLastBucket && overflowEntries )
    {
        overflowEntries = 0;
        parkCount++;
    }
    else if( overflowEntries )
    {
        // Save any entries that don't fill-up a full park for the next bucket
        memcpy( _park7LeftOvers, t6Indices + bucketLength - overflowEntries, overflowEntries * sizeof( uint32 ) );
        
        _bucketParkLeftOversCount = overflowEntries;
        bucketLength -= overflowEntries;
    }

    ASSERT( isLastBucket || bucketLength / kEntriesPerPark * kEntriesPerPark == bucketLength );

    byte* parkBuffer = (byte*)ioQueue.GetBuffer( parkSize * parkCount );

    const size_t sizeWritten = TableWriter::WriteP7<BB_MAX_JOBS>( *context.threadPool, 
                                context.p3ThreadCount, bucketLength, t6Indices, parkBuffer );
    ASSERT( sizeWritten <= parkSize * parkCount  );

    ioQueue.WriteFile( FileId::PLOT, 0, parkBuffer, sizeWritten );
    ioQueue.ReleaseBuffer( parkBuffer );
    ioQueue.CommitCommands();

    context.plotTableSizes[(int)TableId::Table7] += sizeWritten;
}

//-----------------------------------------------------------
void DiskPlotPhase3::DeleteFile( FileId fileId, uint32 bucket )
{
#if BB_DP_DBG_P3_KEEP_FILES
    return;
#endif

    _context.ioQueue->DeleteFile( fileId, bucket );
}

//-----------------------------------------------------------
void DiskPlotPhase3::DeleteBucket( FileId fileId )
{
#if BB_DP_DBG_P3_KEEP_FILES
    return;
#endif

    _context.ioQueue->DeleteBucket( fileId );
}
*/


