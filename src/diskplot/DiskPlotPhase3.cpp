#include "DiskPlotPhase3.h"
#include "util/BitField.h"
#include "algorithm/RadixSort.h"
#include "jobs/StripAndSortMap.h"

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


struct LPStep1Job : MTJob<LPStep1Job>
{
    DiskPlotContext* _context;
    TableId          rTable;
    
    uint64*          markedEntries;
    uint64*          lMap;
    uint64*          rMap;
    Pairs            rTablePairs;
    uint64*         _linePoints;        // Buffer for line points
    

    void Run() override;
};

//-----------------------------------------------------------
DiskPlotPhase3::DiskPlotPhase3( DiskPlotContext& context, const Phase3Data& phase3Data )
    : _context   ( context    )
    , _phase3Data( phase3Data )
{
    memset( _tableEntryCount, 0, sizeof( _tableEntryCount ) );

    DiskPlotContext& context = _context;
    DiskBufferQueue& ioQueue = *context.ioQueue;

    const uint64 maxEntries       = 1ull << _K;
    const uint64 tableMaxLength   = phase3Data.maxTableLength;
    const uint32 bucketEntryCount = phase3Data.bucketMaxSize;

    // Init our buffers
    const size_t fileBlockSize        = ioQueue.BlockSize();

    const size_t markedEntriesSize    = phase3Data.bitFieldSize;
    const size_t rTableMapBucketSize  = RoundUpToNextBoundary( bucketEntryCount * sizeof( uint64 ), fileBlockSize );
    const size_t rTableLPtrBucketSize = RoundUpToNextBoundary( bucketEntryCount * sizeof( uint32 ), fileBlockSize );
    const size_t rTableRPtrBucketSize = RoundUpToNextBoundary( bucketEntryCount * sizeof( uint16 ), fileBlockSize );
    
    const size_t lTableBucketSize     = RoundUpToNextBoundary( bucketEntryCount * sizeof( uint64 ), fileBlockSize );

    const size_t lpBucketSize         = RoundUpToNextBoundary( bucketEntryCount * sizeof( uint64 ), fileBlockSize );

    byte* heap = context.heapBuffer;

    _markedEntries     = (uint64*)heap;
    heap += markedEntriesSize;

    _rMap[0]              = (uint64*)heap; heap += rTableMapBucketSize;
    _rMap[1]              = (uint64*)heap; heap += rTableMapBucketSize;

    _rTablePairs[0].left  = (uint32*)heap; heap += rTableLPtrBucketSize;
    _rTablePairs[1].left  = (uint32*)heap; heap += rTableLPtrBucketSize;

    _rTablePairs[0].right = (uint16*)heap; heap += rTableRPtrBucketSize;
    _rTablePairs[1].right = (uint16*)heap; heap += rTableRPtrBucketSize;

    _lMap[0]    = (uint64*)heap; heap += lTableBucketSize;
    _lMap[1]    = (uint64*)heap; heap += lTableBucketSize;

    _tmpRMap    = (uint64*)heap; heap += rTableMapBucketSize;
    _linePoints = (uint64*)heap; heap += lpBucketSize;

    size_t totalSize = 
        markedEntriesSize        + 
        rTableMapBucketSize  * 3 + 
        rTableLPtrBucketSize * 2 + 
        rTableRPtrBucketSize * 2 + 
        lpBucketSize;

    // Reset our heap to the remainder of what we're not using
    const size_t fullHeapSize  = context.heapSize + context.ioHeapSize;
    const size_t heapRemainder = fullHeapSize - totalSize;

    ioQueue.ResetHeap( heapRemainder, heap );
}

//-----------------------------------------------------------
DiskPlotPhase3::~DiskPlotPhase3()
{

}

//-----------------------------------------------------------
void DiskPlotPhase3::Run()
{
    for( TableId table = TableId::Table2; table < TableId::Table7; table++ )
    {
        ProcessTable( table );
    }
}

//-----------------------------------------------------------
void DiskPlotPhase3::ProcessTable( const TableId rTable )
{
    DiskPlotContext& context = _context;
    DiskBufferQueue& ioQueue = *context.ioQueue;

    // Reset Fence
    _rTableFence.Reset( P3FenceId::Start );
    _lTableFence.Reset( 0 );

    TableFirstStep( rTable );
//    TableSecondPass( rTable );
    
}

//-----------------------------------------------------------
void DiskPlotPhase3::TableFirstStep( const TableId rTable )
{
    DiskPlotContext& context         = _context;
    DiskBufferQueue& ioQueue         = *context.ioQueue;
    Fence&           lTableFence     = _lTableFence;
    Fence&           rTableFence     = _rTableFence;

    const TableId lTable             = rTable - 1;
    const uint64  maxEntries         = 1ull << _K;
    const uint64  lTableEntryCount   = context.entryCounts[(int)lTable];
    const uint64  rTableEntryCount   = context.entryCounts[(int)rTable];

    const FileId markedEntriesFileId = TableIdToMarkedEntriesFileId( rTable );
    const FileId lMapId              = rTable == TableId::Table2 ? FileId::X : TableIdToMapFileId( lTable );
    const FileId rMapId              = TableIdToMapFileId( rTable );
    const FileId rPtrsRId            = TableIdToBackPointerFileId( rTable ); 
    const FileId rPtrsLId            = rPtrsRId + 1;

    // Prepare our files for reading
    ioQueue.SeekBucket( markedEntriesFileId, 0, SeekOrigin::Begin );
    ioQueue.SeekBucket( lMapId             , 0, SeekOrigin::Begin );
    ioQueue.SeekBucket( rMapId             , 0, SeekOrigin::Begin );
    ioQueue.SeekBucket( rPtrsRId           , 0, SeekOrigin::Begin );
    ioQueue.SeekBucket( rPtrsLId           , 0, SeekOrigin::Begin );
    ioQueue.CommitCommands();

    const size_t lMapEntrySize = rTable == TableId::Table2 ? sizeof( uint32 ) : sizeof( uint64 );
    const size_t rMapEntrySize = rTable == TableId::Table2 ? sizeof( uint64 ) : sizeof( uint32 );

    // Read first bucket
    {
        const uint32 bucketLength = (uint32)( maxEntries / BB_DP_BUCKET_COUNT );

        // Read L Table 1st bucket
        ioQueue.ReadFile( lMapId, 0, _lMap[0], bucketLength * lMapEntrySize );
        ioQueue.SignalFence( _lTableFence, 1 );

        // Read R Table marks
        ioQueue.ReadFile( markedEntriesFileId, 0, _markedEntries, _phase3Data.bitFieldSize );

        // Read R Table 1st bucket
        ioQueue.ReadFile( rPtrsRId, 0, _rTablePairs[0].left , bucketLength * sizeof( uint32 ) );
        ioQueue.ReadFile( rPtrsLId, 0, _rTablePairs[0].right, bucketLength * sizeof( uint16 ) );
        ioQueue.SignalFence( _rTableFence, P3FenceId::RTableLoaded );

        ioQueue.ReadFile( rMapId  , 0, _rMap[0], bucketLength * rMapEntrySize );
        ioQueue.SignalFence( _rTableFence, P3FenceId::RMapLoaded );

        ioQueue.CommitCommands();
    }

    // Start processing buckets
    for( uint bucket = 0; bucket < BB_DP_BUCKET_COUNT; bucket++ )
    {
        const bool isCurrentBucketLastBucket = bucket == BB_DP_BUCKET_COUNT - 1;
        
        if( !isCurrentBucketLastBucket )
        {
            // Load the next bucket on the background
            const uint32 nextBucket             = bucket + 1;
            const bool   isNextBucketLastBucket = nextBucket == BB_DP_BUCKET_COUNT - 1;
            
            uint32 nextBucketLengthL = (uint32)( maxEntries / BB_DP_BUCKET_COUNT );
            uint32 nextBucketLengthR = nextBucketLengthL;

            if( isNextBucketLastBucket )
            {
                const uint64 nEntriesLoaded = (BB_DP_BUCKET_COUNT-1) * ( maxEntries / BB_DP_BUCKET_COUNT );

                nextBucketLengthL = (uint32)( lTableEntryCount - nEntriesLoaded );
                nextBucketLengthR = (uint32)( rTableEntryCount - nEntriesLoaded );
            }

            // Load L Table
            const uint32 lMapBucket = lTable == TableId::Table1 ? 0 : nextBucket;
            ioQueue.ReadFile( lMapId, lMapBucket, _lMap[1], nextBucketLengthL * lMapEntrySize );
            ioQueue.SignalFence( _lTableFence, nextBucket + 1 );

            // Load R Table
            const uint32 nextRFenceIdx = nextBucket * P3FenceId::FENCE_COUNT;

            ioQueue.ReadFile( rPtrsRId, 0, _rTablePairs[1].left , nextBucketLengthR * sizeof( uint32 ) );
            ioQueue.ReadFile( rPtrsLId, 0, _rTablePairs[1].right, nextBucketLengthR * sizeof( uint16 ) );
            ioQueue.SignalFence( rTableFence, P3FenceId::RTableLoaded + nextRFenceIdx );
            
            const uint32 rMapBucket = rTable == TableId::Table2 ? nextBucket : 0;
            ioQueue.ReadFile( rMapId, rMapBucket, _rMap[1], nextBucketLengthR * rMapEntrySize );
            ioQueue.SignalFence( rTableFence, P3FenceId::RMapLoaded + nextRFenceIdx );

            ioQueue.CommitCommands();
        }

        // Process the bucket
        BucketFirstStep( rTable, bucket );

        // Swap buffers
        std::swap( _lMap[0]       , _lMap[1] );
        std::swap( _rMap[0]       , _rMap[1] );
        std::swap( _rTablePairs[0], _rTablePairs[1] );
    }
}

//-----------------------------------------------------------
void DiskPlotPhase3::BucketFirstStep( const TableId rTable, const uint32 bucket, 
                                      const uint32 entryCount )
{
    DiskPlotContext& context     = _context;
    DiskBufferQueue& ioQueue     = *context.ioQueue;
    ThreadPool&      threadPool  = *context.threadPool;
    Fence&           lTableFence = _lTableFence;
    Fence&           rTableFence = _rTableFence;

    const TableId lTable           = rTable - 1;
    const uint64  maxEntries       = 1ull << _K;
    const uint64  lTableEntryCount = context.entryCounts[(int)lTable];
    const uint64  rTableEntryCount = context.entryCounts[(int)rTable];

    const bool isLastBucket = bucket == BB_DP_BUCKET_COUNT - 1;

    uint32 nextBucketLengthL = (uint32)( maxEntries / BB_DP_BUCKET_COUNT );
    uint32 nextBucketLengthR = nextBucketLengthL;

    if( isLastBucket )
    {
        const uint64 nEntriesLoaded = (BB_DP_BUCKET_COUNT-1) * ( maxEntries / BB_DP_BUCKET_COUNT );

        nextBucketLengthL = (uint32)( lTableEntryCount - nEntriesLoaded );
        nextBucketLengthR = (uint32)( rTableEntryCount - nEntriesLoaded );
    }

    // Wait for the L bucket to be loaded
    lTableFence.Wait( bucket + 1 );

    // Strip and sort the L map.
    uint32* lTableMap = nullptr;

    if( rTable > TableId::Table2 )
    {
        lTableMap = (uint32*)_tmpRMap;
        uint32* outKey = lTableMap + lTableEntryCount;

        StripMapJob::RunJob( *context.threadPool, context.threadCount,
            lTableEntryCount, _lMap[0], outKey, lTableMap );
    }
    else
        lTableMap = (uint32*)_lMap[0];


    // Convert to line points
    // uint64* linePoints = (uint64*)ioQueue.GetBuffer( sizeof( uint64 ) * entryCount );
    // uint32* rMapStripped = (uint32*)ioQueue.GetBuffer( sizeof( uint64 ) * entryCount );

    const uint32 rFenceIdx = bucket * P3FenceId::FENCE_COUNT;
    // rTableFence.Wait( P3FenceId::RTableLoaded + rFenceIdx );
    rTableFence.Wait( P3FenceId::RMapLoaded + rFenceIdx );

    uint32 prunedEntryCount = PointersToLinePoints( entryCount, _markedEntries, (uint32*)_rMap[0], _rTablePairs[0], lTableMap, _linePoints );

    // Distribute line points to buckets along with the map
    // void DistributeLinePoints( const TableId rTable, const uint32 entryCount, const uint64* linePoints, const uint32* rMap );

}

//-----------------------------------------------------------
uint32 DiskPlotPhase3::PointersToLinePoints( 
    const uint32 entryCount, const uint64* markedEntries, 
    const uint32* lTable, const Pairs pairs,
    const uint32* rMapIn, uint32* rMapOut,
    uint64* outLinePoints )
{
    return 0;
}

//-----------------------------------------------------------
// void LPJob::Run()
// {
//     FirstPass();
//     SyncThreads();
//     SecondPass();
// }


// //-----------------------------------------------------------
// void LPJob::FirstPass()
// {
//     DiskPlotContext& context = *this->_context;
//     DiskBufferQueue& ioQueue = *context.ioQueue;

//     const TableId rTable = this->_rTable;
//     const TableId lTable = rTable - 1;
//     ASSERT( rTable >= TableId::Table2 );

//     BitField markedEntries( this->_markedEntries );

//     // #TODO: Suspend the other threads hard here, don't just spin
//     if( this->IsControlThread() )
//     {
//         this->LockThreads();

//         Fence& fence = *_readFence;

//         fence.Reset( P3FenceId::Start );

//         const FileId lMapId  = rTable == TableId::Table2 ? FileId::X : TableIdToMapFileId( lTable );
//         const FileId rPtrId  = TableIdToBackPointerFileId( rTable );
//         const FileId rMapId  = TableIdToMapFileId( rTable );
//         const FileId rPtrsId = TableIdToBackPointerFileId( rTable );

//          // Load rPtrs and rMap and lMap (or x, if lTable is 1)
//         ioQueue.ReadFile( TableIdToMarkedEntriesFileId( rTable ), 0, _markedEntries, _markedEntriesSize );
//         ioQueue.SignalFence( fence, P3FenceId::MarkedEntriesLoaded );
        
//         ioQueue.ReadFile( rPtrId    , 0, _rTablePairs[0].left, rBucketEntryCount * sizeof( uint32 ) );
//         ioQueue.ReadFile( rPtrId + 1, 0, _rTablePairs[0].left, rBucketEntryCount * sizeof( uint16 ) );
//         ioQueue.SignalFence( fence, P3FenceId::RTableLoaded );

//         ioQueue.ReadFile( rMapId, 0, _rMap[0], rBucketEntryCount * sizeof( uint64 ) );
//         ioQueue.SignalFence( fence, P3FenceId::RMapLoaded );


//         // We always need 2 L table buckets loaded at a time to ensure that our
//         // R ptrs have the entries they expect, since the R entries are offset by BC groups size
//         const uint32 lBucket0EntryCount = context.bucketCounts[(int)lTable][0];
//         const uint32 lBucket1EntryCount = context.bucketCounts[(int)lTable][1];

//         const size_t lMapEntrySize = rTable == TableId::Table2 ? sizeof( uint32 ) : sizeof( uint64 );

//         ioQueue.ReadFile( lMapId, 0, _lMap[0], lBucket0EntryCount * lMapEntrySize );
//         ioQueue.SignalFence( fence, P3FenceId::RTableLoaded );
//         ioQueue.ReadFile( lMapId, 1, _lMap[1], lBucket0EntryCount * lMapEntrySize );
//         ioQueue.SignalFence( fence, P3FenceId::RTableLoaded + P3FenceId::FENCE_COUNT );
//         ioQueue.CommitCommands();

//         fence.Wait( P3FenceId::RTableLoaded + P3FenceId::FENCE_COUNT );

//         ioQueue.SeekBucket( lMapId , 0, SeekOrigin::Begin );
//         ioQueue.SeekBucket( rMapId , 0, SeekOrigin::Begin );
//         ioQueue.SeekBucket( rPtrsId, 0, SeekOrigin::Begin );
//         ioQueue.CommitCommands();

//         this->ReleaseThreads();
//     }
//     else
//     {
//         this->WaitForRelease();
//     }


//     for( uint bucket = 0; bucket < BB_DP_BUCKET_COUNT; bucket++ )
//     {
        
//     }
// }

// //-----------------------------------------------------------
// void LPJob::FirstPassBucket( const uint32 bucket )
// {

// }