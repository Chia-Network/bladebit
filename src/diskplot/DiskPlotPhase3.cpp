#include "DiskPlotPhase3.h"
#include "util/BitField.h"
#include "algorithm/RadixSort.h"
#include "jobs/StripAndSortMap.h"
#include "memplot/LPGen.h"

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


struct ConvertToLPJob : MTJob<ConvertToLPJob>
{
    DiskPlotContext* context;
    TableId          rTable;
    
    uint32           bucketEntryCount;
    const uint64*    markedEntries;
    const uint32*    lMap;
    const uint32*    rMap;
    Pairs            rTablePairs;

    uint64*          linePoints;        // Buffer for line points/pruned pairs
    uint32*          rMapPruned;        // Where we store our pruned R map

    int64            prunedEntryCount;

    void Run() override;

    void DistributeToBuckets( const int64 enytryCount, uint64* linePoints, uint32* map );
};

//-----------------------------------------------------------
DiskPlotPhase3::DiskPlotPhase3( DiskPlotContext& context, const Phase3Data& phase3Data )
    : _context   ( context    )
    , _phase3Data( phase3Data )
{
    memset( _tableEntryCount, 0, sizeof( _tableEntryCount ) );

    DiskBufferQueue& ioQueue = *context.ioQueue;

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

    maxBucketLength += 1024;

    // Init our buffers
    const size_t fileBlockSize        = ioQueue.BlockSize();

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
{}

//-----------------------------------------------------------
void DiskPlotPhase3::Run()
{
    for( TableId table = TableId::Table2; table < TableId::Table7; table++ )
    {
        Log::Line( "Compressing Tables %u and %u...", table, table+1 );
        const auto timer = TimerBegin();

        ProcessTable( table );

        const auto elapsed = TimerEnd( timer );
        Log::Line( "Finished compression in %.2lf seconds.", elapsed );
    }
}

//-----------------------------------------------------------
void DiskPlotPhase3::ProcessTable( const TableId rTable )
{
    DiskPlotContext& context = _context;
    DiskBufferQueue& ioQueue = *context.ioQueue;

    // Reset Fence
    _readFence.Reset( P3FenceId::Start );

    TableFirstStep( rTable );
//    TableSecondPass( rTable );
    
}

//-----------------------------------------------------------
void DiskPlotPhase3::TableFirstStep( const TableId rTable )
{
    DiskPlotContext& context         = _context;
    DiskBufferQueue& ioQueue         = *context.ioQueue;
    Fence&           readFence       = _readFence;

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
        ioQueue.ReadFile( lMapId, 0, _lMap[0], lBucketLength * sizeof( uint32 ) );;

        // Read R Table marks
        ioQueue.ReadFile( markedEntriesFileId, 0, _markedEntries, _phase3Data.bitFieldSize );

        // Read R Table 1st bucket
        ioQueue.ReadFile( rPtrsRId, 0, _rTablePairs[0].left , rBucketLength * sizeof( uint32 ) );
        ioQueue.ReadFile( rPtrsLId, 0, _rTablePairs[0].right, rBucketLength * sizeof( uint16 ) );

        ioQueue.ReadFile( rMapId, 0, _rMap[0], rBucketLength * sizeof( int32 ) );
        ioQueue.SignalFence( readFence, 1 );

        ioQueue.CommitCommands();
    }

    // Reset offsets
    _rTableOffset = 0;
    _rTableBucket = 0;

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

        // Copy last entries from current bucket to last bucket
        memcpy( _lMap[1], _lMap[0] + context.bucketCounts[(int)lTable][bucket], P3_EXTRA_L_ENTRIES_TO_LOAD * sizeof( uint32 ) );

        // Swap buffers
        std::swap( _lMap[0]       , _lMap[1] );
        std::swap( _rMap[0]       , _rMap[1] );
        std::swap( _rTablePairs[0], _rTablePairs[1] );
    }
}

//-----------------------------------------------------------
void DiskPlotPhase3::BucketFirstStep( const TableId rTable, const uint32 bucket )
{
    DiskPlotContext& context       = _context;
    DiskBufferQueue& ioQueue       = *context.ioQueue;
    ThreadPool&      threadPool    = *context.threadPool;
    Fence&           readFence     = _readFence;

    const TableId lTable           = rTable - 1;

    const bool isLastBucket = bucket == BB_DP_BUCKET_COUNT - 1;

    // uint32 bucketEntryCountL = (uint32)( maxEntries / BB_DP_BUCKET_COUNT );
    const uint32 bucketEntryCountR = context.ptrTableBucketCounts[(int)rTable][bucket];

    // Wait for the bucket to be loaded
    readFence.Wait( bucket + 1 );

    // Convert to line point

    #if _DEBUG
    {
        const uint32 r = _rTablePairs[0].left[bucketEntryCountR-1] + _rTablePairs[0].right[bucketEntryCountR-1];
        const uint32 lTableBucketLength = _context.bucketCounts[(int)lTable][bucket] + P3_EXTRA_L_ENTRIES_TO_LOAD;
        ASSERT( r < lTableBucketLength );
    }
    #endif
    
    const uint32 prunedEntryCount = 
        PointersToLinePoints( 
            rTable,
            bucketEntryCountR, _markedEntries, 
            _lMap[0], 
            _rTablePairs[0], _rMap[0], 
            _rPrunedMap, _linePoints );

    // Distribute line points to buckets along with the map
    DistributeLinePoints( rTable, prunedEntryCount, _linePoints, _rPrunedMap );
}

//-----------------------------------------------------------
uint32 DiskPlotPhase3::PointersToLinePoints( 
    TableId rTable, 
    const uint32 entryCount, const uint64* markedEntries, 
    const uint32* lTable, 
    const Pairs pairs, const uint32* rMapIn, 
    uint32* rMapOut, uint64* outLinePoints )
{
    const uint32 threadCount = _context.threadCount;

    MTJobRunner<ConvertToLPJob> jobs( *_context.threadPool );

    for( uint32 i = 0; i < threadCount; i++ )
    {
        ConvertToLPJob& job = jobs[i];

        job.context = &_context;
        job.rTable  = rTable;

        job.bucketEntryCount = entryCount;
        job.markedEntries    = markedEntries;
        job.lMap             = lTable;
        job.rTablePairs      = pairs;
        job.rMap             = rMapIn;
        job.linePoints       = outLinePoints;
        job.rMapPruned       = rMapOut;
    }

    jobs.Run( threadCount );

    uint32 prunedEntryCount = 0;
    for( uint32 i = 0; i < threadCount; i++ )
        prunedEntryCount += (uint32)jobs[i].prunedEntryCount;

    return prunedEntryCount;
}

//-----------------------------------------------------------
void DiskPlotPhase3::DistributeLinePoints( 
    const TableId rTable, const uint32 entryCount, 
    const uint64* linePoints, const uint32* rMap )
{
    
}

//-----------------------------------------------------------
void ConvertToLPJob::Run()
{
    DiskPlotContext& context = *this->context;

    const int32 threadCount = (int32)this->JobCount();
    int64       entryCount  = (int64)( this->bucketEntryCount / threadCount );

    const int64 offset      = entryCount * (int32)this->JobId();
    
    if( this->IsLastThread() )
        entryCount += (int64)this->bucketEntryCount - entryCount * threadCount;

    const int64 end = offset + entryCount;

    const BitField markedEntries( (uint64*)this->markedEntries );

    // First, scan our entries in order to prune them
    int64 prunedLength = 0;

    for( int64 i = offset; i < end; i++)
    {
        if( markedEntries[i] )
            prunedLength ++;
    }

    this->prunedEntryCount = prunedLength;

    this->SyncThreads();

    // Set our destination offset
    int64 dstOffset = 0;

    for( int32 i = 0; i < (int32)this->JobId(); i++ )
        dstOffset += GetJob( i ).prunedEntryCount;

    // Copy pruned entries into new buffer
    // #TODO: heck if doing 1 pass per buffer performs better
    const uint32* rMap  = this->rMap;
    const Pairs   pairs = this->rTablePairs;


    struct Pair
    {
        uint32 left;
        uint32 right;
    };

    Pair*   outPairsStart = (Pair*)(this->linePoints + dstOffset);
    Pair*   outPairs      = outPairsStart;
    uint32* outRMap       = this->rMapPruned + dstOffset;

    for( int64 i = offset; i < end; i++)
    {
        if( !markedEntries[i] )
            continue;

        outPairs->left  = pairs.left[i];
        outPairs->right = outPairs->left + pairs.right[i];

        *outRMap        = rMap[i];

        outPairs++;
        outRMap++;
    }

    // Now we can convert our pruned pairs to line points
    {
        const uint32* lTable = this->lMap;
        
        uint64*       outLinePoints = this->linePoints + dstOffset;
        for( int64 i = 0; i < prunedLength; i++ )
        {
            Pair p = outPairsStart[i];
            const uint64 x = lTable[p.left ];
            const uint64 y = lTable[p.right];
            
            outLinePoints[i] = SquareToLinePoint( x, y );
        }
        // const uint64* lpEnd         = outLinePoints + prunedLength;

        // do
        // {
        //     Pair p = *((Pair*)outLinePoints);
        //     const uint64 x = lTable[p.left ];
        //     const uint64 y = lTable[p.right];
            
        //     *outLinePoints = SquareToLinePoint( x, y );

        // } while( ++outLinePoints < lpEnd );
    }
}

//-----------------------------------------------------------
void ConvertToLPJob::DistributeToBuckets( const int64 enytryCount, uint64* linePoints, uint32* map )
{

}

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