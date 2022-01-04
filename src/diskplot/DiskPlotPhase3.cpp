#include "DiskPlotPhase3.h"
#include "util/BitField.h"
#include "algorithm/RadixSort.h"

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


struct LPPass1Job : MTJob<LPPass1Job>
{
    DiskPlotContext* _context;
    TableId          _rTable;
    
    size_t  _markedEntriesSize;
    uint64* _markedEntries;
    uint64* _lMap       [2];
    uint64* _rMap       [2];
    Pairs   _rTablePairs[2];
    uint64* _linePoints;        // Buffer for line points
    

    void Run() override;
};

//-----------------------------------------------------------
DiskPlotPhase3::DiskPlotPhase3( DiskPlotContext& context )
    : _context( context )
{
    memset( _tableEntryCount, 0, sizeof( _tableEntryCount ) );
}

//-----------------------------------------------------------
DiskPlotPhase3::~DiskPlotPhase3()
{

}

//-----------------------------------------------------------
void DiskPlotPhase3::Run()
{   
    DiskPlotContext& context = _context;
    DiskBufferQueue& ioQueue = *context.ioQueue;

    // Init our buffers
    {
        const uint64 maxEntries           = 1ull << _K;
        const size_t fileBlockSize        = ioQueue.BlockSize();
        
        const size_t markedEntriesSize    = RoundUpToNextBoundary( RoundUpToNextBoundary( (size_t)maxEntries / 8, 8 ), fileBlockSize );  // Round up to 64-bit boundary, then to block size boundary
        const uint64 bucketEntryCount     = maxEntries / BB_DP_BUCKET_COUNT;

        const size_t rTableMapBucketSize  = RoundUpToNextBoundary( bucketEntryCount * sizeof( uint64 ), fileBlockSize );
        const size_t rTableLPtrBucketSize = RoundUpToNextBoundary( bucketEntryCount * sizeof( uint32 ), fileBlockSize );
        const size_t rTableRPtrBucketSize = RoundUpToNextBoundary( bucketEntryCount * sizeof( uint16 ), fileBlockSize );
        
        // The L table needs some extra "prefix" space for us to store entries still needed
        // from the previous bucket, when we switch buckets tothe next one.
        const size_t lTablePrefixSize     = RoundUpToNextBoundary( kBC * sizeof( uint64 ), fileBlockSize );
        const size_t lTableBucketSize     = RoundUpToNextBoundary( bucketEntryCount * sizeof( uint64 ), fileBlockSize ) + lTablePrefixSize;

        byte* heap = context.heapBuffer;
        
        _markedEntries     = (uint64*)heap;
        _markedEntriesSize = markedEntriesSize;
        heap += markedEntriesSize;

        _rMap[0] = (uint64*)heap; heap += rTableMapBucketSize;
        _rMap[1] = (uint64*)heap; heap += rTableMapBucketSize;

        _rTablePairs[0].left = (uint32*)heap; heap += rTableLPtrBucketSize;
        _rTablePairs[1].left = (uint32*)heap; heap += rTableLPtrBucketSize;

        _rTablePairs[0].right = (uint16*)heap; heap += rTableRPtrBucketSize;
        _rTablePairs[1].right = (uint16*)heap; heap += rTableRPtrBucketSize;

        _lMap[0] = (uint64*)heap; heap += lTableBucketSize;
        _lMap[1] = (uint64*)heap; heap += lTableBucketSize;

        // Start the buffers after the prefix, since we copy carry-over data to that section.
        _lMap[0] += lTablePrefixSize;
        _lMap[1] += lTablePrefixSize;
    }

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

    _lTableEntriesLoaded = 0;
    _lBucketsLoading     = 0;
    _lBucketsConsumed    = 0;

    TableFirstPass( rTable  );
    TableSecondPass( rTable );
    
}

//-----------------------------------------------------------
void DiskPlotPhase3::TableFirstPass( const TableId rTable )
{
    DiskPlotContext& context     = _context;
    DiskBufferQueue& ioQueue     = *context.ioQueue;
    Fence&           lTableFence = _lTableFence;
    Fence&           rTableFence = _rTableFence;

    const size_t markedEntriesSize = _markedEntriesSize;
    const uint64 maxEntries        = 1ull << _K;

    const TableId lTable = rTable - 1;

    const FileId markedEntriesFileId = TableIdToMarkedEntriesFileId( rTable );
    const FileId lMapId              = rTable == TableId::Table2 ? FileId::X : TableIdToMapFileId( lTable );
    const FileId rMapId              = TableIdToMapFileId( rTable );
    const FileId rPtrsRId            = TableIdToBackPointerFileId( rTable ); 
    const FileId rPtrsLId            = rPtrsRId + 1;

    // Read the first bucket's worth of data required and the rTable's marked entries (the whole thing)
    ioQueue.SeekBucket( markedEntriesFileId, 0, SeekOrigin::Begin );
    ioQueue.SeekBucket( lMapId             , 0, SeekOrigin::Begin );
    ioQueue.SeekBucket( rMapId             , 0, SeekOrigin::Begin );
    ioQueue.SeekBucket( rPtrsRId           , 0, SeekOrigin::Begin );
    ioQueue.SeekBucket( rPtrsLId           , 0, SeekOrigin::Begin );
    ioQueue.CommitCommands();

    // Load marked entries, rPtrs, rMap and lMap (or x, if lTable is 1)
    ioQueue.ReadFile( markedEntriesFileId, 0, _markedEntries, markedEntriesSize );

    const uint64 rBucketEntryCount = maxEntries / BB_DP_BUCKET_COUNT;
    ioQueue.ReadFile( rPtrsRId, 0, _rTablePairs[0].left , rBucketEntryCount * sizeof( uint32 ) );
    ioQueue.ReadFile( rPtrsLId, 0, _rTablePairs[0].right, rBucketEntryCount * sizeof( uint16 ) );
    ioQueue.ReadFile( rMapId  , 0, _rMap[0]             , rBucketEntryCount * sizeof( uint64 ) );


    const uint32 lBucket0EntryCount = context.bucketCounts[(int)lTable][0];
    const size_t lMapEntrySize      = rTable == TableId::Table2 ? sizeof( uint32 ) : sizeof( uint64 );

    ASSERT( !_lBucketsLoading && !_lBucketsConsumed );
    LoadNextLTableMap( lTable );
    
    // Swap these initially as LoadNextLTableMap always loads in the back buffer,
    // but we want to use that initially loaded buffer as the initial front buffer.
    std::swap( _lMap[0], _lMap[1] );

    // Commit commands and wait for the first bucket to be loaded
    lTableFence.Wait();

    // Start processing buckets
    for( uint bucket = 0; bucket < BB_DP_BUCKET_COUNT; bucket++ )
    {
        // Load the next bucket on the background
        const uint32 nextBucket = bucket + 1;

        if( nextBucket < BB_DP_BUCKET_COUNT )
        {
            const bool isLastBucket = nextBucket == BB_DP_BUCKET_COUNT - 1;

            // Load the L bucket, if one needs to be loaded
            LoadNextLTableMap( lTable );
            
            // Load the R bucket
            uint64 rEntriesToLoad = rBucketEntryCount;

            // The last bucket may have less entries (or more, in the case of overflows) than the previous buckets
            if( isLastBucket )
            {
                const uint64 rEntriesRead      = bucket * (uint64)rBucketEntryCount;
                const uint64 rTableEntryCount  = context.entryCounts[(int)rTable];
                const uint64 remainingREntries = rTableEntryCount - rEntriesRead;

                FatalIf( remainingREntries > rBucketEntryCount, "Overflow entries are not supported yet." );
                    
                rEntriesToLoad = std::min( rBucketEntryCount, remainingREntries );
            }
    
            const uint32 nextRFenceIdx = nextBucket * P3FenceId::FENCE_COUNT;

            ioQueue.ReadFile( rPtrsRId, 0, _rTablePairs[1].left , rEntriesToLoad * sizeof( uint32 ) );
            ioQueue.ReadFile( rPtrsLId, 0, _rTablePairs[1].right, rEntriesToLoad * sizeof( uint16 ) );
            ioQueue.SignalFence( rTableFence, P3FenceId::RTableLoaded + nextRFenceIdx );
            
            ioQueue.ReadFile( rMapId, nextBucket, _rMap[1], rEntriesToLoad * sizeof( uint64 ) );
            ioQueue.SignalFence( rTableFence, P3FenceId::RMapLoaded + nextRFenceIdx );

            ioQueue.CommitCommands();
        }

        // Process the bucket
        BucketFirstPass( rTable, bucket );

        // Swap buffers
        std::swap( _rMap[0]       , _rMap[1] );
        std::swap( _rTablePairs[0], _rTablePairs[1] );
    }
}

//-----------------------------------------------------------
void DiskPlotPhase3::BucketFirstPass( const TableId rTable, const uint32 bucket )
{   
    DiskPlotContext& context     = _context;
    DiskBufferQueue& ioQueue     = *context.ioQueue;
    ThreadPool&      threadPool  = *context.threadPool;
    Fence&           lTableFence = _lTableFence;
    Fence&           rTableFence = _rTableFence;

    const bool   isLastBucket      = bucket + 1 == BB_DP_BUCKET_COUNT;
    const uint32 fenceIdx          = bucket * P3FenceId::FENCE_COUNT;

    const uint64 rTableEntryCount  = context.entryCounts[(int)rTable];

    const uint64 maxEntries        = 1ull << _K;
    const uint32 lEntriesPerBucket = (uint32)( maxEntries / BB_DP_BUCKET_COUNT );
    const uint32 rEntriesPerBucket = (uint32)( maxEntries / BB_DP_BUCKET_COUNT );   // #NOTE: For now they're the same, but L bucket count might be changed

    // Current offset of the entries. That is, the absolute entry index
    const uint64 lEntryOffset = (uint64)lEntriesPerBucket * _lBucketsConsumed;
    const uint64 rEntryOffset = (uint64)rEntriesPerBucket * bucket;

    uint32 rBucketEntries = rEntriesPerBucket;

    BitField markedEntries( _markedEntries );


    // Check how many entries we have for the rBucket
    uint32 rEntryCount = rEntriesPerBucket;

    if( isLastBucket )
    {
        const uint64 remainingTableEntries = rTableEntryCount - rEntryOffset;

        FatalIf( remainingTableEntries > entriesPerBucket, "Overflow entries are not yet supported." );

        rEntryCount = (uint32)std::min( rEntryCount, (uint32)remainingTableEntries );
    }

    // Ensure our R table pointers are loaded
    rTableFence.Wait( P3FenceId::RTableLoaded + fenceIdx );

    const Pairs rTablePtrs = 
    { 
        .left  = _rTablePairs[0].left, 
        .right = _rTablePairs[0].right 
    };

    /**
     * @brief #TODO: Continue work in this function when we come back.
     * 
     */
    // Process R table entries until we have finished them
    for( uint32 rEntriesProcessed = 0 ; rEntriesProcessed < rEntryCount; rEntriesProcessed += rEntryCount )
    {
        // Find which is the greatest marked R entry and what address on the L table it needs
        const uint64 maxLAddress       = entryOffset + entriesPerBucket;
        uint32       remainderREntries = 0;

        // #TODO: Do NOT do this on the last bucket
        for( int64 i = entryOffset + (int64)rEntryCount-1; i >= (int64)entryOffset; i-- )
        {
            if( !markedEntries[i] )
                continue;

            const uint64 lAddress = (uint64)rTablePtrs.left[i] + rTablePtrs.right[i];

            ASSERT( lAddress >= entryOffset );

            // Do we have this L table index loaded?
            if( lAddress <= maxLAddress )
            {
                const uint64 newEntryCount = (uint64)( i - entryOffset );
                
                remainderREntries = rEntryCount - newEntryCount;
                rEntryCount       = newEntryCount;

                break;
            }
        }

        // Ensure the L table is loaded
        lTableFence.Wait( _lBucketsConsumed );

        ASSERT( (uint64)rTablePtrs.left + rTablePtrs.right <= _lMap )

        const uint32* lTable = (uint32*)_lMap[0];

        if( rTable > TableId::Table2 )
        {
            // #TODO: Process L table map
            uint64* lMap = (uint64*)lTable;
        }

        // Convert line points 
        uint64* linePoints = _linePoints;
        uint32  lpCount    = PointersToLinePoints( rEntryCount, _markedEntries, rTablePtrs, lTable, linePoints );

        // Mark this L bucket as consumed, and load next L bucket
        _lBucketsConsumed++;
        std::swap( _lMap[0], _lMap[1] );

        if( rEntriesProcessed < rBucketEntries )
        {
            LoadNextLTableMap( rTable - 1 );
        }
    }


    // Split RMap into 2 separate 32-bit buffers containing the origin and destination (y-sorted) index.
    // #TODO: Just have these be 2 files to begin with, no need to split them then.
    // #TODO: Do this without treating this as 64-bit. For testing, for now its ok.
    // #TODO: Limit the sort to 3 passes, we don't need to do the final pass since we know
    //          all entries have the same MSByte since they are in the same bucket.
    //        NOTE: This may NOT be the case if we start writing them without the MSByte
    //        since we may have to at some point given that we can store these relative to its
    //        bucket so that we can have > 2^K entries (no dropped entries).
    // {
    //     uint64* rMap
        rTableFence.Wait( P3FenceId::RMapLoaded + fenceIdx );
    //     RadixSort256::SortWithKey<BB_MAX_JOBS>( pool, _rMap

    // }

    /// #TODO: Distribute LPs and sorted rMap into buckets
}

//-----------------------------------------------------------
void DiskPlotPhase3::LoadNextLTableMap( const TableId lTable )
{
    const uint32 bucketsLoading = _lBucketsLoading - _lBucketsConsumed;
    if( bucketsLoading >= 2 || _lBucketsLoading == BB_DP_BUCKET_COUNT )
    {
        ASSERT( bucketsLoading <= 2 );
        return;
    }

    const uint32 bucket         = _lBucketsLoading;

    const FileId lMapId         = lTable == TableId::Table1 ? FileId::X : TableIdToMapFileId( lTable );
    const size_t lMapEntrySize  = lTable == TableId::Table1 ? sizeof( uint32 ) : sizeof( uint64 );
    const uint32 lMapBucket     = lTable == TableId::Table1 ? 0 : bucket;

    const uint64 tableEntryCount = _context.entryCounts[(int)lTable];

    const uint32 entriesPerBucket = (1ull << _K) / BB_DP_BUCKET_COUNT;
    const uint64 remainder        = tableEntryCount - (uint64)entriesPerBucket * bucket;
    const uint32 entryCount       = (uint32)std::min( (uint64)entriesPerBucket, remainder );

    if( bucket + 1 == BB_DP_BUCKET_COUNT )
        FatalIf( remainder > entriesPerBucket, "Overflow entries not supported." );

    const size_t loadSize         = lMapEntrySize * entryCount;

    DiskBufferQueue& ioQueue = *_context.ioQueue;

    ioQueue.ReadFile( lMapId, lMapBucket, _lMap[1], loadSize ); // Always load in the back-buffer (_lMap[1])
    ioQueue.SignalFence( _lTableFence, bucket );
    ioQueue.CommitCommands();

    _lBucketsLoading ++;
}

//-----------------------------------------------------------
uint32 DiskPlotPhase3::PointersToLinePoints( 
    const uint32 entryCount, const uint64* markedEntries, 
    const Pairs pairs, const uint32* lTable, uint64* outLinePoints )
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