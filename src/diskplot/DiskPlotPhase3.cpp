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

    uint32           bucketEntryCount;
    const uint64*    markedEntries;
    const uint32*    lMap;
    const uint32*    rMap;
    Pairs            rTablePairs;

    uint64*          linePoints;        // Buffer for line points/pruned pairs
    uint32*          rMapPruned;        // Where we store our pruned R map

    int64            prunedEntryCount;      // Pruned entry count per thread.
    int64            totalPrunedEntryCount; // Pruned entry count accross all threads
    
    void Run() override
    ;
    // For distributing
    uint32*          bucketCounts;          // Total count of entries per bucket (used by first thread)
    uint64*          lpOutBuffer;
    uint32*          keyOutBuffer;

    void DistributeToBuckets( const int64 enytryCount, const uint64* linePoints, const uint32* map );
};

//-----------------------------------------------------------
DiskPlotPhase3::DiskPlotPhase3( DiskPlotContext& context, const Phase3Data& phase3Data )
    : _context   ( context    )
    , _phase3Data( phase3Data )
{
    memset( _tableEntryCount, 0, sizeof( _tableEntryCount ) );

    DiskBufferQueue& ioQueue = *context.ioQueue;

    // Open required files
    ioQueue.InitFileSet( FileId::LP_2, "lp_2", BB_DPP3_LP_BUCKET_COUNT );
    ioQueue.InitFileSet( FileId::LP_3, "lp_3", BB_DPP3_LP_BUCKET_COUNT );
    ioQueue.InitFileSet( FileId::LP_4, "lp_4", BB_DPP3_LP_BUCKET_COUNT );
    ioQueue.InitFileSet( FileId::LP_5, "lp_5", BB_DPP3_LP_BUCKET_COUNT );
    ioQueue.InitFileSet( FileId::LP_6, "lp_6", BB_DPP3_LP_BUCKET_COUNT );
    ioQueue.InitFileSet( FileId::LP_7, "lp_7", BB_DPP3_LP_BUCKET_COUNT );

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
    TableSecondStep( rTable );
}

///
/// First Step
///
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
            rTable,
            bucketEntryCountR, _markedEntries, 
            _lMap[0], 
            _rTablePairs[0], _rMap[0], 
            _rPrunedMap, _linePoints );

    // #TODO: Update entry count in buckets    

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
    uint64* outLinePoints = this->linePoints + dstOffset;
    {
        const uint32* lTable = this->lMap;
        
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

    this->DistributeToBuckets( prunedLength, outLinePoints, outRMap );
}

//-----------------------------------------------------------
void ConvertToLPJob::DistributeToBuckets( const int64 entryCount, const uint64* linePoints, const uint32* key )
{
    uint32 counts[BB_DPP3_LP_BUCKET_COUNT];
    uint32 pfxSum[BB_DPP3_LP_BUCKET_COUNT];

    // Count entries per bucket
    for( const uint64* lp = linePoints, *end = lp + entryCount; lp < end; lp++ )
    {
        const uint64 bucket = (*lp) >> 56; ASSERT( bucket < BB_DPP3_LP_BUCKET_COUNT );
        counts[bucket]++;
    }
    
    this->CalculatePrefixSum( BB_DPP3_LP_BUCKET_COUNT, counts, pfxSum, this->bucketCounts );

    uint64* lpOutBuffer  = nullptr;
    uint32* keyOutBuffer = nullptr;

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
        const uint64 bucket   = lp >> 56;
        const uint32 dstIndex = --pfxSum[bucket];

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
        
        ioQueue.WriteBuckets( lpFileId, lpOutBuffer, lpSizes );
        ioQueue.ReleaseBuffer( lpOutBuffer );
        ioQueue.ReleaseBuffer( lpSizes );

        ioQueue.WriteBuckets( lpKeyFilId, keyOutBuffer, keySizes );
        ioQueue.ReleaseBuffer( keyOutBuffer );
        ioQueue.ReleaseBuffer( keySizes );

        ioQueue.CommitCommands();
    }
}



///
/// Seconds Step
///
//-----------------------------------------------------------
void DiskPlotPhase3::TableSecondStep( const TableId rTable )
{
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


    uint64 entryOffset   = 0;
    uint32 bucketsLoaded = 0;
    BucketBuffers buffers[BB_DPP3_LP_BUCKET_COUNT];

    // Use a capture lampda for now, but change this to a non-capturing one later maybe
    auto LoadBucket = [&]( uint32 bucket, bool forceLoad ) -> BucketBuffers
    {
        const uint32 bucketLength = context.ptrTableBucketCounts[(int)rTable][bucket];

        const size_t lpBucketSize  = sizeof( uint64 ) * bucketLength;
        const size_t mapBucketSize = sizeof( uint32 ) * bucketLength;

        uint64* linePoints = (uint64*)ioQueue.GetBuffer( lpBucketSize , forceLoad );
        uint32* map        = (uint32*)ioQueue.GetBuffer( mapBucketSize, forceLoad );

        const uint32 fenceIdx = bucket * Step2FenceId::FENCE_COUNT;

        ioQueue.ReadFile( lpId , bucket, linePoints, lpBucketSize  );
        ioQueue.SignalFence( readFence, Step2FenceId::LPLoaded + fenceIdx );

        ioQueue.ReadFile( keyId, bucket, map, mapBucketSize );
        ioQueue.SignalFence( readFence, Step2FenceId::MapLoaded + fenceIdx );
        
        ioQueue.CommitCommands();

        return {
            .linePoints = linePoints,
            .map        = map
        };
    };

    buffers[0] = LoadBucket( 0, true );
    bucketsLoaded++;

    for( uint32 bucket = 0; bucket < BB_DPP3_LP_BUCKET_COUNT; bucket++ )
    {
        const uint32 nextBucket   = bucket + 1;
        const bool   isLastBucket = nextBucket == BB_DPP3_LP_BUCKET_COUNT - 1;

        if( !isLastBucket )
        {
            // #TODO: Make background loading optional if we have no buffers available,
            //        then force-load if we don't have the current bucket pre-loaded.
            buffers[nextBucket] = LoadBucket( nextBucket, true );
            bucketsLoaded++;
        }

        const uint32 bucketLength = context.bucketCounts[(int)rTable][bucket];
        
        const uint32 fenceIdx = bucket * Step2FenceId::FENCE_COUNT;
        readFence.Wait( Step2FenceId::MapLoaded + fenceIdx );

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

        // #TODO: Deltafy, compress and write bucket to plot file in a park

        entryOffset += bucketLength;
    }
}

struct WriteLPMapJob : MTJob<WriteLPMapJob>
{
    uint32 bucket     ;
    uint32 entryCount ;
    uint64 entryOffset;

    const uint32* inKey;
    const uint64* outMap;

    uint32* counts;
    uint32* bucketCounts;

    void Run() override;
};

//-----------------------------------------------------------
void WriteLPMapJob::Run()
{
    const uint32 threadCount = this->JobCount();
    
    uint32 entriesPerThread  = this->entryCount / threadCount;

    uint32 offset = entriesPerThread * this->JobId();

    if( this->IsLastThread() )
        entriesPerThread += this->entryCount - entriesPerThread * threadCount;
}

//-----------------------------------------------------------
void DiskPlotPhase3::WriteLPReverseLookup( 
    const TableId rTable, const uint32* key,
    const uint32 bucket , const uint32  entryCount,
    const uint64 entryOffset )
{
    // Pack entries to a reverse lookup map and sort them
    // into their buckets of origin (before sorted to line point)
    ASSERT( entryOffset + entryCount <= 0xFFFFFFFFull );

    auto& ioQueue = *_context.ioQueue;

    const size_t bufferSize =  sizeof( uint64 ) * entryCount;

    uint64* outMap       = (uint64*)ioQueue.GetBuffer( bufferSize );
    uint32* bucketCounts = (uint32*)ioQueue.GetBuffer( BB_DPP3_LP_BUCKET_COUNT * sizeof( uint32 ) );

    const uint32 threadCount = _context.threadCount;

    memset( bucketCounts, 0, sizeof( bucketCounts ) );

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

    jobs[threadCount-1].bucketCounts = bucketCounts;
    jobs.Run( threadCount );

    // Update count to sizes
    for( uint32 i = 0; i < BB_DPP3_LP_BUCKET_COUNT; i++ )
        bucketCounts[i] *= sizeof( uint64 );

    // Write to disk
    const FileId mapId = TableIdToLinePointMapFileId( rTable );

    ioQueue.WriteBuckets( mapId, outMap, bucketCounts );
    ioQueue.ReleaseBuffer( outMap );
    ioQueue.CommitCommands();
}

