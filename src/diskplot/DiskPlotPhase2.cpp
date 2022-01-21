#include "DiskPlotPhase2.h"
#include "util/BitField.h"
#include "algorithm/RadixSort.h"
#include "jobs/StripAndSortMap.h"

// Fence ids used when loading buckets
struct FenceId
{
    enum
    {
        None = 0,
        MapLoaded,
        PairsLoaded,

        FenceCount
    };
};

struct MarkJob : MTJob<MarkJob>
{
    TableId          table;
    uint32           entryCount;
    Pairs            pairs;
    const uint32*    map;

    DiskPlotContext* context;

    uint64*          lTableMarkedEntries;
    uint64*          rTableMarkedEntries;

    uint64           lTableOffset;
    uint32           pairBucket;
    uint32           pairBucketOffset;


public:
    void Run() override;

    template<TableId table>
    void MarkEntries();

    template<TableId table>
    inline int32 MarkStep( int32 i, const int32 entryCount, BitField lTable, const BitField rTable,
                           uint64 lTableOffset, const Pairs& pairs, const uint32* map );
};

//-----------------------------------------------------------
DiskPlotPhase2::DiskPlotPhase2( DiskPlotContext& context )
    : _context( context )
{
    memset( _bucketBuffers, 0, sizeof( _bucketBuffers ) );
}

//-----------------------------------------------------------
DiskPlotPhase2::~DiskPlotPhase2()
{
}

//-----------------------------------------------------------
void DiskPlotPhase2::Run()
{
    DiskPlotContext& context = _context;
    DiskBufferQueue& queue   = *context.ioQueue;

    #if BB_DP_DBG_SKIP_PHASE_2
    {
        return;
    }
    #endif

    uint64 largestTableLength = 1;
    for( TableId table = TableId::Table2; table <= TableId::Table7; table++ )
        largestTableLength = std::max( largestTableLength, context.entryCounts[(int)table] );
    

    // Determine what size we can use to load
    // #TODO: Support overflow entries.
    const uint64 maxEntries          = 1ull << _K;
    const size_t bitFieldSize        = RoundUpToNextBoundary( (size_t)largestTableLength / 8, 8 );  // Round up to 64-bit boundary
    const size_t bitFieldBuffersSize = bitFieldSize * 2;
    
    const uint32 mapBucketEvenSize   = (uint32)( maxEntries / BB_DP_BUCKET_COUNT );
    const uint32 mapBucketMaxSize    = std::max( mapBucketEvenSize, (uint32)( largestTableLength - mapBucketEvenSize * (BB_DP_BUCKET_COUNT-1) ) );
    const size_t tmpMapBufferSize    = mapBucketMaxSize * sizeof( uint64 );

    const size_t fullHeapSize        = context.heapSize + context.ioHeapSize;
    const size_t heapRemainder       = fullHeapSize - bitFieldBuffersSize - tmpMapBufferSize;

    // Reserve the remainder of the heap for reading R table backpointers
    queue.ResetHeap( heapRemainder, context.heapBuffer + bitFieldBuffersSize + tmpMapBufferSize );
    
    // Prepare 2 marking bitfields for dual-buffering
    uint64* bitFields[2];
    bitFields[0] = (uint64*)context.heapBuffer;
    bitFields[1] = (uint64*)( context.heapBuffer + bitFieldSize );

    // Prepare map buffer
    _tmpMap = (uint64*)( context.heapBuffer + bitFieldSize*2 );

    // Prepare our fences
    Fence bitFieldFence, bucketLoadFence, mapWriteFence;
    _bucketReadFence = &bucketLoadFence;
    _mapWriteFence   = &mapWriteFence;

    // Set write fence as signalled initially
    _mapWriteFence->Signal();

    // Mark all tables
    FileId lTableFileId = FileId::MARKED_ENTRIES_6;

    for( TableId table = TableId::Table7; table > TableId::Table2; table = table-1 )
    {
        const auto timer = TimerBegin();

        uint64* lMarkingTable = bitFields[0];
        uint64* rMarkingTable = bitFields[1];

        MarkTable( table, lMarkingTable, rMarkingTable );

        //
        // #TEST
        //
        #if 0
        if( 0 )
        {
            uint32* lPtrBuf = bbcvirtalloc<uint32>( 1ull << _K );
            uint16* rPtrBuf = bbcvirtalloc<uint16>( 1ull << _K );

            byte* rMarkedBuffer = bbcvirtalloc<byte>( 1ull << _K );
            byte* lMarkedBuffer = bbcvirtalloc<byte>( 1ull << _K );
            
            for( TableId rTable = TableId::Table7; rTable > TableId::Table1; rTable = rTable-1 )
            {
                const TableId lTable = rTable-1;

                const uint64 rEntryCount = context.entryCounts[(int)rTable];
                const uint64 lEntryCount = context.entryCounts[(int)lTable];

                // BitField rMarkedEntries( (uint64*)rMarkedBuffer );
                // BitField lMarkedEntries( (uint64*)lMarkedBuffer );

                Log::Line( "Reading R table %u...", rTable+1 );
                {
                    const FileId rTableIdL = TableIdToBackPointerFileId( rTable );
                    const FileId rTableIdR = (FileId)((int)rTableIdL + 1 );

                    Fence fence;
                    queue.ReadFile( rTableIdL, 0, lPtrBuf, sizeof( uint32 ) * rEntryCount );
                    queue.ReadFile( rTableIdR, 0, rPtrBuf, sizeof( uint16 ) * rEntryCount );
                    queue.SignalFence( fence );
                    queue.CommitCommands();
                    fence.Wait();
                }


                uint32* lPtr = lPtrBuf;
                uint16* rPtr = rPtrBuf;
                
                uint64 lEntryOffset = 0;
                uint64 rTableOffset = 0;

                Log::Line( "Marking entries..." );
                for( uint32 bucket = 0; bucket < BB_DP_BUCKET_COUNT; bucket++ )
                {
                    const uint32 rBucketCount = context.ptrTableBucketCounts[(int)rTable][bucket];
                    const uint32 lBucketCount = context.bucketCounts[(int)lTable][bucket];

                    for( uint e = 0; e < rBucketCount; e++ )
                    {
                        // #NOTE: The bug is related to this.
                        //        Somehow the entries we get from the R table
                        //        are not filtering properly...
                        //        We tested without this and got the exact same
                        //        results from the reference implementation
                        if( rTable < TableId::Table7 )
                        {
                            const uint64 rIdx = rTableOffset + e;
                            // if( !rMarkedEntries.Get( rIdx ) )
                            if( !rMarkedBuffer[rIdx] )
                                continue;
                        }

                        uint64 l = (uint64)lPtr[e] + lEntryOffset;
                        uint64 r = (uint64)rPtr[e] + l;

                        ASSERT( l < ( 1ull << _K ) );
                        ASSERT( r < ( 1ull << _K ) );

                        lMarkedBuffer[l] = 1;
                        lMarkedBuffer[r] = 1;
                        // lMarkedEntries.Set( l );
                        // lMarkedEntries.Set( r );
                    }

                    lPtr += rBucketCount;
                    rPtr += rBucketCount;

                    lEntryOffset += lBucketCount;
                    rTableOffset += context.bucketCounts[(int)rTable][bucket];
                }

                uint64 prunedEntryCount = 0;
                Log::Line( "Counting entries." );
                for( uint64 e = 0; e < lEntryCount; e++ )
                {
                    if( lMarkedBuffer[e] )
                        prunedEntryCount++;
                    // if( lMarkedEntries.Get( e ) )
                }

                Log::Line( " %llu/%llu (%.2lf%%)", prunedEntryCount, lEntryCount,
                    ((double)prunedEntryCount / lEntryCount) * 100.0 );
                Log::Line("");

                // Swap marking tables and zero-out the left one.
                std::swap( lMarkedBuffer, rMarkedBuffer );
                memset( lMarkedBuffer, 0, 1ull << _K );
            }
        }
        #endif

        // Ensure the last table finished writing to the bitfield
        if( table < TableId::Table7 )
            bitFieldFence.Wait();

        // Submit l marking table for writing
        queue.WriteFile( lTableFileId, 0, lMarkingTable, bitFieldSize );
        queue.SignalFence( bitFieldFence );
        queue.CommitCommands();

        // Swap marking tables
        std::swap( bitFields[0], bitFields[1] );
        lTableFileId = (FileId)( (int)lTableFileId - 1 );

        const double elapsed = TimerEnd( timer );
        Log::Line( "Finished marking table %d in %.2lf seconds.", table, elapsed );

        // #TEST:
        // if( table < TableId::Table7 )
        {
            BitField markedEntries( bitFields[1] );
            uint64 lTableEntries = context.entryCounts[(int)table-1];

            uint64 bucketsTotalCount = 0;
            for( uint64 e = 0; e < BB_DP_BUCKET_COUNT; ++e )
                bucketsTotalCount += context.ptrTableBucketCounts[(int)table-1][e];

            ASSERT( bucketsTotalCount == lTableEntries );

            uint64 lTablePrunedEntries = 0;

            for( uint64 e = 0; e < lTableEntries; ++e )
            {
                if( markedEntries.Get( e ) )
                    lTablePrunedEntries++;
            }

            Log::Line( "Table %u entries: %llu/%llu (%.2lf%%)", table,
                       lTablePrunedEntries, lTableEntries, ((double)lTablePrunedEntries / lTableEntries ) * 100.0 );
            Log::Line( "" );

        }
    }

    bitFieldFence.Wait();
    queue.CompletePendingReleases();
}


//-----------------------------------------------------------
void DiskPlotPhase2::MarkTable( TableId table, uint64* lTableMarks, uint64* rTableMarks )
{
    DiskPlotContext& context = _context;
    DiskBufferQueue& queue   = *context.ioQueue;

    const FileId rMapId       = table < TableId::Table7 ? TableIdToMapFileId( table ) : FileId::None;
    const FileId rTableLPtrId = TableIdToBackPointerFileId( table );
    const FileId rTableRPtrId = (FileId)((int)rTableLPtrId + 1 );

    // Seek the table files back to the beginning
    queue.SeekFile( rTableLPtrId, 0, 0, SeekOrigin::Begin );
    queue.SeekFile( rTableRPtrId, 0, 0, SeekOrigin::Begin );
    queue.CommitCommands();

    if( rMapId != FileId::None )
    {
        for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
            queue.SeekFile( rMapId, i, 0, SeekOrigin::Begin );
        
        queue.CommitCommands();
    }

    const uint64 maxEntries          = 1ull << _K;
    const uint32 maxEntriesPerBucket = (uint32)( maxEntries / (uint64)BB_DP_BUCKET_COUNT );
    const uint64 tableEntryCount     = context.entryCounts[(int)table];
    
    _bucketsLoaded = 0;
    _bucketReadFence->Reset( 0 );

    const uint32 threadCount = context.threadCount;

    uint64 lTableEntryOffset     = 0;
    uint32 pairBucket            = 0;   // Pair bucket we are processing (may be different than 'bucket' which refers to the map bucket)
    uint32 pairBucketEntryOffset = 0;   // Offset in the current pair bucket 

    for( uint32 bucket = 0; bucket - BB_DP_BUCKET_COUNT; bucket++ )
    {
        uint32  bucketEntryCount;
        uint64* unsortedMapBuffer;
        Pairs   pairs;

        // Load as many buckets as we can in the background
        LoadNextBuckets( table, bucket, unsortedMapBuffer, pairs, bucketEntryCount );

        uint32* map = nullptr;
        const uint32 waitFenceId = bucket * FenceId::FenceCount;
        
        if( rMapId != FileId::None )
        {
            // Wait for the map to finish loading
            _bucketReadFence->Wait( FenceId::MapLoaded + waitFenceId );

            // Ensure the map buffer isn't being used anymore (we use the tmp map buffer for this)
            _mapWriteFence->Wait();

            // Sort the lookup map and strip out the origin index
            map = SortAndStripMap( unsortedMapBuffer, bucketEntryCount );

            // Write the map back to disk & release the buffer
            // queue.ReleaseBuffer( _bucketBuffers[bucket].map );
            // queue.WriteFile( rMapId, 0, map, bucketEntryCount * sizeof( uint32 ) );
            queue.SignalFence( *_mapWriteFence );
            queue.CommitCommands();
        }

        // Wait for the pairs to finish loading
        _bucketReadFence->Wait( FenceId::PairsLoaded + waitFenceId );

        // Mark the entries on this bucket
        MTJobRunner<MarkJob> jobs( *context.threadPool );

        for( uint i = 0; i < threadCount; i++ )
        {
            MarkJob& job = jobs[i];

            job.table               = (TableId)table;
            job.entryCount          = bucketEntryCount;
            job.pairs               = pairs;
            job.map                 = map;
            job.context             = &context;

            job.lTableMarkedEntries = lTableMarks;
            job.rTableMarkedEntries = rTableMarks;

            job.lTableOffset        = lTableEntryOffset;
            job.pairBucket          = pairBucket;
            job.pairBucketOffset    = pairBucketEntryOffset;
        }

        jobs.Run( threadCount );

        // Release the paiors buffer we just used
        ASSERT( _bucketBuffers[bucket].pairs.left < queue.Heap().Heap() + queue.Heap().HeapSize()  );
        // queue.ReleaseBuffer( _bucketBuffers[bucket].pairs.left );
        // queue.CommitCommands();

        // Update our offsets
        lTableEntryOffset     = jobs[0].lTableOffset;
        pairBucket            = jobs[0].pairBucket;
        pairBucketEntryOffset = jobs[0].pairBucketOffset;
    }
}

//-----------------------------------------------------------
void DiskPlotPhase2::LoadNextBuckets( TableId table, uint32 bucket, uint64*& outMapBuffer, Pairs& outPairsBuffer, uint32& outBucketEntryCount )
{
    DiskPlotContext& context = _context;
    DiskBufferQueue& queue   = *context.ioQueue;

    const FileId rMapId              = table < TableId::Table7 ? TableIdToMapFileId( table ) : FileId::None;
    const FileId rTableLPtrId        = TableIdToBackPointerFileId( table );
    const FileId rTableRPtrId        = (FileId)((int)rTableLPtrId + 1 );

    const uint64 maxEntries          = 1ull << _K;
    const uint32 maxEntriesPerBucket = (uint32)( maxEntries / (uint64)BB_DP_BUCKET_COUNT );
    const uint64 tableEntryCount     = context.entryCounts[(int)table];

    // Load as many buckets as we're able to
    const uint32 maxBucketsToLoad = _bucketsLoaded + 2;   // Only load 2 buckets per pass max for now (Need to allow space for map and table writes as well)

    while( _bucketsLoaded < BB_DP_BUCKET_COUNT )
    {
        const uint32 bucketToLoadEntryCount = _bucketsLoaded < BB_DP_BUCKET_COUNT - 1 ?
                                              maxEntriesPerBucket :
                                              (uint32)( tableEntryCount - maxEntriesPerBucket * ( BB_DP_BUCKET_COUNT - 1 ) ); // Last bucket

        // #TODO: I think we need to ne loading a different amount for the L table and the R table on the last bucket.
        // #TODO: Block-align size?
        // Reserve a buffer to load both a map bucket and the same amount of entries worth of pairs.
        const size_t mapReadSize  = rMapId != FileId::None ? sizeof( uint64 ) * bucketToLoadEntryCount : 0;
        const size_t lReadSize    = sizeof( uint32 ) * bucketToLoadEntryCount;
        const size_t rReadSize    = sizeof( uint16 ) * bucketToLoadEntryCount;
        const size_t pairReadSize = lReadSize + rReadSize;
        const size_t totalSize    = mapReadSize + pairReadSize;

        // Break out if a buffer isn't available, and we don't actually require one
        if( !queue.Heap().CanAllocate( totalSize ) && _bucketsLoaded > bucket )
            break;
        
        PairAndMap& buffer = _bucketBuffers[_bucketsLoaded];  // Store the buffer for the other threads to use
        ZeroMem( &buffer );
        
        if( mapReadSize > 0 )
            buffer.map = bbvirtalloc<uint64>( mapReadSize ); //(uint64*)queue.GetBuffer( mapReadSize , true );

        buffer.pairs.left  = bbvirtalloc<uint32>( pairReadSize ); //(uint32*)queue.GetBuffer( pairReadSize, true );
        buffer.pairs.right = (uint16*)( buffer.pairs.left + bucketToLoadEntryCount );

        const uint32 loadFenceId = _bucketsLoaded * FenceId::FenceCount;

        if( mapReadSize > 0 )
        {
            queue.ReadFile( rMapId, _bucketsLoaded, buffer.map, mapReadSize );
            queue.SignalFence( *_bucketReadFence, FenceId::MapLoaded + loadFenceId );

            // Seek the file back to origin, and over-write it.
            // If it's not the origin bucket, then just delete the file, don't need it anymore
            // if( _bucketsLoaded == 0 )
            //     queue.SeekFile( rMapId, 0, 0, SeekOrigin::Begin );
            // else
                // queue.DeleteFile( rMapId, _bucketsLoaded );
        }

        queue.ReadFile( rTableLPtrId, 0, buffer.pairs.left , lReadSize );
        queue.ReadFile( rTableRPtrId, 0, buffer.pairs.right, rReadSize );
        queue.SignalFence( *_bucketReadFence, FenceId::PairsLoaded + loadFenceId );

        queue.CommitCommands();
        _bucketsLoaded++;

        if( _bucketsLoaded >= maxBucketsToLoad )
            break;
    }
    

    {
        ASSERT( _bucketsLoaded > bucket );
        
        const uint32 entryCount = bucket < BB_DP_BUCKET_COUNT - 1 ?
                                    maxEntriesPerBucket :
                                    (uint32)( tableEntryCount - maxEntriesPerBucket * ( BB_DP_BUCKET_COUNT - 1 ) ); // Last bucket

        const PairAndMap& buffer = _bucketBuffers[bucket];

        outMapBuffer        = rMapId != FileId::None ? buffer.map : nullptr;
        outPairsBuffer      = buffer.pairs;
        outBucketEntryCount = entryCount;
    }
}

//-----------------------------------------------------------
uint32* DiskPlotPhase2::SortAndStripMap( uint64* map, uint32 entryCount )
{
    // #TODO: Move to shared function now.
    MTJobRunner<StripMapJob> jobs( *_context.threadPool );

    const uint32 threadCount      = _context.threadCount;
    const uint32 entriesPerThread = entryCount / threadCount;

    uint32* outMap = (uint32*)_tmpMap;
    uint32* key    = outMap + entryCount;

    for( uint32 i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];

        job.entryCount = entriesPerThread;
        job.inMap      = map    + entriesPerThread * i;
        job.outKey     = key    + entriesPerThread * i;
        job.outMap     = outMap + entriesPerThread * i;
    }

    const uint32 trailingEntries = entryCount - entriesPerThread * threadCount;
    jobs[threadCount-1].entryCount += trailingEntries;

    jobs.Run( threadCount );

    // Now sort it
    uint32* tmpKey = (uint32*)map;
    uint32* tmpMap = tmpKey + entryCount;
    
    RadixSort256::SortWithKey<BB_MAX_JOBS>( *_context.threadPool, key, tmpKey, outMap, tmpMap, entryCount );

    return outMap;
}

//-----------------------------------------------------------
void MarkJob::Run()
{
    switch( this->table )
    {
        case TableId::Table7: this->MarkEntries<TableId::Table7>(); return;
        case TableId::Table6: this->MarkEntries<TableId::Table6>(); return;
        case TableId::Table5: this->MarkEntries<TableId::Table5>(); return;
        case TableId::Table4: this->MarkEntries<TableId::Table4>(); return;
        case TableId::Table3: this->MarkEntries<TableId::Table3>(); return;

        default:
            ASSERT( 0 );
            return;
    }

    ASSERT( 0 );
}

//-----------------------------------------------------------
template<TableId table>
void MarkJob::MarkEntries()
{
    DiskPlotContext& context = *this->context;
    DiskBufferQueue& queue   = *context.ioQueue;

    const uint32 jobId               = this->JobId();
    const uint32 threadCount         = this->JobCount();
    const uint64 maxEntries          = 1ull << _K;
    const uint32 maxEntriesPerBucket = (uint32)( maxEntries / (uint64)BB_DP_BUCKET_COUNT );
    const uint64 tableEntryCount     = context.entryCounts[(int)table];

    BitField lTableMarkedEntries( this->lTableMarkedEntries );
    BitField rTableMarkedEntries( this->rTableMarkedEntries );

    // Zero-out our portion of the bit field and sync, do this only on the first run
    if( this->pairBucket == 0 && this->pairBucketOffset == 0 )
    {
        const size_t bitFieldSize  = RoundUpToNextBoundary( (size_t)maxEntries / 8, 8 );  // Round up to 64-bit boundary

              size_t sizePerThread = bitFieldSize / threadCount;
        const size_t sizeRemainder = bitFieldSize - sizePerThread * threadCount;

        byte* buffer = ((byte*)this->lTableMarkedEntries) + sizePerThread * jobId;

        if( jobId == threadCount - 1 )
            sizePerThread += sizeRemainder;

        memset( buffer, 0, sizePerThread );
        this->SyncThreads();
    }

    const uint32* map   = this->map;
    Pairs         pairs = this->pairs;
    
    uint32 bucketEntryCount = this->entryCount;

    // Determine how many passes we need to run for this bucket.
    // Passes are determined depending on the range were currently processing
    // on the pairs buffer. Since they have different starting offsets after each
    // L table bucket length that generated its pairs, we need to update that offset
    // after we reach the boundary of the buckets that generated the pairs.
    while( bucketEntryCount )
    {
        uint32 pairBucket           = this->pairBucket;
        uint32 pairBucketOffset     = this->pairBucketOffset;
        uint64 lTableOffset         = this->lTableOffset;

        uint32 pairBucketEntryCount = context.ptrTableBucketCounts[(int)table][pairBucket];

        uint32 passEntryCount       = std::min( pairBucketEntryCount - pairBucketOffset, bucketEntryCount );

        // Prune the table
        {
            // We need a minimum number of entries per thread to ensure that we don't,
            // write to the same qword in the bit field. So let's ensure that each thread
            // has at least more than 2 groups worth of entries.
            // There's an average of 284,190 entries per bucket, which means each group
            // has an about 236.1 entries. We round up to 280 entries.
            // We use minimum 3 groups and round up to 896 entries per thread which gives us
            // 14 QWords worth of area each threads can reference.
            const uint32 minEntriesPerThread = 896;
            
            uint32 threadsToRun     = threadCount;
            uint32 entriesPerThread = passEntryCount / threadsToRun;
            
            while( entriesPerThread < minEntriesPerThread && threadsToRun > 1 )
                entriesPerThread = passEntryCount / --threadsToRun;

            // Only run with as many threads as we have filtered
            if( jobId < threadsToRun )
            {
                const uint32* jobMap   = map; 
                Pairs         jobPairs = pairs;

                jobMap         += entriesPerThread * jobId;
                jobPairs.left  += entriesPerThread * jobId;
                jobPairs.right += entriesPerThread * jobId;

                // Add any trailing entries to the last thread
                // #NOTE: Ensure this is only updated after we get the pairs offset
                uint32 trailingEntries = passEntryCount - entriesPerThread * threadsToRun;
                uint32 lastThreadId    = threadsToRun - 1;
                if( jobId == lastThreadId )
                    entriesPerThread += trailingEntries;

                // Mark entries in 2 steps to ensure the previous thread does NOT
                // write to the same QWord at the same time as the current thread.
                // (May happen when the prev thread writes to the end entries & the 
                //   current thread is writing to its beginning entries)
                const int32 fistStepEntryCount = (int32)( entriesPerThread / 2 );
                int32 i = 0;

                // 1st step
                i = this->MarkStep<table>( i, fistStepEntryCount, lTableMarkedEntries, rTableMarkedEntries, lTableOffset, jobPairs, jobMap );
                this->SyncThreads();

                // 2nd step
                this->MarkStep<table>( i, entriesPerThread, lTableMarkedEntries, rTableMarkedEntries, lTableOffset, jobPairs, jobMap );
            }
            else
            {
                this->SyncThreads();    // Sync for 2nd step
            }

            this->SyncThreads();    // Sync after marking finished
        }


        // Update our position on the pairs table
        bucketEntryCount -= passEntryCount;
        pairBucketOffset += passEntryCount;

        map         += passEntryCount;
        pairs.left  += passEntryCount;
        pairs.right += passEntryCount;

        if( pairBucketOffset < pairBucketEntryCount )
        {
            this->pairBucketOffset = pairBucketOffset;
        }
        else
        {
            // Update our left entry offset by adding the number of entries in the
            // l table bucket index that matches our paid bucket index
            this->lTableOffset += context.bucketCounts[(int)table-1][pairBucket];

            // Move to next pairs bucket
            this->pairBucket ++;
            this->pairBucketOffset = 0;
        }
    }
}

//-----------------------------------------------------------
template<TableId table>
inline int32 MarkJob::MarkStep( int32 i, const int32 entryCount, BitField lTable, const BitField rTable,
                                uint64 lTableOffset, const Pairs& pairs, const uint32* map )
{
    for( ; i < entryCount; i++ )
    {
        if constexpr ( table < TableId::Table7 )
        {
            // #TODO: This map needs to support overflow addresses...
            const uint64 rTableIdx = map[i];
            if( !rTable.Get( rTableIdx ) )
                continue;
        }

        const uint64 left  = lTableOffset + pairs.left [i];
        const uint64 right = left         + pairs.right[i];

        lTable.Set( left  );
        lTable.Set( right );
    }

    return i;
}

