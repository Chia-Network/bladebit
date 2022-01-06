#include "DiskPlotPhase2.h"
#include "util/BitField.h"

struct MarkJob : MTJob<MarkJob>
{
    DiskPlotContext* context;
    uint64*          lTableMarkedEntries;
    uint64*          rTableMarkedEntries;
    // DoubleBuffer*    pairBuffers;
    // size_t           pairBufferSize;
    TableId          table;

    // Set by the control thread
    Fence*           bucketFence;
    byte**           bucketBuffers;

    void Run() override;

    template<TableId table>
    void MarkEntries();

    void SortLookupMap( uint64* unsortedMap, const uint32 entryCount );

    enum FenceId
    {
        None = 0,
        MapLoaded,
        PairsLoaded,

        FenceCount
    };
};

//-----------------------------------------------------------
DiskPlotPhase2::DiskPlotPhase2( DiskPlotContext& context )
    : _context( context )
{

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

    // Seek the table files back to the beginning
    for( int tableIdx = (int)TableId::Table2; tableIdx <= (int)TableId::Table7; tableIdx++ )
    {
        const FileId leftId  = TableIdToBackPointerFileId( (TableId)tableIdx );
        const FileId rightId = (FileId)( (int)leftId + 1 );
        
        queue.SeekFile( leftId , 0, 0, SeekOrigin::Begin );
        queue.SeekFile( rightId, 0, 0, SeekOrigin::Begin );
    }
    queue.CommitCommands();

    // Determine what size we can use to load
    const uint64 maxEntries       = 1ull << _K;
    const size_t bitFieldSize     = RoundUpToNextBoundary( (size_t)maxEntries / 8, 8 );  // Round up to 64-bit boundary
    const size_t bitFieldBuffersSize = bitFieldSize * 2;

    const uint32 threadCount      = 1;// context.threadCount;
    const size_t fullHeapSize     = context.heapSize + context.ioHeapSize;
    const size_t heapRemainder    = fullHeapSize - bitFieldBuffersSize;

    // Prepare 2 marking bitfields for dual-buffering
    Fence bitFieldFence;
    int   bitFieldIndex = 0;

    byte* bitFields[2];
    bitFields[0] =  context.heapBuffer;
    bitFields[1] =  bitFields[0] + bitFieldSize;
    // bbvirtalloc<byte>( 1ull << _K );
    // bbvirtalloc<byte>( 1ull << _K );

    // Reserve the remainder of the heap for reading R table backpointers
    queue.ResetHeap( heapRemainder, context.heapBuffer + bitFieldBuffersSize );

    Fence bucketFence;

    uint64* rTableBitField = nullptr;

    FileId  lTableFileId   = FileId::MARKED_ENTRIES_6;

    // Mark all tables
    for( int tableIdx = (int)TableId::Table7; tableIdx > (int)TableId::Table2; tableIdx-- )
    {
        const auto timer = TimerBegin();

        MTJobRunner<MarkJob> jobs( *context.threadPool );

        uint64* lTableBitField = (uint64*)( bitFields[bitFieldIndex] );

        for( uint i = 0; i < threadCount; i++ )
        {
            MarkJob& job = jobs[i];

            job.context             = &context;
            job.lTableMarkedEntries = lTableBitField;
            job.rTableMarkedEntries = rTableBitField;
            job.table               = (TableId)tableIdx;

            job.bucketFence         = &bucketFence;
            job.bucketBuffers       = nullptr;
        }

        //
        // #TEST
        //
        // if( 0 )
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

        jobs.Run( threadCount );

        rTableBitField = lTableBitField;
        bitFieldIndex  = ( bitFieldIndex + 1 ) % 2;

        // Ensure the last table finished writing to the bitfield
        if( tableIdx < (int)TableId::Table7 )
            bitFieldFence.Wait();

        // Submit bitfield for writing
        queue.WriteFile( lTableFileId, 0, lTableBitField, bitFieldSize );
        queue.SignalFence( bitFieldFence );
        queue.CommitCommands();

        lTableFileId = (FileId)( (int)lTableFileId - 1 );

        const double elapsed = TimerEnd( timer );
        Log::Line( "Finished marking table %d in %.2lf seconds.", tableIdx, elapsed );

        // #TEST:
        // if( 0 )
        {
            BitField markedEntries( lTableBitField );
            uint64 lTableEntries = context.entryCounts[(int)tableIdx-1];

            uint64 bucketsTotalCount = 0;
            for( uint64 e = 0; e < BB_DP_BUCKET_COUNT; ++e )
                bucketsTotalCount += context.ptrTableBucketCounts[(int)tableIdx-1][e];

            ASSERT( bucketsTotalCount == lTableEntries );

            uint64 lTablePrunedEntries = 0;

            for( uint64 e = 0; e < lTableEntries; ++e )
            {
                if( markedEntries.Get( e ) )
                    lTablePrunedEntries++;
            }

            Log::Line( "Table %u entries: %llu/%llu (%.2lf%%)", tableIdx,
                       lTablePrunedEntries, lTableEntries, ((double)lTablePrunedEntries / lTableEntries ) * 100.0 );

        }
    }

    bitFieldFence.Wait();
    queue.CompletePendingReleases();
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
    const size_t entrySize           = sizeof( uint32 ) + sizeof( uint16 );
    const uint64 maxEntries          = 1ull << _K;
    const uint32 maxEntriesPerBucket = (uint32)( maxEntries / (uint64)BB_DP_BUCKET_COUNT );
    const uint64 tableEntryCount     = context.entryCounts[(int)table];

    BitField lTableMarkedEntries( this->lTableMarkedEntries );
    BitField rTableMarkedEntries( this->rTableMarkedEntries );

    uint32 bucketsLoaded     = 0;
    uint32 currentPairBucket = 0;   // For tracking the current offset needed to use with the pairs
    uint64 entryOffset       = 0;   // Entry offset given the current pair bucket region
    uint64 mapOffset         = 0;

    byte* bucketBuffers[BB_DP_BUCKET_COUNT];    // Contains pointers to all the buckets that we've loaded so far.


    if( this->IsControlThread() )
        this->bucketBuffers = bucketBuffers;

    // Zero-out our portion of the bit field and sync (we sync when grabbing the buckets)
    {
        const size_t bitFieldSize  = RoundUpToNextBoundary( (size_t)maxEntries / 8, 8 );  // Round up to 64-bit boundary

              size_t sizePerThread = bitFieldSize / threadCount;
        const size_t sizeRemainder = bitFieldSize - sizePerThread * threadCount;

        byte* buffer = ((byte*)this->lTableMarkedEntries) + sizePerThread * jobId;

        if( jobId == threadCount - 1 )
            sizePerThread += sizeRemainder;

        memset( buffer, 0, sizePerThread );
    }

    Pairs   pairs;
    uint32* map;

    // We do this per bucket, because each bucket has an offset associated with it
    for( uint bucket = 0; bucket < BB_DP_BUCKET_COUNT; bucket++ )
    {
        const uint32 bucketEntryCount = bucket < BB_DP_BUCKET_COUNT - 1 ?
                                        maxEntriesPerBucket :
                                        (uint32)( tableEntryCount - maxEntriesPerBucket * ( BB_DP_BUCKET_COUNT - 1 ) ); // Last bucket

        // Ensure we have a bucket loaded from which to read
        if( this->IsControlThread() )
        {
            // Load as many buckets as we can in the background
            while( bucketsLoaded < BB_DP_BUCKET_COUNT )
            {
                const uint32 bucketToLoadEntryCount = bucketsLoaded < BB_DP_BUCKET_COUNT - 1 ?
                                                      maxEntriesPerBucket :
                                                      (uint32)( tableEntryCount - maxEntriesPerBucket * ( BB_DP_BUCKET_COUNT - 1 ) ); // Last bucket

                // Reserve a buffer to load both a map bucket 
                // and the same amount of entries worth of pairs.
                const size_t mapReadSize = sizeof( uint64 ) * bucketToLoadEntryCount;
                const size_t lReadSize   = sizeof( uint32 ) * bucketToLoadEntryCount;
                const size_t rReadSize   = sizeof( uint16 ) * bucketToLoadEntryCount;

                byte* buffer = queue.GetBuffer( mapReadSize + lReadSize + rReadSize, false );
                if( !buffer )
                {   
                    if( bucketsLoaded <= bucket )
                    {
                        // Force-load a bucket (block until we can load one)
                        buffer = queue.GetBuffer( lReadSize + rReadSize, true );
                        ASSERT( buffer );
                    }
                    else
                        break;
                }

                byte* pairsLBuffer = buffer       + mapReadSize; 
                byte* pairsRBuffer = pairsLBuffer + lReadSize;

                bucketBuffers[bucketsLoaded] = buffer;  // Store the buffer for the other threads to use

                const FileId rMapId       = TableIdToMapFileId( table );
                const FileId rTableLPtrId = TableIdToBackPointerFileId( table );
                const FileId rTableRPtrId = (FileId)((int)rTableLPtrId + 1 );

                const uint32 loadFenceId = bucketsLoaded++ * FenceCount;

                queue.ReadFile( rMapId, 0, buffer, mapReadSize );
                queue.SignalFence( *this->bucketFence, MapLoaded + loadFenceId );

                queue.ReadFile( rTableLPtrId, 0, pairsLBuffer, lReadSize   );
                queue.ReadFile( rTableRPtrId, 0, pairsRBuffer, rReadSize   );
                queue.SignalFence( *this->bucketFence, PairsLoaded + loadFenceId );

                queue.CommitCommands();
            }

            {
                const uint32 waitFenceId = bucket * FenceCount;

                // #TODO: Allow wait to suspend other threads
                this->LockThreads();
                
                // Wait for the map to finish loading
                this->bucketFence->Wait( MapLoaded + waitFenceId );
                this->ReleaseThreads();

                // Sort the map on its origin index
                auto* lookupMapUnsorted = (uint64*)this->GetJob( 0 ).bucketBuffers[bucket];
                this->SortLookupMap( lookupMapUnsorted, bucketEntryCount );

                // Wait for the pairs to finish loading
                this->LockThreads();
                this->bucketFence->Wait( PairsLoaded + waitFenceId );
                this->ReleaseThreads();
            }
        }
        else
        {
            // Wait for the map to finish loading
            this->WaitForRelease();

            // Sort the map on its origin index
            auto* lookupMapUnsorted = (uint64*)this->GetJob( 0 ).bucketBuffers[bucket];
            this->SortLookupMap( lookupMapUnsorted, bucketEntryCount );

            // Wait for the pairs to finish loading
            this->WaitForRelease();
        }


        // Bucket has been loaded, we can now use it
        uint32 entriesPerThread = bucketEntryCount / threadCount;

        {
            uint64* mapBuffer = this->GetJob( 0 ).bucketBuffers[bucket];

            pairs.left  = (uint32*)( mapBuffer  + bucketEntryCount ); 
            pairs.right = (uint16*)( pairs.left + bucketEntryCount );

            pairs.left  += entriesPerThread * jobId;
            pairs.right += entriesPerThread * jobId;
        }

        // Add any trailing entries to the last thread
        // #NOTE: Ensure this is only updated after we get the pairs offset
        if( this->JobId() == threadCount - 1 )
        {
            const uint32 trailingEntries = bucketEntryCount - entriesPerThread * threadCount;
            entriesPerThread += trailingEntries;
        }

        // Determine how many passes we need to run for this bucket.
        // Passes are determined depending on the range were currently processing
        // on the pairs buffer. Since they have different starting offsets after each
        // L table bucket length that generated its pairs, we need to update that offset
        // after we reach the boundary of the buckets that generated the pairs.
        uint32 passCount = 0;

        //
        // Mark used entries on the left table, given the right table's back pointers
        //

        // Divide this into 2 steps so that we ensure the previous thread does NOT
        // write to the same field at the same time as the current thread.
        // (May happen when the prev thread writes to the end entries & the 
        //   current thread is writing to its beginning entries)
        const int64 firstPassCount = (int64)entriesPerThread / 2;

        const int64 rTableOffset = (int64)entryOffset + entriesPerThread * jobId;
        int64 i;

        // First pass
        for( i = 0; i < firstPassCount; i++ )
        {
            if constexpr ( table < TableId::Table7 )
            {
                const uint64 rTableIdx = rTableOffset + (uint64)i;
                if( !rTableMarkedEntries.Get( rTableIdx ) )
                    continue;
            }

            const uint64 left  = pairs.left[i] + entryOffset;
            const uint64 right = left + pairs.right[i];

            lTableMarkedEntries.Set( left  );
            lTableMarkedEntries.Set( right );
        }

        this->SyncThreads();

        // Second pass
        for( ; i < (int64)entriesPerThread; i++ )
        {
            if constexpr ( table < TableId::Table7 )
            {
                const uint64 rTableIdx = rTableOffset + (uint64)i;
                if( !rTableMarkedEntries.Get( rTableIdx ) )
                    continue;
            }

            const uint64 left  = pairs.left[i] + entryOffset;
            const uint64 right = left + pairs.right[i];

            lTableMarkedEntries.Set( left  );
            lTableMarkedEntries.Set( right );
        }

        //
        // Move on to the next bucket,
        // updating our offset by the entry count of the L table
        // bucket which generated our R table entries.
        //
        entryOffset += context.bucketCounts[(int)table-1][bucket];

        // Release the bucket buffer
        if( this->IsControlThread() )
        {
            queue.ReleaseBuffer( bucketBuffers[bucket] );
            queue.CommitCommands();
        }
    }
}


