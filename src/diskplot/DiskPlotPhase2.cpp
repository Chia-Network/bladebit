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
    AutoResetSignal*  bucketFences;
    byte**            bucketBuffers;
    // Pairs            pairs; // Set by the control thread

    void Run() override;

    template<TableId table>
    void MarkEntries();
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
    // const size_t bitFieldQWordCount = bitFieldSize / sizeof( uint64 );

    const uint32 threadCount      = context.threadCount;
    const size_t fullHeapSize     = context.heapSize + context.ioHeapSize;
    const size_t heapRemainder    = fullHeapSize - bitFieldBuffersSize;
    // const size_t pairBufferSize   = heapRemainder / 2;
    // const size_t entrySize        = sizeof( uint32 ) + sizeof( uint16 );

    // Prepare 2 marking bitfields for dual-buffering
    AutoResetSignal bitFieldFence;
    int             bitFieldIndex = 0;

    byte* bitFields[2];
    bitFields[0] = context.heapBuffer;
    bitFields[1] = bitFields[0] + bitFieldSize;
    // bitFields[2] = bitFields[1] + bitFieldSize;
    

    // Reserve the remainder of the heap for reading R table backpointers
    queue.ResetHeap( heapRemainder, context.heapBuffer + bitFieldBuffersSize );

    // #TODO: Use a single fence with a counter instead.
    AutoResetSignal bucketFences[BB_DP_BUCKET_COUNT];

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
            // job.pairBufferSize      = (uint32)pairBufferSize;
            job.table               = (TableId)tableIdx;

            job.bucketFences     = bucketFences;
            job.bucketBuffers    = nullptr;
        }

        jobs.Run();

        rTableBitField = lTableBitField;
        bitFieldIndex  = ( bitFieldIndex + 1 ) % 2;

        // Ensure the last table finished writing to the bitfield
        if( tableIdx < (int)TableId::Table7 )
            bitFieldFence.Wait();

        // Submit bitfield for writing
        queue.WriteFile( lTableFileId, 0, lTableBitField, bitFieldSize );
        queue.AddFence( bitFieldFence );
        queue.CommitCommands();

        lTableFileId = (FileId)( (int)lTableFileId - 1 );

        const double elapsed = TimerEnd( timer );
        Log::Line( "Finished marking table %d in %.2lf seconds.", tableIdx, elapsed );
    }

    bitFieldFence.Wait();
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
    DiskPlotContext& context      = *this->context;
    DiskBufferQueue& queue        = *context.ioQueue;
    // const TableId    table        = this->table;
    const uint32     jobId        = this->JobId();
    const uint32     threadCount  = this->JobCount();
    const size_t     entrySize    = sizeof( uint32 ) + sizeof( uint16 );
    const uint64     maxEntries   = 1ull << _K;

    // const uint64     totalEntries = context.entryCount[(int)this->table];
    // ASSERT( totalEntries <= maxEntries );

    // DoubleBuffer* pairBuffer = this->pairBuffers;
    BitField      lTableMarkedEntries( this->lTableMarkedEntries );
    BitField      rTableMarkedEntries( this->rTableMarkedEntries );

    byte* bucketBuffers[BB_DP_BUCKET_COUNT];
    uint  bucketsLoaded = 0;

    if( this->IsControlThread() )
        this->bucketBuffers = bucketBuffers;


    {
        // Zero-out our portion of the bit field and sync
        const size_t bitFieldSize  = RoundUpToNextBoundary( (size_t)maxEntries / 8, 8 );  // Round up to 64-bit boundary

              size_t sizePerThread = bitFieldSize / threadCount;
        const size_t sizeRemainder = bitFieldSize - sizePerThread * threadCount;

        byte* buffer = ((byte*)this->lTableMarkedEntries) + sizePerThread * jobId;

        if( jobId == threadCount - 1 )
            sizePerThread += sizeRemainder;

        memset( buffer, 0, sizePerThread );
    }
    
    const FileId lTableId = TableIdToBackPointerFileId( this->table );
    const FileId rTableId = (FileId)((int)lTableId + 1 );

    Pairs pairs;
    uint64 entryOffset = 0;

    // We do this per bucket, because each bucket has an offset associated with it
    for( uint bucket = 0; bucket < BB_DP_BUCKET_COUNT; bucket++ )
    {
        const uint32 bucketEntryCount = context.bucketCounts[(int)table][bucket];
        
        uint32 entriesPerThread = bucketEntryCount / threadCount;

        // Ensure we have a bucket to read from
        if( this->IsControlThread() )
        {
            // Load as many buckets as we can in the background
            while( bucketsLoaded < BB_DP_BUCKET_COUNT )
            {
                const uint32 bucketToLoadEntryCount = context.bucketCounts[(int)table][bucketsLoaded];

                const size_t lReadSize = sizeof( uint32 ) * bucketToLoadEntryCount;
                const size_t rReadSize = sizeof( uint16 ) * bucketToLoadEntryCount;

                byte* lBuffer = queue.GetBuffer( lReadSize + rReadSize, false );
                if( !lBuffer )
                {
                    ASSERT( bucketsLoaded > bucket );
                    break;
                }

                byte* rBuffer = lBuffer + lReadSize;
                bucketBuffers[bucketsLoaded] = (byte*)lBuffer;  // Store the buffer for the other threads to use

                queue.ReadFile( lTableId, 0, lBuffer, lReadSize );
                queue.ReadFile( rTableId, 0, rBuffer, rReadSize );
                
                // #TODO: Use a single fence with a counter. For now use multiple
                queue.AddFence( this->bucketFences[bucketsLoaded] );

                queue.CommitCommands();
                bucketsLoaded ++;
            }

            // Ensure the current bucket has been loaded already
            this->LockThreads();
            this->bucketFences[bucket].Wait();
            this->ReleaseThreads();
        }
        else
        {
            this->WaitForRelease();
        }

        pairs.left  = (uint32*)this->GetJob( 0 ).bucketBuffers[bucket];
        pairs.right = (uint16*)( pairs.left + bucketEntryCount );

        pairs.left  += entriesPerThread * jobId;
        pairs.right += entriesPerThread * jobId;

        // Add any trailing entries to the last thread
        // #NOTE: Ensure this is only updated after we get the pairs offset
        if( this->JobId() == threadCount - 1 )
        {
            const uint32 trailingEntries = bucketEntryCount - entriesPerThread * threadCount;
            entriesPerThread += trailingEntries;
        }

        //
        // Mark used entries on the left table, given the right table's back pointers
        //

        // Divide this into 2 steps so that we ensure the previous thread does NOT
        // write to the same field at the same time as the current thread.
        // (May happen when the prev thread writes to the end entries & the 
        //   current thread is writing to its beginning entries)
        const int64 firstPassCount = (int64)entriesPerThread - kBC; 
        
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
        // Move on to the next bucket
        //
        entryOffset += bucketEntryCount;

        // Release the bucket buffer
        if( this->IsControlThread() )
        {
            queue.ReleaseBuffer( bucketBuffers[bucket] );
            queue.CommitCommands();
        }
    }
}


