#include "DiskPlotPhase2.h"
#include "util/BitField.h"

struct MarkJob : MTJob<MarkJob>
{
    DiskPlotContext* context;
    uint64*          bitField;
    DoubleBuffer*    pairBuffers;
    size_t           pairBufferSize;
    TableId          table;

    Pairs            pairs; // Set by the control thread

    void Run() override;
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
    // const size_t bitFieldQWordCount = bitFieldSize / sizeof( uint64 );

    const uint32 threadCount      = context.threadCount;
    const size_t fullHeapSize     = context.heapSize + context.ioHeapSize;
    const size_t heapRemainder    = fullHeapSize - bitFieldSize * 2;
    const size_t pairBufferSize   = heapRemainder / 2;
    const size_t entrySize        = sizeof( uint32 ) + sizeof( uint16 );

    // Prepare 2 marking bitfields for dual-buffering
    DoubleBuffer bitFields;
    bitFields.front = context.heapBuffer;
    bitFields.back  = context.heapBuffer + bitFieldSize;
    // bitFields.fence.Signal();

    // Setup another double buffer for reading
    DoubleBuffer pairBuffers;
    pairBuffers.front = bitFields.back + bitFieldSize * 2;
    pairBuffers.back  = pairBuffers.front + pairBufferSize;
    // pairBuffers.fence.Signal();

    for( int tableIdx = (int)TableId::Table7; tableIdx >= (int)TableId::Table2; tableIdx-- )
    {
        MTJobRunner<MarkJob> jobs( *context.threadPool );

        for( uint i = 0; i < threadCount; i++ )
        {
            MarkJob& job = jobs[i];

            job.context          = &context;
            job.bitField         = (uint64*)bitFields.front;
            job.pairBuffers      = &pairBuffers;
            job.pairBufferSize   = (uint32)pairBufferSize;
            job.table            = (TableId)tableIdx;
            job.pairs.left       = nullptr;
            job.pairs.right      = nullptr;
        }

        jobs.Run();
    }
}


//-----------------------------------------------------------
void MarkJob::Run()
{
    DiskPlotContext& context      = *this->context;
    DiskBufferQueue& queue        = *context.ioQueue;
    const TableId    table        = this->table;
    const uint32     jobId        = this->JobId();
    const uint32     threadCount  = this->JobCount();
    const size_t     entrySize    = sizeof( uint32 ) + sizeof( uint16 );
    const uint64     maxEntries   = 1ull << _K;

    // const uint64     totalEntries = context.entryCount[(int)this->table];
    // ASSERT( totalEntries <= maxEntries );


    DoubleBuffer* pairBuffer = this->pairBuffers;
    BitField      markedEntries( this->bitField );

    {
        // Zero-out our portion of the bit field and sync
        const size_t bitFieldSize = RoundUpToNextBoundary( (size_t)maxEntries / 8, 8 );  // Round up to 64-bit boundary

              size_t sizePerThread = bitFieldSize / threadCount;
        const size_t sizeRemainder = bitFieldSize - sizePerThread * threadCount;

        byte* buffer = ((byte*)this->bitField) + sizePerThread * jobId;

        if( jobId == threadCount - 1 )
            sizePerThread += sizeRemainder;

        memset( buffer, 0, sizePerThread );
    }
    
    const FileId lTableId = TableIdToBackPointerFileId( this->table );
    const FileId rTableId = (FileId)((int)lTableId + 1 );

    Pairs pairs;
    uint64 entryOffset = 0;

    // Load the first chunk
    if( this->IsControlThread() )
    {
        const uint32 bucketEntryCount = context.bucketCounts[(int)table][0];

        const size_t lReadSize = sizeof( uint32 ) * bucketEntryCount;
        const size_t rReadSize = sizeof( uint16 ) * bucketEntryCount;

        uint32* lBuffer = (uint32*)pairBuffer->back;
        uint32* rBuffer = lBuffer + bucketEntryCount;

        queue.ReadFile( lTableId, 0, lBuffer , lReadSize );
        queue.ReadFile( rTableId, 0, rBuffer, rReadSize );
        queue.AddFence( pairBuffers->fence );
        queue.CommitCommands();
    }

    // We do this per bucket, because each bucket has an offset associated with it
    for( uint bucket = 0; bucket < BB_DP_BUCKET_COUNT; bucket++ )
    {
        const uint32 bucketEntryCount = context.bucketCounts[(int)table][bucket];
        
        uint32 entriesPerThread = bucketEntryCount / threadCount;

        // Ensure the bucket has finished loading
        if( this->IsControlThread() )
        {
            this->LockThreads();

            pairBuffers->Flip();

            // Assign the buffer as pairs
            pairs.left  = (uint32*)pairBuffer->front;
            pairs.right = (uint16*)(pairs.left + bucketEntryCount);
            
            this->pairs = pairs;

            this->ReleaseThreads();

            // Load next bucket if we need to
            const uint32 nextBucket = bucket + 1;

            if( nextBucket < BB_DP_BUCKET_COUNT )
            {
                const uint32 nextBucketEntryCount = context.bucketCounts[(int)table][nextBucket];

                // #TODO: Change this to load as many buckets as we can in the background,
                //        as long as there is heaps space for it.
                const size_t lReadSize = sizeof( uint32 ) * nextBucketEntryCount;
                const size_t rReadSize = sizeof( uint16 ) * nextBucketEntryCount;

                uint32* lBuffer = (uint32*)pairBuffer->back;
                uint32* rBuffer = lBuffer + nextBucketEntryCount;

                queue.ReadFile( lTableId, 0, lBuffer, lReadSize );
                queue.ReadFile( rTableId, 0, rBuffer, rReadSize );
                queue.AddFence( pairBuffers->fence );
                queue.CommitCommands();
            }
        }
        else
        {
            this->WaitForRelease();

            pairs = this->GetJob( 0 ).pairs;
            pairs.left  += entriesPerThread;
            pairs.right += entriesPerThread;
        }

        // Add any trailing entries to the last thread
        // #NOTE: Ensure this is only updated after we get the pairs offset
        if( this->JobId() == threadCount - 1 )
        {
            const uint32 trailingEntries = bucketEntryCount - entriesPerThread * threadCount;
            entriesPerThread += trailingEntries;
        }

        // Mark used entries on the left table, given the right table back pointers
        for( int64 i = 0; i < (int64)entriesPerThread; i++ )
        {
            // #TODO: Unroll this a bit if beneficial
            const uint64 left  = pairs.left [i] + entryOffset;
            const uint64 right = left + pairs.right[i];

            markedEntries.Set( left  );
            markedEntries.Set( right );

            // #TODO: Divide this into 2 steps so that we ensure the previous thread does NOT
            //        write to the same field at the same time as the current thread.
            //        (May happen when the prev thread writes to the end entries & the current thread is writing to
            //          its beginning entries)
        }

        entryOffset += bucketEntryCount;
    }
}


