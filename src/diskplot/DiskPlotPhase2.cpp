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
    // We need to reserve 2.5GiB of heap space for the bit arrays
    // used for marking entries. The rest will be used to load
    // the back pointers.
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
    bitFields.fence.Signal();

    // Setup another double buffer for reading
    DoubleBuffer pairBuffers;
    pairBuffers.front = bitFields.back + bitFieldSize;
    pairBuffers.back  = pairBuffers.front + pairBufferSize;
    pairBuffers.fence.Signal();

    for( int tableIdx = (int)TableId::Table2; tableIdx <= (int)TableId::Table7; tableIdx++ )
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

    // Calculate how many passes we have to make to process all the entries
    const uint32 threadCount      = this->JobCount();
    const uint64 maxEntries       = 1ull << _K;
    const uint64 totalEntries     = context.entryCount[(int)this->table];
    ASSERT( totalEntries <= maxEntries );

    const size_t entrySize        = sizeof( uint32 ) + sizeof( uint16 );
    const size_t rTableSize       = totalEntries * entrySize;

    const size_t chunkSize        = this->pairBufferSize;

    uint32 chunkCount             = rTableSize <= chunkSize ? 1u : (uint32)( rTableSize / chunkSize );
    uint64 entriesPerChunk        = chunkSize / entrySize;
    uint64 trailingEntries        = totalEntries - entriesPerChunk * chunkCount;

    uint64 entriesPerThread       = entriesPerChunk / threadCount;

    DoubleBuffer* pairBuffer      = this->pairBuffers;

    // Adjust trailing entrties to add the remainder from chunks
    uint64 chunkRemainders = entriesPerChunk - entriesPerThread * threadCount;
    trailingEntries += chunkRemainders;

    uint32 trailingChunk = 0xFFFFFFFF;
    if( trailingEntries >= entriesPerChunk )
    {
        chunkCount++;
        trailingEntries -= entriesPerChunk;
    }

    if( trailingEntries )
    {
        trailingChunk = chunkCount;
        chunkCount++;
    }

    // #TODO: Need to zero-out our portion of the bit field and sync
    BitField markedEntries( this->bitField );
    
    const FileId lTableId = TableIdToBackPointerFileId( this->table );
    const FileId rTableId = (FileId)((int)lTableId + 1 );

    Pairs pairs;

    uint64 offset = 0;

    // Load the first chunk
    if( this->IsControlThread() )
    {
        const size_t lReadSize = sizeof( uint32 ) * entriesPerChunk;
        const size_t rReadSize = sizeof( uint16 ) * entriesPerChunk;

        pairs.left  = (uint32*)pairBuffer->back;
        pairs.right = (uint16*)( pairs.left + entriesPerChunk );

        queue.ReadFile( lTableId, 0, pairs.left , lReadSize );
        queue.ReadFile( rTableId, 0, pairs.right, rReadSize );
        queue.AddFence( pairBuffers->fence );
        queue.CommitCommands();
    }

    for( uint chunk = 0; chunk < chunkCount; chunk++ )
    {
        // If it is the trailing chunk, adjust our entry count
        if( chunk == trailingChunk )
        {
            // #TODO: Adjust entry count and active threads
        }

        // Ensure the buffer has been loaded & load the next one in the background, if we need to
        if( this->IsControlThread() )
        {
            this->LockThreads();

            pairBuffers->Flip();

            pairs.left  = (uint32*)pairBuffer->back;
            pairs.right = (uint16*)( pairs.left + entriesPerChunk );

            this->pairs = pairs;
            this->ReleaseThreads();

            // Load the next buffer in the background
            const uint32 nextChunk = chunk + 1;

            if( nextChunk < chunkCount )
            {
                // Determine how many entries the next chunk has
                uint32 entriesToLoad = entriesPerChunk;
                if( nextChunk == trailingChunk )
                    entriesToLoad = trailingEntries;

                uint32* bgLeft  = (uint32*)pairBuffer->back;
                uint32* bgRight = bgLeft + entriesToLoad;

                const size_t lReadSize = sizeof( uint32 ) * entriesToLoad;
                const size_t rReadSize = sizeof( uint16 ) * entriesToLoad;
                
                queue.ReadFile( lTableId, 0, bgLeft, lReadSize );
                queue.ReadFile( rTableId, 0, bgRight, rReadSize );
                queue.AddFence( pairBuffers->fence );
                queue.CommitCommands();
            }
        }
        else
        {
            this->WaitForRelease();

            pairs = this->GetJob( 0 ).pairs;
            pairs.left  += entriesPerChunk;
            pairs.right += entriesPerChunk;
        }


        // Start marking entries
        for( uint64 i = offset, end = offset + entriesPerThread; i < entriesPerThread; i++ )
        {

        }

        offset += entriesPerChunk;
    }


    

    uint64 remainingEntries = totalEntries;

    // AutoResetSignal* fence = this->fence;

    while( remainingEntries )
    {
        if( this->IsControlThread() )
        {
            // Grab a buffer to load
        }
    }
}


