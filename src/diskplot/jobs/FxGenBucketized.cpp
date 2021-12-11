#include "FxGenBucketized.h"
#include "threading/ThreadPool.h"
#include "plotshared/MTJob.h"
#include "diskplot/DiskBufferQueue.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"

struct FxBucketJob : MTJob<FxBucketJob>
{
    // Inputs
    uint32        bucketIdx;
    uint32        entryCount;
    uint32        sortKeyOffset;        // Count at which we start counting a sort key.
    uint32        fileBlockSize;        // For direct I/O
    TableId       table;

    Pairs         pairs;
    const uint32* yIn;
    const uint64* metaInA;
    const uint64* metaInB;
    const uint32* counts;               // Each thread's entry count per bucket

    // Temp/working space
    uint32*       yTmp;
    uint64*       metaTmpA;
    uint64*       metaTmpB;

    // Outputs, if not disk-based
    uint32*       yOut;
    uint32*       sortKeyOut;
    uint64*       metaOutA;
    uint64*       metaOutB;
    byte*         bucketIdOut;

    uint32*       totalBucketCounts;    // Total counts per for all buckets. Used by the control thread

    // For chunked jobs:
    uint32           entriesPerChunk;
    uint32           chunkCount;
    uint32           trailingChunkEntries;
    uint32           entriesOffset;         // How many entries to move forward our pointer for the next chunk
    DiskBufferQueue* queue;
    bool             writeToDisk;
    uint32           bufferSizeY;
    uint32           bufferSizeMetaA;
    uint32           bufferSizeMetaB;

    void Run() override;

    template<TableId tableId, typename TMetaA, typename TMetaB>
    void DistributeIntoBuckets(
        const uint32  entryCount,
        const uint32  sortKeyOffset,
        const byte*   bucketIndices,
        const uint32* y,                // Unsorted table data
        const TMetaA* metaA,
        const TMetaB* metaB,
        uint32*       yBuckets,         // Output buckets
        uint32*       sortKey,          // Reverse lookup index/sort key to map where the pairs/back pointers are supposed to go, without sorting them.
        TMetaA*       metaABuckets,
        TMetaB*       metaBBuckets,
        uint32        counts         [BB_DP_BUCKET_COUNT],  // Current job's count per bucket
        uint32        outBucketCounts[BB_DP_BUCKET_COUNT],  // Entry count per bucket (across all threads)
        const size_t  fileBlockSize
    );

    void CalculatePrefixSum( 
        uint32       counts      [BB_DP_BUCKET_COUNT],  // Entry count for this thread
        uint32       pfxSum      [BB_DP_BUCKET_COUNT],  // Prefix sum for this thread
        uint32       bucketCounts[BB_DP_BUCKET_COUNT],  // Entries per bucket, across all threads
        const size_t fileBlockSize                      // For aligning data when direct IO is enabled
    );

    template<TableId tableId, bool WriteToDisk>
    void RunForTable();
};

//-----------------------------------------------------------
template<TableId tableId>
void GenFxBucketizedChunked(
    DiskBufferQueue* diskQueue,
    size_t           chunkSize,
    bool             directIO,

    ThreadPool&      threadPool,
    uint             threadCount,
    
    uint             bucketIdx,        // Inputs
    uint             entryCount, 
    uint             sortKeyOffset, 
    Pairs            pairs,
    const uint32*    yIn,
    const uint64*    metaInA, 
    const uint64*    metaInB,

    byte*            bucketIdOut,     // Tmp
    uint32*          yTmp,
    uint64*          metaTmpA,
    uint64*          metaTmpB,

    uint32*          yOut,             // Outputs
    uint64*          metaOutA,
    uint64*          metaOutB,

    uint32           bucketCounts[BB_DP_BUCKET_COUNT] );


//-----------------------------------------------------------
template<TableId table>
void FxGenBucketized<table>::GenerateFxBucketizedToDisk(
    DiskBufferQueue& diskQueue,
    size_t           writeInterval,

    ThreadPool&      pool, 
    uint32           threadCount,

    const uint32     bucketIdx,
    const uint32     entryCount,
    const uint32     sortKeyOffset,
    Pairs            pairs,
    byte*            bucketIndices,

    const uint32*    yIn,
    const uint64*    metaAIn,
    const uint64*    metaBIn,

    uint32*          yTmp,
    uint64*          metaATmp,
    uint64*          metaBTmp,

    uint32           bucketCounts[BB_DP_BUCKET_COUNT]
)
{
    GenFxBucketizedChunked<table>(
        &diskQueue, 
        writeInterval, 
        diskQueue.UseDirectIO(),
        
        pool, 
        threadCount,

        bucketIdx,
        entryCount,
        sortKeyOffset,
        pairs,
        yIn,
        metaAIn,
        metaBIn,

        bucketIndices,
        yTmp,
        metaATmp,
        metaBTmp,
        
        nullptr,            // These will be obtained from the disk queue
        nullptr,
        nullptr,

        bucketCounts
    );
}

//-----------------------------------------------------------
// template<TableId table>
// void FxGenBucketized<table>::GenerateFxBucketizedInMemory(
//     ThreadPool&      pool, 
//     uint32           threadCount,
//     uint32           bucketIdx,
//     const uint32     entryCount,
//     Pairs            pairs,
//     byte*            bucketIndices,

//     const uint32*    yIn,
//     const uint64*    metaAIn,
//     const uint64*    metaBIn,

//     uint32*          yTmp,
//     uint32*          metaATmp,
//     uint32*          metaBTmp,

//     uint32*          yOut,
//     uint32*          metaAOut,
//     uint32*          metaBOut,

//     uint32           bucketCounts[BB_DP_BUCKET_COUNT]
// )
// {
//     GenFxBucketizedChunked<table>(
//         nullptr, 0, false, // Chunk size of 0 means unchunked
//         pool, threadCount,
        
//         bucketIdx,
//         entryCount,
//         pairs,
        
//         yIn,
//         metaAIn,
//         metaBIn,

//         bucketIndices,
//         yTmp,
//         metaATmp,
//         metaBTmp,
        
//         yOut,
//         metaAOut,
//         metaBOut,

//         bucketCounts
//     );
// }


//-----------------------------------------------------------
template<TableId tableId>
void GenFxBucketizedChunked(
    DiskBufferQueue* diskQueue,
    size_t           chunkSize,
    bool             directIO,

    ThreadPool&      threadPool,
    uint             threadCount,
    
    uint             bucketIdx,        // Inputs
    uint             entryCount, 
    uint             sortKeyOffset, 
    Pairs            pairs,
    const uint32*    yIn,
    const uint64*    metaInA, 
    const uint64*    metaInB,

    byte*            bucketIdOut,     // Tmp
    uint32*          yTmp,
    uint64*          metaTmpA,
    uint64*          metaTmpB,

    uint32*          yOut,             // Outputs
    uint64*          metaOutA,
    uint64*          metaOutB,

    uint32           bucketCounts[BB_DP_BUCKET_COUNT] )
{
    const size_t outMetaSizeA = TableMetaOut<tableId>::SizeA;
    const size_t outMetaSizeB = TableMetaOut<tableId>::SizeB;

    // Size: Y + sortKey + metadata (A + B)
    const size_t sizePerEntry = sizeof( uint32 ) * 2 + outMetaSizeA + outMetaSizeB;

    // We need to adjust the chunk buffer size to leave some space for us to be
    // able to align each bucket start pointer to the block size of the output device
    const size_t fileBlockSize         = diskQueue ? 0ull : diskQueue->BlockSize();
    const size_t bucketBlockAlignSize  = fileBlockSize * BB_DP_BUCKET_COUNT;
    const size_t usableChunkSize       = directIO == false ? chunkSize : chunkSize - bucketBlockAlignSize * 2;

    const uint32 entriesPerChunk       = chunkSize == 0 ? entryCount : (uint32)( usableChunkSize / sizePerEntry );

    const uint32 sizeYBuffer           = (uint32)( sizeof( uint32 ) * entriesPerChunk );
    const uint32 sizeMetaABuffer       = (uint32)( outMetaSizeA * entriesPerChunk );
    const uint32 sizeMetaBBuffer       = (uint32)( outMetaSizeB * entriesPerChunk );

    uint32 chunkCount                  = entryCount / entriesPerChunk;
    uint32 chunkTrailingEntries        = entryCount - entriesPerChunk * chunkCount;
    uint32 entriesPerThread            = entriesPerChunk / threadCount;

    ASSERT( entriesPerThread > 0 );

    // If the leftover entries per chunk add up to a full chunk, then count it
    while( chunkTrailingEntries >= entriesPerChunk )
    {
        chunkCount++;
        chunkTrailingEntries -= entriesPerChunk;
    }

    // Left-over entries per thread, per chunk can be spread out
    // between threads since we are guaranteed they still fit in a chunk,
    // and are less than thread count
    uint32 threadTrailingEntries = entriesPerChunk - entriesPerThread * threadCount;
    uint32 entriesOffset         = 0;

    MTJobRunner<FxBucketJob> jobs( threadPool );

    for( uint i = 0; i < threadCount; i++ )
    {
        FxBucketJob& job = jobs[i];

        job.queue         = diskQueue;
        job.bucketIdx     = bucketIdx;
        job.entryCount    = entriesPerThread;
        job.sortKeyOffset = sortKeyOffset;
        job.fileBlockSize = (uint32)fileBlockSize;
        job.table         = tableId;

        job.pairs         = pairs;
        job.yIn           = yIn;
        job.metaInA       = metaInA;
        job.metaInB       = metaInB;
        job.counts        = nullptr;

        job.yTmp          = yTmp;
        job.metaTmpA      = metaTmpA;
        job.metaTmpB      = metaTmpB;

        job.yOut          = yOut;
        job.metaOutA      = metaOutA;
        job.metaOutB      = metaOutB;
        job.bucketIdOut   = bucketIdOut;

        job.totalBucketCounts    = bucketCounts;

        // job.chunkSize            = usableChunkSize;
        job.writeToDisk          = true;
        job.chunkCount           = chunkCount;
        job.entriesPerChunk      = entriesPerChunk;
        job.trailingChunkEntries = chunkTrailingEntries;
        job.bufferSizeY          = sizeYBuffer;
        job.bufferSizeMetaA      = sizeMetaABuffer;
        job.bufferSizeMetaB      = sizeMetaBBuffer;

        if( threadTrailingEntries )
        {
            job.entryCount ++;
            threadTrailingEntries --;
        }

        job.entriesOffset = entriesOffset;
        entriesOffset += job.entryCount;
        sortKeyOffset += job.entryCount;

        pairs.left  += job.entryCount;
        pairs.right += job.entryCount;

        bucketIdOut += job.entryCount;

        metaTmpA    += job.entryCount;
        yTmp        += job.entryCount;

        if constexpr ( outMetaSizeB > 0 )
            metaTmpB += job.entryCount;
    }

    jobs.Run( threadCount );
}

//-----------------------------------------------------------
void FxBucketJob::Run()
{
    switch( this->table )
    {
        case TableId::Table1: 
            this->writeToDisk ? RunForTable<TableId::Table1, true>() : RunForTable<TableId::Table1, false>(); 
            return;
        case TableId::Table2: 
            this->writeToDisk ? RunForTable<TableId::Table2, true>() : RunForTable<TableId::Table2, false>(); 
            return;
        case TableId::Table3: 
            this->writeToDisk ? RunForTable<TableId::Table3, true>() : RunForTable<TableId::Table3, false>(); 
            return;
        case TableId::Table4: 
            this->writeToDisk ? RunForTable<TableId::Table4, true>() : RunForTable<TableId::Table4, false>(); 
            return;
        case TableId::Table5: 
            this->writeToDisk ? RunForTable<TableId::Table5, true>() : RunForTable<TableId::Table5, false>(); 
            return;
        case TableId::Table6: 
            this->writeToDisk ? RunForTable<TableId::Table6, true>() : RunForTable<TableId::Table6, false>(); 
            return;
        case TableId::Table7: 
            this->writeToDisk ? RunForTable<TableId::Table7, true>() : RunForTable<TableId::Table7, false>(); 
            return;
        
        case TableId::_Count:
        default:
            ASSERT( 0 );
        break;
    }
}

//-----------------------------------------------------------
template<TableId tableId, bool WriteToDisk>
void FxBucketJob::RunForTable()
{
    using TMetaA = typename TableMetaOut<tableId>::MetaA;
    using TMetaB = typename TableMetaOut<tableId>::MetaB;

    const size_t SizeMetaA = TableMetaOut<tableId>::SizeA;
    const size_t SizeMetaB = TableMetaOut<tableId>::SizeB;

    DiskBufferQueue* diskQueue    = this->queue;

    uint32        chunkCount      = this->chunkCount;
    uint32        entryCount      = this->entryCount;
    uint32        sortKeyOffset   = this->sortKeyOffset;
    const uint64  bucket          = ((uint64)this->bucketIdx) << 32;
    const size_t  fileBlockSize   = this->fileBlockSize;
    const uint32  entriesPerChunk = this->entriesPerChunk;

    Pairs         pairs           = this->pairs;
    const uint32* yIn             = this->yIn;
    const uint64* metaInA         = this->metaInA;
    const uint64* metaInB         = this->metaInB;

    // Temporary buffers used when we're calculating FX, but before distribution into buckets.
    uint32* yTmp          = this->yTmp ;
    uint64* metaATmp      = this->metaTmpA;
    uint64* metaBTmp      = this->metaTmpB;
    byte*   bucketIndices = this->bucketIdOut;

    // If writing to mmemory, these are not null.
    // When writing to disk, we get a buffer from the queue.
    // #TODO: We should be able to ping-pong the in/out buffers
    //        instead when processing in-memory.
    uint32* yOut          = this->yOut;
    uint64* metaOutA      = this->metaOutA;
    uint64* metaOutB      = this->metaOutB;

    uint32* sizes         = nullptr;
    uint32* sortKey       = nullptr;
    uint32* metaASizes    = nullptr;
    uint32* metaBSizes    = nullptr;


    // #TODO: Pass these as job input
    const bool isEven        = static_cast<uint>( tableId ) & 1;

    const FileId yFileId       = isEven ? FileId::Y1       : FileId::Y0;
    const FileId metaAFileId   = isEven ? FileId::META_A_0 : FileId::META_A_1;
    const FileId metaBFileId   = isEven ? FileId::META_B_0 : FileId::META_B_1;
    const FileId sortKeyFileId = TableIdToSortKeyId( tableId );


    uint counts      [BB_DP_BUCKET_COUNT];  // How many entries we have per bucket for this thread only.
                                            // Has to be defined out here instead of inside DistributeIntoBuckets()
                                            // because we need its scope to live on when another thread exists that function.

    uint bucketCounts[BB_DP_BUCKET_COUNT];  // Count how many entries we have per bucket, across all threads.
                                            // This is only used by the main thread

    // If there's a left-over chunk, account for it here
    uint32 trailingChunk = 0xFFFFFFFF;
    if( this->trailingChunkEntries )
    {
        trailingChunk = chunkCount;
        chunkCount ++;
    }

    // Start processing chunks
    for( uint chunk = 0; chunk < chunkCount; chunk++ )
    {
        // Adjust entry count if its the trailing chunk
        if( chunk == trailingChunk )
        {
            ASSERT( this->trailingChunkEntries );
            
            // Trailing chunk is special-case, has les entries than a full chunk.
            // We may not have to use all threads here as the trailing entries may
            // be less than we can handle per thread.
            uint32 threadCount      = this->_jobCount;

            uint32 trailingEntries  = this->trailingChunkEntries;
            uint32 entriesPerThread = trailingEntries / threadCount;

            while( !entriesPerThread )
            {
                entriesPerThread = trailingEntries / --threadCount;
            }

            // Reduce the number of jobs running if we now have
            // less thread counts than we started with
            if( threadCount != this->JobCount() )
            {
                if( !this->ReduceThreadCount( threadCount ) )
                    return;
            }

            // Get new trailing entries
            trailingEntries -= entriesPerThread * threadCount;

            // Set our new starting point
            uint32 chunkOffset = entriesPerChunk * chunk + entriesPerThread * _jobId;

            // Spread out any left-over entries accross the remaining threads
            if( _jobId < trailingEntries )
            {
                entriesPerThread ++;
                chunkOffset += _jobId;  // Add one entry for each job below ours
            }
            else
            {
                chunkOffset += trailingEntries; // Add add all trailing entries spread over the first threads
            }

            // Set new entry count & starting pos
            entryCount = entriesPerThread;
            
            pairs = this->GetJob( 0 ).pairs;
            pairs.left  += chunkOffset;
            pairs.right += chunkOffset;
        }

        // Grab a buffer
        if constexpr ( WriteToDisk )
        {
            ASSERT( diskQueue );

            if( this->IsControlThread() )
            {
                this->LockThreads();

                sizes   = (uint32*)diskQueue->GetBuffer( BB_DP_BUCKET_COUNT * sizeof( uint32 ) );
                yOut    = (uint32*)diskQueue->GetBuffer( this->bufferSizeY );
                sortKey = (uint32*)diskQueue->GetBuffer( this->bufferSizeY );

                if constexpr ( SizeMetaA > 0 )
                {
                    metaOutA   = (uint64*)diskQueue->GetBuffer( this->bufferSizeMetaA );
                    metaASizes = (uint32*)diskQueue->GetBuffer( BB_DP_BUCKET_COUNT * sizeof( uint32 ) );
                }

                if constexpr ( SizeMetaB > 0 )
                {
                    metaOutB   = (uint64*)diskQueue->GetBuffer( this->bufferSizeMetaB );
                    metaBSizes = (uint32*)diskQueue->GetBuffer( BB_DP_BUCKET_COUNT * sizeof( uint32 ) );
                }

                this->yOut       = yOut    ;
                this->sortKeyOut = sortKey ;
                this->metaOutA   = metaOutA;
                this->metaOutB   = metaOutB;

                this->ReleaseThreads();
            }
            else
            {
                this->WaitForRelease();

                yOut     = this->GetJob( 0 ).yOut;
                sortKey  = this->GetJob( 0 ).sortKeyOut;
                metaOutA = this->GetJob( 0 ).metaOutA;
                metaOutB = this->GetJob( 0 ).metaOutB;
            }
        }

        // Calculate fx for chunk
        ComputeFxForTable<tableId>( 
            bucket, entryCount, pairs,
            yIn, metaInA, metaInB,
            yTmp, bucketIdOut, metaTmpA, metaTmpB
            ,this->_jobId ); // #TODO: Remove job Id. It's for testing

        // Distribute entries into their corresponding buckets
        DistributeIntoBuckets<tableId, TMetaA, TMetaB>(
            entryCount, sortKeyOffset, bucketIndices,
            yTmp, (TMetaA*)metaATmp, (TMetaB*)metaBTmp,
            yOut, sortKey, (TMetaA*)metaOutA, (TMetaB*)metaOutB,
            counts, bucketCounts, fileBlockSize );

        if( this->IsControlThread() )
        {
            uint32* totalBucketCounts = this->totalBucketCounts;

            // Add to the the total count
            for( uint32 i = 0; i < BB_DP_BUCKET_COUNT; i++ )
                totalBucketCounts[i] += bucketCounts[i];
        }

        // Submit bucket buffers
        if constexpr ( WriteToDisk )
        {
            ASSERT( diskQueue );
            
            if( IsControlThread() )
            {
                this->LockThreads();
                
                for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
                    sizes[i] = (uint32)( bucketCounts[i] * sizeof( uint32 ) );

                diskQueue->WriteBuckets( yFileId, yOut, sizes );
                diskQueue->WriteBuckets( sortKeyFileId, sortKey, sizes );
                diskQueue->ReleaseBuffer( sizes   );
                diskQueue->ReleaseBuffer( yOut    );
                diskQueue->ReleaseBuffer( sortKey );
                diskQueue->CommitCommands();

                if constexpr ( SizeMetaA > 0 )
                {
                    for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
                        metaASizes[i] = (uint32)( bucketCounts[i] * SizeMetaA );

                    diskQueue->WriteBuckets( metaAFileId, metaOutA, metaASizes );
                    diskQueue->ReleaseBuffer( metaASizes );
                    diskQueue->ReleaseBuffer( metaOutA );
                    diskQueue->CommitCommands();
                }

                if constexpr ( SizeMetaB > 0 )
                {
                    for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
                        metaBSizes[i] = (uint32)( bucketCounts[i] * SizeMetaB );

                    diskQueue->WriteBuckets( metaBFileId, metaOutB, metaBSizes );
                    diskQueue->ReleaseBuffer( metaBSizes );
                    diskQueue->ReleaseBuffer( metaOutB );
                    diskQueue->CommitCommands();
                }

                this->ReleaseThreads();
            }
            else
            {
                this->WaitForRelease();
            }
        }

        // Go to next chunk
        pairs.left  += entriesPerChunk;
        pairs.right += entriesPerChunk;
    }
}

//-----------------------------------------------------------
template<TableId tableId, typename TMetaA, typename TMetaB>
void FxBucketJob::DistributeIntoBuckets(
    const uint32  entryCount,
    const uint32  sortKeyOffset,
    const byte*   bucketIndices,
    const uint32* y,
    const TMetaA* metaA,
    const TMetaB* metaB,
    uint32*       yBuckets,
    uint32*       sortKey,
    TMetaA*       metaABuckets,
    TMetaB*       metaBBuckets,
    uint32        counts         [BB_DP_BUCKET_COUNT],
    uint32        outBucketCounts[BB_DP_BUCKET_COUNT],
    const size_t  fileBlockSize
)
{
    const size_t metaSizeA = TableMetaOut<tableId>::SizeA;
    const size_t metaSizeB = TableMetaOut<tableId>::SizeB;

    uint32 pfxSum[BB_DP_BUCKET_COUNT];

    // Count our buckets
    memset( counts, 0, sizeof( uint32 ) * BB_DP_BUCKET_COUNT );
    for( const byte* ptr = bucketIndices, *end = ptr + entryCount; ptr < end; ptr++ )
    {
        ASSERT( *ptr <= ( 0b111111u ) );
        counts[*ptr] ++;
    }

    this->CalculatePrefixSum( counts, pfxSum, outBucketCounts, fileBlockSize );

    // #TODO: We may need an overflow sort key for when we have more
    //        entries than the k range can hold.
    uint32 key = sortKeyOffset;

    // #TODO: Unroll this a bit?
    // #TODO: Should this be done per output type (copy the pfxSum and reset it)?
    //        Explore, since it may perform better given the random access.
    // Distribute values into buckets at each thread's given offset
    for( uint i = 0; i < entryCount; i++ )
    {
        const uint32 dstIdx = --pfxSum[bucketIndices[i]];

        yBuckets[dstIdx] = y[i];
        sortKey [dstIdx] = key++;

        if constexpr ( metaSizeA > 0 )
            metaABuckets[dstIdx] = metaA[i];

        if constexpr ( metaSizeB > 0 )
            metaBBuckets[dstIdx] = metaB[i];
    }
}

//-----------------------------------------------------------
inline void FxBucketJob::CalculatePrefixSum( 
    uint32       counts      [BB_DP_BUCKET_COUNT],
    uint32       pfxSum      [BB_DP_BUCKET_COUNT],
    uint32       bucketCounts[BB_DP_BUCKET_COUNT],
    const size_t fileBlockSize
)
{
    const uint32 jobId    = this->JobId();
    const uint32 jobCount = this->JobCount();

    // This holds the count of extra entries added per-bucket
    // to align each bucket starting address to disk block size.
    // Only used when fileBlockSize > 0
    uint32 entryPadding[BB_DP_BUCKET_COUNT];

    this->counts = counts;
    this->SyncThreads();

    // Add up all of the jobs counts
    memset( pfxSum, 0, sizeof( uint32 ) * BB_DP_BUCKET_COUNT );

    for( uint i = 0; i < jobCount; i++ )
    {
        const uint* tCounts = this->GetJob( i ).counts;

        for( uint j = 0; j < BB_DP_BUCKET_COUNT; j++ )
            pfxSum[j] += tCounts[j];
    }

    // If we're the control thread, retain the total bucket count
    if( this->IsControlThread() )
    {
        memcpy( bucketCounts, pfxSum, sizeof( uint32 ) * BB_DP_BUCKET_COUNT );
    }

    // Only do this if using Direct IO
    // We need to align our bucket totals to the  file block size boundary
    // so that each block buffer is properly aligned for direct io.
    if( fileBlockSize )
    {
        #if _DEBUG
            size_t bucketAddress = 0;
        #endif

        for( uint i = 0; i < BB_DP_BUCKET_COUNT-1; i++ )
        {
            const uint32 count = pfxSum[i];

            pfxSum[i]       = RoundUpToNextBoundary( count * sizeof( uint32 ), (int)fileBlockSize ) / sizeof( uint32 );
            entryPadding[i] = pfxSum[i] - count;

            #if _DEBUG
                bucketAddress += pfxSum[i] * sizeof( uint32 );
                ASSERT( bucketAddress / fileBlockSize * fileBlockSize == bucketAddress );
            #endif
        }

        #if _DEBUG
        // if( this->IsControlThread() )
        // {
        //     size_t totalSize = 0;
        //     for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
        //         totalSize += pfxSum[i];

        //     totalSize *= sizeof( uint32 );
        //     Log::Line( "Total Size: %llu", totalSize );
        // }   
        #endif
    }

    // Calculate the prefix sum
    for( uint i = 1; i < BB_DP_BUCKET_COUNT; i++ )
        pfxSum[i] += pfxSum[i-1];

    // Subtract the count from all threads after ours 
    // to get the correct prefix sum for this thread
    for( uint t = jobId+1; t < jobCount; t++ )
    {
        const uint* tCounts = this->GetJob( t ).counts;

        for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
            pfxSum[i] -= tCounts[i];
    }

    if( fileBlockSize )
    {
        // Now that we have the starting addresses of the buckets
        // at a block-aligned position, we need to substract
        // the padding that we added to align them, so that
        // the entries actually get writting to the starting
        // point of the address

        for( uint i = 0; i < BB_DP_BUCKET_COUNT-1; i++ )
            pfxSum[i] -= entryPadding[i];
    }
}


#pragma GCC diagnostic pop

