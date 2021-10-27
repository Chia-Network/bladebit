#include "DiskPlotPhase1.h"
#include "pos/chacha8.h"
#include "Util.h"
#include "util/Log.h"
#include "algorithm/RadixSort.h"

// Test
#include "io/FileStream.h"
#include "SysHost.h"

struct WriteFileJob
{
    const char* filePath;

    size_t      size  ;
    size_t      offset;
    byte*       buffer;
    bool        success;

    static void Run( WriteFileJob* job );
};


//-----------------------------------------------------------
DiskPlotPhase1::DiskPlotPhase1( DiskPlotContext& cx )
    : _cx( cx )
    //, _diskQueue( cx.workBuffer, cx.diskFlushSize, (uint)(cx.bufferSizeBytes / cx.diskFlushSize) - 1 )
{
    ASSERT( cx.tmpPath );
    _diskQueue = new DiskBufferQueue( cx.tmpPath, cx.workBuffer, cx.bufferSizeBytes, cx.diskQueueThreadCount );
}

//-----------------------------------------------------------
void DiskPlotPhase1::Run()
{
#if !BB_DP_DBG_READ_EXISTING_F1
    GenF1();
#else
    {
        size_t pathLen = strlen( _cx.tmpPath );
        pathLen += sizeof( BB_DP_DBG_READ_BUCKET_COUNT_FNAME );

        std::string bucketsPath = _cx.tmpPath;
        if( bucketsPath[bucketsPath.length() - 1] != '/' && bucketsPath[bucketsPath.length() - 1] != '\\' )
            bucketsPath += "/";

        bucketsPath += BB_DP_DBG_READ_BUCKET_COUNT_FNAME;

        FileStream fBucketCounts;
        if( fBucketCounts.Open( bucketsPath.c_str(), FileMode::Open, FileAccess::Read ) )
        {
            size_t sizeRead = fBucketCounts.Read( _bucketCounts, sizeof( _bucketCounts ) );
            FatalIf( sizeRead != sizeof( _bucketCounts ), "Invalid bucket counts." );
        }
        else
        {
            GenF1();

            fBucketCounts.Close();
            FatalIf( !fBucketCounts.Open( bucketsPath.c_str(), FileMode::Create, FileAccess::Write ), "File to open bucket counts file" );
            FatalIf( fBucketCounts.Write( _bucketCounts, sizeof( _bucketCounts ) ) != sizeof( _bucketCounts ), "Failed to write bucket counts.");
        }

    }
#endif

    ForwardPropagate();
}

//-----------------------------------------------------------
void DiskPlotPhase1::GenF1()
{
    DiskPlotContext& cx   = _cx;
    ThreadPool&      pool = *cx.threadPool;

    // Prepare ChaCha key
    byte key[32] = { 1 };
    memcpy( key + 1, cx.plotId, 31 );

    // Prepare jobs
    const uint64 entryCount      = 1ull << _K;
    const size_t entryTotalSize  = entryCount * sizeof( uint32 );
    const uint32 entriesPerBlock = (uint32)( kF1BlockSize / sizeof( uint32 ) );
    const uint32 blockCount      = (uint32)( entryCount / entriesPerBlock );

    // #TODO: Enforce a minimum chunk size, and ensure that a (chunk size - counts) / threadCount
    // /      is sufficient to bee larger that the cache line size
    // const size_t countsBufferSize = sizeof( uint32 ) * BB_DP_BUCKET_COUNT;

    const size_t chunkBufferSize  = cx.diskFlushSize;
    const uint32 blocksPerChunk   = (uint32)( chunkBufferSize / ( kF1BlockSize * 2 ) );                 // * 2 because we also have to process/write the x values
    const uint32 chunkCount       = CDivT( blockCount, blocksPerChunk );                                // How many chunks we need to process
    const uint32 lastChunkBlocks  = blocksPerChunk - ( ( chunkCount * blocksPerChunk ) - blockCount );  // Last chunk might not need to process all blocks

    // Threads operate on a chunk at a time.
    const uint32 threadCount      = pool.ThreadCount();
    const uint32 blocksPerThread  = blocksPerChunk / threadCount;
    
    uint32 trailingBlocks = blocksPerChunk - ( blocksPerThread * threadCount );

    // #TODO: Ensure each thread has at least one block.
    ASSERT( blocksPerThread > 0 );

    uint  x            = 0;
    byte* blocksRoot   = _diskQueue->GetBuffer( blocksPerChunk * kF1BlockSize * 2 );
    byte* blocks       = blocksRoot;
    byte* xBuffer      = blocks + blocksPerChunk * kF1BlockSize;

    // Allocate buffers to track the remainders that are not multiple of the block size of the drive.
    // We do double-buffering here as we these buffers are tiny and we don't expect to get blocked by them.
    const size_t driveBlockSize = _diskQueue->BlockSize();
    const size_t remaindersSize = driveBlockSize * BB_DP_BUCKET_COUNT * 2;      // Double-buffered
    byte*        remainders    = _diskQueue->GetBuffer( remaindersSize * 2 );   // Allocate 2, one for y one for x. They are used together.

    memset( _bucketCounts, 0, sizeof( _bucketCounts ) );

    MTJobRunner<GenF1Job> f1Job( pool );

    for( uint i = 0; i < threadCount; i++ )
    {
        GenF1Job& job = f1Job[i];
        job.key            = key;
        job.blocksPerChunk = blocksPerChunk;
        job.chunkCount     = chunkCount;
        job.blockCount     = blocksPerThread;
        job.x              = x;

        job.buffer         = blocks;
        job.xBuffer        = (uint32*)xBuffer;

        job.counts         = nullptr;
        job.bucketCounts   = nullptr;
        job.buckets        = nullptr;
        
        job.diskQueue        = _diskQueue;
        job.remaindersBuffer = nullptr;

        if( trailingBlocks > 0 )
        {
            job.blockCount++;
            trailingBlocks--;
        }

        x       += job.blockCount * entriesPerBlock * chunkCount;
        blocks  += job.blockCount * kF1BlockSize;
        xBuffer += job.blockCount * kF1BlockSize;
    }

    f1Job[0].bucketCounts     = _bucketCounts;
    f1Job[0].remaindersBuffer = remainders;

    Log::Line( "Generating f1..." );
    double elapsed = f1Job.Run();
    Log::Line( "Finished f1 generation in %.2lf seconds. ", elapsed );

    // Release our buffers
    {
        AutoResetSignal fence;

        _diskQueue->ReleaseBuffer( blocksRoot );
        _diskQueue->ReleaseBuffer( remainders );
        _diskQueue->AddFence( fence );
        _diskQueue->CommitCommands();

        fence.Wait();
        _diskQueue->CompletePendingReleases();
    }
}

//-----------------------------------------------------------
void GenF1Job::Run()
{
    const uint32 entriesPerBlock  = kF1BlockSize / sizeof( uint32 );
    const uint32 kMinusKExtraBits = _K - kExtraBits;
    const uint32 bucketShift      = (8u - (uint)kExtraBits);
    
    const uint32 jobId      = this->JobId();
    const uint32 jobCount   = this->JobCount();

    byte*        blocks     = this->buffer;
    const uint32 blockCount = this->blockCount;
    const uint32 chunkCount = this->chunkCount;
    const uint64 entryCount = blockCount * (uint64)entriesPerBlock;

    const size_t bufferSize = this->blocksPerChunk * kF1BlockSize;

    DiskBufferQueue& queue         = *this->diskQueue;
    const size_t     fileBlockSize = queue.BlockSize();
    

    uint32 x = this->x;

    chacha8_ctx chacha;
    chacha8_keysetup( &chacha, key, 256, NULL );

    uint counts[BB_DP_BUCKET_COUNT];
    uint pfxSum[BB_DP_BUCKET_COUNT];


    // These are used only by the control thread
    // 
    // #NOTE: This is a lot of stuff allocated in the stack,
    //  but are thread's stack space is large enough.
    //  Consider allocating it int the heap however
    DoubleBuffer* remainders        = nullptr;
    uint*         remainderSizes    = nullptr;
    uint*         bucketCounts      = nullptr;
    uint*         totalBucketCounts = nullptr;

//     DoubleBuffer remainders       [BB_DP_BUCKET_COUNT];
//     uint         remainderSizes   [BB_DP_BUCKET_COUNT];
//     uint         bucketCounts     [BB_DP_BUCKET_COUNT];
//     uint         totalBucketCounts[BB_DP_BUCKET_COUNT];

    if( IsControlThread() )
    {
        // #TODO: _malloca seems to be giving issues on windows, so we're heap-allocating...
        remainders        = bbmalloc<DoubleBuffer>( sizeof( DoubleBuffer ) * BB_DP_BUCKET_COUNT );
        remainderSizes    = bbmalloc<uint32>      ( sizeof( uint )         * BB_DP_BUCKET_COUNT );
        bucketCounts      = bbmalloc<uint32>      ( sizeof( uint )         * BB_DP_BUCKET_COUNT );
        totalBucketCounts = bbmalloc<uint32>      ( sizeof( uint )         * BB_DP_BUCKET_COUNT );

        memset( remainderSizes   , 0, sizeof( uint32 ) * BB_DP_BUCKET_COUNT );
        memset( totalBucketCounts, 0, sizeof( uint32 ) * BB_DP_BUCKET_COUNT );

        // Layout for the buffers is:
        //
        // front: [y0][x0][y1][x1]...[y63][x63]
        // back:  [y0][x0][y1][x1]...[y63][x63]
        // 
        // So all 'front' buffers are contiguous for all buckets,
        // then follow all the 'back' buffers for all buckets.

        byte* front = this->remaindersBuffer;
        byte* back  = this->remaindersBuffer + fileBlockSize * BB_DP_BUCKET_COUNT * 2;

        for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
        {
            DoubleBuffer* dbuf = new ( (void*)&remainders[i] ) DoubleBuffer();

            dbuf->front = front;
            dbuf->back  = back;

            front += fileBlockSize * 2;
            back  += fileBlockSize * 2;
        }
    }

    for( uint i = 0; i < chunkCount; i++ )
    {
        chacha8_get_keystream( &chacha, x, blockCount, blocks );

        // Count how many entries we have per bucket
        memset( counts, 0, sizeof( counts ) );

        const uint32* block = (uint32*)blocks;

        // #TODO: If last chunk, get the block count for last chunk.
        // #TODO: Process chunk in its own function

        for( uint j = 0; j < blockCount; j++ )
        {
            // Unroll a whole block

            // Determine the bucket id by grabbing the lowest kExtrabits, the highest
            // kExtraBits from the LSB. This is equivalent to the kExtraBits MSbits of the entry
            // once it is endian-swapped later.
            // 0x3F == 6 bits( kExtraBits )
            const uint32 e0  = ( block[0 ] >> bucketShift ) & 0x3F; ASSERT( e0  < 256u );
            const uint32 e1  = ( block[1 ] >> bucketShift ) & 0x3F; ASSERT( e1  < 256u );
            const uint32 e2  = ( block[2 ] >> bucketShift ) & 0x3F; ASSERT( e2  < 256u );
            const uint32 e3  = ( block[3 ] >> bucketShift ) & 0x3F; ASSERT( e3  < 256u );
            const uint32 e4  = ( block[4 ] >> bucketShift ) & 0x3F; ASSERT( e4  < 256u );
            const uint32 e5  = ( block[5 ] >> bucketShift ) & 0x3F; ASSERT( e5  < 256u );
            const uint32 e6  = ( block[6 ] >> bucketShift ) & 0x3F; ASSERT( e6  < 256u );
            const uint32 e7  = ( block[7 ] >> bucketShift ) & 0x3F; ASSERT( e7  < 256u );
            const uint32 e8  = ( block[8 ] >> bucketShift ) & 0x3F; ASSERT( e8  < 256u );
            const uint32 e9  = ( block[9 ] >> bucketShift ) & 0x3F; ASSERT( e9  < 256u );
            const uint32 e10 = ( block[10] >> bucketShift ) & 0x3F; ASSERT( e10 < 256u );
            const uint32 e11 = ( block[11] >> bucketShift ) & 0x3F; ASSERT( e11 < 256u );
            const uint32 e12 = ( block[12] >> bucketShift ) & 0x3F; ASSERT( e12 < 256u );
            const uint32 e13 = ( block[13] >> bucketShift ) & 0x3F; ASSERT( e13 < 256u );
            const uint32 e14 = ( block[14] >> bucketShift ) & 0x3F; ASSERT( e14 < 256u );
            const uint32 e15 = ( block[15] >> bucketShift ) & 0x3F; ASSERT( e15 < 256u );

            counts[e0 ] ++;
            counts[e1 ] ++;
            counts[e2 ] ++;
            counts[e3 ] ++;
            counts[e4 ] ++;
            counts[e5 ] ++;
            counts[e6 ] ++;
            counts[e7 ] ++;
            counts[e8 ] ++;
            counts[e9 ] ++;
            counts[e10] ++;
            counts[e11] ++;
            counts[e12] ++;
            counts[e13] ++;
            counts[e14] ++;
            counts[e15] ++;

            block += entriesPerBlock;
        }


        // Wait for all threads to finish ChaCha generation
        this->counts = counts;
        SyncThreads();

        // Add up all of the jobs counts
        memset( pfxSum, 0, sizeof( pfxSum ) );

        for( uint j = 0; j < jobCount; j++ )
        {
            const uint* tCounts = GetJob( j ).counts;

            for( uint k = 0; k < BB_DP_BUCKET_COUNT; k++ )
                pfxSum[k] += tCounts[k];
        }

        // If we're the control thread, retain the total bucket count for this chunk
        uint32 totalCount = 0;
        if( this->IsControlThread() )
        {
            memcpy( bucketCounts, pfxSum, sizeof( pfxSum ) );
        }
        
        // #TODO: Only do this for the control thread
        for( uint j = 0; j < BB_DP_BUCKET_COUNT; j++ )
            totalCount += pfxSum[j];

        // Calculate the prefix sum for this thread
        for( uint j = 1; j < BB_DP_BUCKET_COUNT; j++ )
            pfxSum[j] += pfxSum[j-1];

        // Subtract the count from all threads after ours
        for( uint t = jobId+1; t < jobCount; t++ )
        {
            const uint* tCounts = GetJob( t ).counts;

            for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
                pfxSum[i] -= tCounts[i];
        }

        // Now we know the offset where we can start storing bucketized y values
        block = (uint*)blocks;
        uint32* sizes   = nullptr;
        uint32* buckets = nullptr;
        uint32* xBuffer = nullptr;
        
        // Grab a buffer from the queue
        if( this->LockThreads() )
        {
            sizes   = (uint32*)queue.GetBuffer( (sizeof( uint32 ) * BB_DP_BUCKET_COUNT ) );
            buckets = (uint32*)queue.GetBuffer( bufferSize );
            xBuffer = (uint32*)queue.GetBuffer( bufferSize );

            this->buckets = buckets;
            this->xBuffer = xBuffer;
            this->ReleaseThreads();
        }
        else
        {
            WaitForRelease();
            buckets = GetJob( 0 ).buckets;
            xBuffer = GetJob( 0 ).xBuffer;
        }

        ASSERT( pfxSum[63] <= totalCount );

        // Distribute values into buckets at each thread's given offset
        for( uint j = 0; j < blockCount; j++ )
        {
            // chacha output is treated as big endian, therefore swap, as required by chiapos
            const uint32 y0  = Swap32( block[0 ] );
            const uint32 y1  = Swap32( block[1 ] );
            const uint32 y2  = Swap32( block[2 ] );
            const uint32 y3  = Swap32( block[3 ] );
            const uint32 y4  = Swap32( block[4 ] );
            const uint32 y5  = Swap32( block[5 ] );
            const uint32 y6  = Swap32( block[6 ] );
            const uint32 y7  = Swap32( block[7 ] );
            const uint32 y8  = Swap32( block[8 ] );
            const uint32 y9  = Swap32( block[9 ] );
            const uint32 y10 = Swap32( block[10] );
            const uint32 y11 = Swap32( block[11] );
            const uint32 y12 = Swap32( block[12] );
            const uint32 y13 = Swap32( block[13] );
            const uint32 y14 = Swap32( block[14] );
            const uint32 y15 = Swap32( block[15] );

            const uint32 idx0  = --pfxSum[y0  >> kMinusKExtraBits];
            const uint32 idx1  = --pfxSum[y1  >> kMinusKExtraBits];
            const uint32 idx2  = --pfxSum[y2  >> kMinusKExtraBits];
            const uint32 idx3  = --pfxSum[y3  >> kMinusKExtraBits];
            const uint32 idx4  = --pfxSum[y4  >> kMinusKExtraBits];
            const uint32 idx5  = --pfxSum[y5  >> kMinusKExtraBits];
            const uint32 idx6  = --pfxSum[y6  >> kMinusKExtraBits];
            const uint32 idx7  = --pfxSum[y7  >> kMinusKExtraBits];
            const uint32 idx8  = --pfxSum[y8  >> kMinusKExtraBits];
            const uint32 idx9  = --pfxSum[y9  >> kMinusKExtraBits];
            const uint32 idx10 = --pfxSum[y10 >> kMinusKExtraBits];
            const uint32 idx11 = --pfxSum[y11 >> kMinusKExtraBits];
            const uint32 idx12 = --pfxSum[y12 >> kMinusKExtraBits];
            const uint32 idx13 = --pfxSum[y13 >> kMinusKExtraBits];
            const uint32 idx14 = --pfxSum[y14 >> kMinusKExtraBits];
            const uint32 idx15 = --pfxSum[y15 >> kMinusKExtraBits];

            // Add the x as the kExtraBits, and strip away the high kExtraBits,
            // which is now our bucket id, and place each entry into it's respective bucket
            // #NOTE: False sharing can occur here
            buckets[idx0 ] = ( y0  << kExtraBits ) | ( ( x + 0  ) >> kMinusKExtraBits );
            buckets[idx1 ] = ( y1  << kExtraBits ) | ( ( x + 1  ) >> kMinusKExtraBits );
            buckets[idx2 ] = ( y2  << kExtraBits ) | ( ( x + 2  ) >> kMinusKExtraBits );
            buckets[idx3 ] = ( y3  << kExtraBits ) | ( ( x + 3  ) >> kMinusKExtraBits );
            buckets[idx4 ] = ( y4  << kExtraBits ) | ( ( x + 4  ) >> kMinusKExtraBits );
            buckets[idx5 ] = ( y5  << kExtraBits ) | ( ( x + 5  ) >> kMinusKExtraBits );
            buckets[idx6 ] = ( y6  << kExtraBits ) | ( ( x + 6  ) >> kMinusKExtraBits );
            buckets[idx7 ] = ( y7  << kExtraBits ) | ( ( x + 7  ) >> kMinusKExtraBits );
            buckets[idx8 ] = ( y8  << kExtraBits ) | ( ( x + 8  ) >> kMinusKExtraBits );
            buckets[idx9 ] = ( y9  << kExtraBits ) | ( ( x + 9  ) >> kMinusKExtraBits );
            buckets[idx10] = ( y10 << kExtraBits ) | ( ( x + 10 ) >> kMinusKExtraBits );
            buckets[idx11] = ( y11 << kExtraBits ) | ( ( x + 11 ) >> kMinusKExtraBits );
            buckets[idx12] = ( y12 << kExtraBits ) | ( ( x + 12 ) >> kMinusKExtraBits );
            buckets[idx13] = ( y13 << kExtraBits ) | ( ( x + 13 ) >> kMinusKExtraBits );
            buckets[idx14] = ( y14 << kExtraBits ) | ( ( x + 14 ) >> kMinusKExtraBits );
            buckets[idx15] = ( y15 << kExtraBits ) | ( ( x + 15 ) >> kMinusKExtraBits );

            // Store the x that generated this y
            xBuffer[idx0 ] = x + 0 ;
            xBuffer[idx1 ] = x + 1 ;
            xBuffer[idx2 ] = x + 2 ;
            xBuffer[idx3 ] = x + 3 ;
            xBuffer[idx4 ] = x + 4 ;
            xBuffer[idx5 ] = x + 5 ;
            xBuffer[idx6 ] = x + 6 ;
            xBuffer[idx7 ] = x + 7 ;
            xBuffer[idx8 ] = x + 8 ;
            xBuffer[idx9 ] = x + 9 ;
            xBuffer[idx10] = x + 10;
            xBuffer[idx11] = x + 11;
            xBuffer[idx12] = x + 12;
            xBuffer[idx13] = x + 13;
            xBuffer[idx14] = x + 14;
            xBuffer[idx15] = x + 15;

            block += entriesPerBlock;
            x     += entriesPerBlock;
        }

        // Now this chunk can be submitted to the write queue, and we can continue to the next one.
        // After all the chunks have been written, we can read back from disk to sort each bucket
        if( this->LockThreads() )
        {
            // Calculate the disk block-aligned size
            // #TODO: Don't do this if not using direct IO?
            for( uint j = 0; j < BB_DP_BUCKET_COUNT; j++ )
                sizes[j] = (uint32)( ( bucketCounts[j] * sizeof( uint32 ) ) / fileBlockSize * fileBlockSize );

            queue.WriteBuckets( FileId::Y, buckets, sizes );
            queue.WriteBuckets( FileId::X, xBuffer, sizes );
            queue.CommitCommands();


            // If we're not at our last chunk, we need to shave-off
            // any entries that will not align to the file block size and
            // leave them in our buckets for the next run.
            SaveBlockRemainders( buckets, xBuffer, bucketCounts, remainders, remainderSizes );

            queue.ReleaseBuffer( sizes   );
            queue.ReleaseBuffer( buckets );
            queue.ReleaseBuffer( xBuffer );
            queue.CommitCommands();

            // Add total bucket counts
            for( uint j = 0; j < BB_DP_BUCKET_COUNT; j++ )
                totalBucketCounts[j] += bucketCounts[j];

            this->ReleaseThreads();
        }
        else
            this->WaitForRelease();
    }

    if( IsControlThread() )
    {
        // we need to write out any remainders as a whole block
        WriteFinalBlockRemainders( remainders, remainderSizes );

        // Copy final total bucket counts
        memcpy( this->bucketCounts, totalBucketCounts, sizeof( uint32 )* BB_DP_BUCKET_COUNT );

        // Wait for our commands to finish
        AutoResetSignal fence;
        queue.AddFence( fence );
        queue.CommitCommands();
        fence.Wait();

        // Destruct & free our remainders
        for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
            remainders[i].~DoubleBuffer();

        free( remainders        );
        free( remainderSizes    );
        free( bucketCounts      );
        free( totalBucketCounts );
    }
}

//-----------------------------------------------------------
inline void GenF1Job::SaveBlockRemainders( uint32* yBuffer, uint32* xBuffer, const uint32* bucketCounts, 
                                           DoubleBuffer* remainderBuffers, uint32* remainderSizes )
{
    DiskBufferQueue& queue               = *this->diskQueue;
    const size_t     fileBlockSize       = queue.BlockSize();
    const size_t     remainderBufferSize = fileBlockSize * BB_DP_BUCKET_COUNT;

    byte* yPtr = (byte*)buckets;
    byte* xPtr = (byte*)xBuffer;

    for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
    {
        const size_t bucketSize       = bucketCounts[i] * sizeof( uint32 );
        const size_t blockAlignedSize = bucketSize / fileBlockSize * fileBlockSize;
                    
        size_t remainderSize = bucketSize - blockAlignedSize;
        ASSERT( remainderSize / 4 * 4 == remainderSize );

        if( remainderSize )
        {
            size_t curRemainderSize = remainderSizes[i];
                        
            const size_t copySize = std::min( remainderSize, fileBlockSize - curRemainderSize );

            DoubleBuffer& buf = remainderBuffers[i];

            byte* yRemainder = buf.front;
            byte* xRemainder = yRemainder + fileBlockSize;

            bbmemcpy_t( yRemainder + curRemainderSize, yPtr + blockAlignedSize, copySize );
            bbmemcpy_t( xRemainder + curRemainderSize, xPtr + blockAlignedSize, copySize );

            curRemainderSize += remainderSize;

            if( curRemainderSize >= fileBlockSize )
            {
                // This may block if the last buffer has not yet finished writing to disk
                buf.Flip();

                // Overflow buffer is full, submit it for writing
                queue.WriteFile( FileId::Y, i, yRemainder, fileBlockSize );
                queue.WriteFile( FileId::X, i, xRemainder, fileBlockSize );
                queue.AddFence( buf.fence );
                queue.CommitCommands();

                // Update new remainder size, if we overflowed our buffer
                // and copy any overflow, if we have some.
                remainderSize = curRemainderSize - fileBlockSize;

                if( remainderSize )
                {
                    yRemainder = buf.front;
                    xRemainder = yRemainder + fileBlockSize;

                    bbmemcpy_t( yRemainder, yPtr + blockAlignedSize + copySize, remainderSize );
                    bbmemcpy_t( xRemainder, xPtr + blockAlignedSize + copySize, remainderSize );
                }

                remainderSizes[i] = 0;
                remainderSize     = remainderSize;
            }

            // Update size
            remainderSizes[i] += (uint)remainderSize;
        }

        yPtr += bucketSize;
        xPtr += bucketSize;
    }
}

//-----------------------------------------------------------
void GenF1Job::WriteFinalBlockRemainders( DoubleBuffer* remainderBuffers, uint32* remainderSizes )
{
    DiskBufferQueue& queue         = *this->diskQueue;
    const size_t     fileBlockSize = queue.BlockSize();

    for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
    {
        const size_t size = remainderSizes[i];
        ASSERT( size / 4 * 4 == size );
        
        if( size == 0 )
            continue;

        const DoubleBuffer& buf = remainderBuffers[i];
        
        byte* yBuffer = buf.front;
        byte* xBuffer = yBuffer + fileBlockSize;

        queue.WriteFile( FileId::Y, i, yBuffer, size );
        queue.WriteFile( FileId::X, i, xBuffer, size );
    }
}


//-----------------------------------------------------------
void DiskPlotPhase1::ForwardPropagate()
{
    DiskBufferQueue& ioDispatch  = *_diskQueue;
    ThreadPool&      threadPool  = *_cx.threadPool;
    const uint       threadCount = _cx.threadCount;

    size_t maxBucketSize = 0;

    // Find the largest bucket so that we can reserve buffers of its size
    for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
        maxBucketSize = std::max( maxBucketSize, (size_t)_bucketCounts[i] );

    maxBucketSize *= sizeof( uint32 );

    // Allocate 2 buffers for loading buckets
    DoubleBuffer bucketBuffers;
    bucketBuffers.front = ioDispatch.GetBuffer( maxBucketSize * 4 );
    bucketBuffers.back  = bucketBuffers.front + maxBucketSize * 2;


    _maxBucketCount = maxBucketSize / sizeof( uint32 );
    _bucketBuffers  = &bucketBuffers;

    // Allocate temp buffers & BC group buffers
    uint32* yTemp           = (uint32*)ioDispatch.GetBuffer( maxBucketSize );
    uint32* metaTemp        = (uint32*)ioDispatch.GetBuffer( maxBucketSize * 4 );
    uint32* groupBoundaries = (uint32*)ioDispatch.GetBuffer( BB_DP_MAX_BC_GROUP_PER_BUCKET * sizeof( uint32 ) );
    
    const size_t entriesPerBucket  = ( 1ull << _K ) / BB_DP_BUCKET_COUNT;
    const uint32 maxPairsPerThread = (uint32)( entriesPerBucket / threadCount + BB_DP_XTRA_MATCHES_PER_THREAD );
    const size_t maxPairsPerbucket = maxPairsPerThread * threadCount;

    uint32* lGroups = (uint32*)ioDispatch.GetBuffer( maxPairsPerbucket * sizeof( uint32 ) );
    uint16* rGroups = (uint16*)ioDispatch.GetBuffer( maxPairsPerbucket * sizeof( uint16 ) );

    // The sort key initially is set to the x buffer
    uint32* sortKey     = (uint32*)( bucketBuffers.front + maxBucketSize );
    uint32* realSortKey = nullptr;

    // Reset the fence as we're about to use it
    bucketBuffers.fence.Wait();

    // Seek all buckets to the start
    ioDispatch.SeekBucket( FileId::Y, 0, SeekOrigin::Begin );
    ioDispatch.SeekBucket( FileId::X, 0, SeekOrigin::Begin );

    // Load initial bucket
    ioDispatch.ReadFile( FileId::Y, 0, bucketBuffers.front, _bucketCounts[0] * sizeof( uint32 ) );
    ioDispatch.ReadFile( FileId::X, 0, sortKey            , _bucketCounts[0] * sizeof( uint32 ) );
    ioDispatch.AddFence( bucketBuffers.fence );
    ioDispatch.CommitCommands();
    bucketBuffers.fence.Wait();

    for( uint bucketIdx = 0; bucketIdx < BB_DP_BUCKET_COUNT; bucketIdx++ )
    {
        Log::Line( "Forward Propagating bucket %-2u", bucketIdx );

        const uint entryCount = _bucketCounts[bucketIdx];

        // Read the next bucket in the background if we're not at the last bucket
        const uint nextBucketIdx = bucketIdx + 1;

        if( nextBucketIdx < BB_DP_BUCKET_COUNT )
        {
            const size_t readSize = _bucketCounts[nextBucketIdx] * sizeof( uint32 );

            ioDispatch.ReadFile( FileId::Y, nextBucketIdx, bucketBuffers.back, readSize );
            ioDispatch.ReadFile( FileId::X, nextBucketIdx, bucketBuffers.back + maxBucketSize, readSize );
            ioDispatch.AddFence( bucketBuffers.fence );
            ioDispatch.CommitCommands();
        }
        else
            bucketBuffers.fence.Signal();

        // Sort our current bucket
        uint32* yBuffer    = (uint32*)bucketBuffers.front;
        uint32* metaBuffer = (uint32*)( bucketBuffers.front + maxBucketSize );

        {
            Log::Line( "  Sorting bucket." );
            auto timer = TimerBegin();
            RadixSort256::SortWithKey<BB_MAX_JOBS>( threadPool, yBuffer, yTemp, metaBuffer, metaTemp, entryCount );
            double elapsed = TimerEnd( timer );

            Log::Line( "  Sorted bucket in %.2lf seconds.", elapsed );
        }

        // Scan for BC groups
        GroupInfo groupInfos[BB_MAX_JOBS];

        const uint32 groupCount = ScanGroups( bucketIdx, yBuffer, entryCount, groupBoundaries, BB_DP_MAX_BC_GROUP_PER_BUCKET, groupInfos );

        // Match pairs
        struct Pairs pairs[BB_MAX_JOBS];

        for( uint i = 0; i < threadCount; i++ )
        {
            pairs[i].left  = lGroups + i * maxPairsPerThread;
            pairs[i].right = rGroups + i * maxPairsPerThread;
        }

        Match( bucketIdx, maxPairsPerThread, yBuffer, groupInfos, pairs );

        // Ensure the next buffer has been read
        bucketBuffers.Flip();
    }
}

//-----------------------------------------------------------
void DiskPlotPhase1::ForwardPropagateTable( TableId table )
{
    
}

//-----------------------------------------------------------
uint32 DiskPlotPhase1::ScanGroups( uint bucketIdx, const uint32* yBuffer, uint32 entryCount, uint32* groups, uint32 maxGroups, GroupInfo groupInfos[BB_MAX_JOBS] )
{
    Log::Line( "  Scanning for groups." );

    auto& cx = _cx;

    ThreadPool& pool               = *cx.threadPool;
    const uint  threadCount        = _cx.threadCount;
    const uint  maxGroupsPerThread = maxGroups / threadCount - 1;   // -1 because we need to add an extra end index to check R group 
                                                                    // without adding an extra 'if'
    MTJobRunner<ScanGroupJob> jobs( pool );

    jobs[0].yBuffer         = yBuffer;
    jobs[0].groupBoundaries = groups;
    jobs[0].bucketIdx       = bucketIdx;
    jobs[0].startIndex      = 0;
    jobs[0].endIndex        = entryCount;
    jobs[0].maxGroups       = maxGroupsPerThread;
    jobs[0].groupCount      = 0;
    
    for( uint i = 1; i < threadCount; i++ )
    {
        ScanGroupJob& job = jobs[i];

        job.yBuffer         = yBuffer;
        job.groupBoundaries = groups + maxGroupsPerThread * i;
        job.bucketIdx       = bucketIdx;
        job.maxGroups       = maxGroupsPerThread;
        job.groupCount      = 0;

        const uint32 idx           = entryCount / threadCount * i;
        const uint32 y             = yBuffer[idx];
        const uint32 curGroup      = y / kBC;
        const uint32 groupLocalIdx = y - curGroup * kBC;

        uint32 targetGroup;

        // If we are already at the start of a group, just use this index
        if( groupLocalIdx == 0 )
        {
            job.startIndex = idx;
        }
        else
        {
            // Choose if we should find the upper boundary or the lower boundary
            const uint32 remainder = kBC - groupLocalIdx;
            
            #if _DEBUG
                bool foundBoundary = false;
            #endif
            if( remainder <= kBC / 2 )
            {
                // Look for the upper boundary
                for( uint32 j = idx+1; j < entryCount; j++ )
                {
                    targetGroup = yBuffer[j] / kBC;
                    if( targetGroup != curGroup )
                    {
                        #if _DEBUG
                            foundBoundary = true;
                        #endif
                        job.startIndex = j; break;
                    }   
                }
            }
            else
            {
                // Look for the lower boundary
                for( uint32 j = idx-1; j >= 0; j-- )
                {
                    targetGroup = yBuffer[j] / kBC;
                    if( targetGroup != curGroup )
                    {
                        #if _DEBUG
                            foundBoundary = true;
                        #endif
                        job.startIndex = j+1; break;
                    }  
                }
            }

            ASSERT( foundBoundary );
        }

        auto& lastJob = jobs[i-1];
        ASSERT( job.startIndex > lastJob.startIndex );  // #TODO: This should not happen but there should
                                                        //        be a pre-check in the off chance that the thread count is really high.
                                                        //        Again, should not happen with the hard-coded thread limit,
                                                        //        but we can check if entryCount / threadCount <= kBC 


        // We add +1 so that the next group boundary is added to the list, and we can tell where the R group ends.
        lastJob.endIndex = job.startIndex + 1;

        ASSERT( yBuffer[job.startIndex-1] / kBC != yBuffer[job.startIndex] / kBC );

        job.groupBoundaries = groups + maxGroupsPerThread * i;
    }

    // Fill in missing data for the last job
    jobs[threadCount-1].endIndex = entryCount;

    // Run the scan job
    const double elapsed = jobs.Run();
    Log::Line( "  Finished group scan in %.2lf seconds." );

    // Get the total group count
    uint groupCount = 0;

    for( uint i = 0; i < threadCount-1; i++ )
    {
        auto& job = jobs[i];

        // Add a trailing end index (but don't count it) so that we can test against it
        job.groupBoundaries[job.groupCount] = jobs[i+1].groupBoundaries[0];

        groupInfos[i].groupBoundaries = job.groupBoundaries;
        groupInfos[i].groupCount      = job.groupCount;
        groupInfos[i].startIndex      = job.startIndex;

        groupCount += job.groupCount;
    }
    
    // Let the last job know where its R group is
    auto& lastJob = jobs[threadCount-1];
    lastJob.groupBoundaries[lastJob.groupCount] = entryCount;

    groupInfos[threadCount-1].groupBoundaries = lastJob.groupBoundaries;
    groupInfos[threadCount-1].groupCount      = lastJob.groupCount;
    groupInfos[threadCount-1].startIndex      = lastJob.startIndex;

    Log::Line( "  Found %u groups.", groupCount );

    return groupCount;
}

//-----------------------------------------------------------
void DiskPlotPhase1::Match( uint bucketIdx, uint maxPairsPerThread, const uint32* yBuffer, GroupInfo groupInfos[BB_MAX_JOBS], Pairs pairs[BB_MAX_JOBS] )
{
    Log::Line( "  Matching groups." );

    auto&      cx          = _cx;
    const uint threadCount = cx.threadCount;

    MTJobRunner<MatchJob> jobs( *cx.threadPool );

    for( uint i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];

        job.yBuffer         = yBuffer;
        job.pairs           = &pairs[i];
        job.bucketIdx       = bucketIdx;
        job.maxPairCount    = maxPairsPerThread;
        job.pairCount       = 0;
        job.groupInfo       = &groupInfos[i];
        job.copyLDst        = nullptr;
        job.copyRDst        = nullptr;
    }

    const double elapsed = jobs.Run();

    Log::Line( "  Finished matching groups in %.2lf seconds.", elapsed );
}

//-----------------------------------------------------------
void ScanGroupJob::Run()
{
    const uint32 maxGroups = this->maxGroups;
    
    uint32* groupBoundaries = this->groupBoundaries;
    uint32  grouipCount     = 0;

    const uint32* yBuffer = this->yBuffer;
    const uint32  start   = this->startIndex;
    const uint32  end     = this->endIndex;

    const uint64 bucket = ( (uint64)this->bucketIdx ) << 32;

    uint64 lastGroup = ( bucket | yBuffer[start] ) / kBC;

    for( uint32 i = start+1; i < end; i++ )
    {
        uint64 group = ( bucket | yBuffer[i] ) / kBC;

        if( group != lastGroup )
        {
            ASSERT( group > lastGroup );

            groupBoundaries[groupCount++] = i;
            lastGroup = group;

            if( groupCount == maxGroups )
            {
                ASSERT( 0 );    // We ought to always have enough space
                                // So this should be an error
                break;
            }
        }
    }

    this->groupCount = groupCount;
}

//-----------------------------------------------------------
void MatchJob::Run()
{
    const uint32* yBuffer         = this->yBuffer;
    const uint32* groupBoundaries = this->groupInfo->groupBoundaries;
    const uint32  groupCount      = this->groupInfo->groupCount;
    const uint32  maxPairs        = this->maxPairCount;
    const uint64  bucket          = ((uint64)this->bucketIdx) << 32;

    Pairs  pairs     = *this->pairs;
    uint32 pairCount = 0;

    uint8  rMapCounts [kBC];
    uint16 rMapIndices[kBC];

    uint64 groupLStart = this->groupInfo->startIndex;
    uint64 groupL      = ( bucket | yBuffer[groupLStart] ) / kBC;

    for( uint32 i = 0; i < groupCount; i++ )
    {
        const uint64 groupRStart = groupBoundaries[i];
        const uint64 groupR      = ( bucket | yBuffer[groupRStart] ) / kBC;

        if( groupR - groupL == 1 )
        {
            // Groups are adjacent, calculate matches
            const uint16 parity           = groupL & 1;
            const uint64 groupREnd        = groupBoundaries[i+1];

            const uint64 groupLRangeStart = groupL * kBC;
            const uint64 groupRRangeStart = groupR * kBC;
            
            ASSERT( groupREnd - groupRStart <= 350 );
            ASSERT( groupLRangeStart == groupRRangeStart - kBC );

            // Prepare a map of range kBC to store which indices from groupR are used
            // For now just iterate our bGroup to find the pairs
           
            // #NOTE: memset(0) works faster on average than keeping a separate a clearing buffer
            memset( rMapCounts, 0, sizeof( rMapCounts ) );

            for( uint64 iR = groupRStart; iR < groupREnd; iR++ )
            {
                uint64 localRY = ( bucket | yBuffer[iR] ) - groupRRangeStart;
                ASSERT( ( bucket | yBuffer[iR] ) / kBC == groupR );

                if( rMapCounts[localRY] == 0 )
                    rMapIndices[localRY] = (uint16)( iR - groupRStart );

                rMapCounts[localRY] ++;
            }

            // For each group L entry
            for( uint64 iL = groupLStart; iL < groupRStart; iL++ )
            {
                const uint64 yL     = bucket | yBuffer[iL];
                const uint64 localL = yL - groupLRangeStart;

                // Iterate kExtraBitsPow = 1 << kExtraBits = 1 << 6 == 64
                // So iterate 64 times for each L entry.
                for( int iK = 0; iK < kExtraBitsPow; iK++ )
                {
                    const uint64 targetR = L_targets[parity][localL][iK];

                    for( uint j = 0; j < rMapCounts[targetR]; j++ )
                    {
                        const uint64 iR = groupRStart + rMapIndices[targetR] + j;

                        ASSERT( iL < iR );

                        // Add a new pair
                        ASSERT( ( iR - iL ) <= 0xFFFF );

                        pairs.left [pairCount] = (uint32)iL;
                        pairs.right[pairCount] = (uint16)(iR - iL);
                        pairCount++;

                        // #TODO: Write to disk if there's a buffer available and we have enough entries to write
                        
                        ASSERT( pairCount <= maxPairs );
                        if( pairCount == maxPairs )
                            goto RETURN;
                    }
                }
            }
        }
        // Else: Not an adjacent group, skip to next one.

        // Go to next group
        groupL      = groupR;
        groupLStart = groupRStart;
    }

RETURN:
    this->pairCount = pairCount;
}




//-----------------------------------------------------------
void WriteFileJob::Run( WriteFileJob* job )
{
    job->success = false;

    FileStream file;
    if( !file.Open( job->filePath, FileMode::Open, FileAccess::Write, FileFlags::NoBuffering | FileFlags::LargeFile ) )
        return;

    ASSERT( job->offset == ( job->offset / file.BlockSize() ) * file.BlockSize() );

    if( !file.Seek( (int64)job->offset, SeekOrigin::Begin ) )
        return;

    // Begin writing at offset
    size_t sizeToWrite = job->size;
    byte*  buffer      = job->buffer;

    while( sizeToWrite )
    {
        const ssize_t sizeWritten = file.Write( buffer, sizeToWrite );
        if( sizeWritten < 1 )
            return;

        ASSERT( (size_t)sizeWritten >= sizeToWrite );
        sizeToWrite -= (size_t)sizeWritten;
    }

    // OK
    job->success = true;
}


//-----------------------------------------------------------
void TestWrites()
{
    // Test file
    const char* filePath = "E:/bbtest.data";

    FileStream file;
    if( !file.Open( filePath, FileMode::Create, FileAccess::Write, FileFlags::LargeFile | FileFlags::NoBuffering ) )
        Fatal( "Failed to open file." );

    const size_t blockSize = file.BlockSize();

    byte key[32] = {
        22, 24, 11, 3, 1, 15, 11, 6, 23, 22,
        22, 24, 11, 3, 1, 15, 11, 6, 23, 22,
        22, 24, 11, 3, 1, 15, 11, 6, 23, 22, 5, 28
    };

    const uint   chachaBlockSize = 64;

    const uint   k          = 30;
    const uint64 entryCount = 1ull << k;
    const uint   blockCount = (uint)( entryCount / chachaBlockSize );
    
    SysHost::SetCurrentThreadAffinityCpuId( 0 );

    uint32* buffer;

    {
        Log::Line( "Allocating %.2lf MB buffer...", (double)( entryCount * sizeof( uint32 ) ) BtoMB );
        auto timer = TimerBegin();

        buffer = (uint32*)SysHost::VirtualAlloc( entryCount * sizeof( uint32 ), true );
        FatalIf( !buffer, "Failed to allocate buffer." );

        double elapsed = TimerEnd( timer );
        Log::Line( "Finished in %.2lf seconds.", elapsed );
    }

    {
        chacha8_ctx chacha;
        ZeroMem( &chacha );

        Log::Line( "Generating ChaCha..." );
        auto timer = TimerBegin();

        chacha8_keysetup( &chacha, key, 256, NULL );
        chacha8_get_keystream( &chacha, 0, blockCount, (byte*)buffer );

        double elapsed = TimerEnd( timer );
        Log::Line( "Finished in %.2lf seconds.", elapsed );
    }

    bool singleThreaded = false;
    
    if( singleThreaded )
    {
        Log::Line( "Started writing to file..." );

        const size_t sizeWrite = entryCount * sizeof( uint );
        const size_t blockSize = file.BlockSize();
        
        size_t blocksToWrite = sizeWrite / blockSize;

        auto timer = TimerBegin();

        do
        {
            ssize_t written = file.Write( buffer, blocksToWrite * blockSize );
            FatalIf( written <= 0, "Failed to write to file." );

            size_t blocksWritten = (size_t)written / blockSize;
            ASSERT( blocksWritten <= blocksToWrite );

            blocksToWrite -= blocksWritten;
        } while( blocksToWrite > 0 );
        
        double elapsed = TimerEnd( timer );
        Log::Line( "Finished in %.2lf seconds.", elapsed );
    }
    else
    {
        const uint threadCount = 1;

        WriteFileJob jobs[threadCount];

        const size_t blockSize       = file.BlockSize();
        const size_t sizeWrite       = entryCount * sizeof( uint );
        const size_t totalBlocks     = sizeWrite / blockSize;
        const size_t blocksPerThread = totalBlocks / threadCount;
        const size_t sizePerThread   = blocksPerThread * blockSize;

        const size_t trailingSize    = sizeWrite - (sizePerThread * threadCount);

        byte* buf = (byte*)buffer;

        for( uint i = 0; i < threadCount; i++ )
        {
            WriteFileJob& job = jobs[i];

            job.filePath = filePath;
            job.success  = false;
            job.size     = sizePerThread;
            job.offset   = sizePerThread * i;
            job.buffer   = buf + job.offset;
        }

        jobs[threadCount-1].size += trailingSize;
        
        ThreadPool pool( threadCount );

        Log::Line( "Writing to file with %u threads.", threadCount );
        
        auto timer = TimerBegin();
        pool.RunJob( WriteFileJob::Run, jobs, threadCount );
        double elapsed = TimerEnd( timer );

        const double bytesPerSecond = sizeWrite / elapsed;
        Log::Line( "Finished writing to file in %.2lf seconds @ %.2lf MB/s.", elapsed, ((double)bytesPerSecond) BtoMB );
    }
}

