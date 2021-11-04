#include "DiskPlotPhase1.h"
#include "Util.h"
#include "util/Log.h"
#include "algorithm/RadixSort.h"
#include "pos/chacha8.h"
#include "b3/blake3.h"

// Test
#include "io/FileStream.h"
#include "SysHost.h"

#if _DEBUG
    #include "../memplot/DbgHelper.h"
#endif

struct WriteFileJob
{
    const char* filePath;

    size_t      size  ;
    size_t      offset;
    byte*       buffer;
    bool        success;

    static void Run( WriteFileJob* job );
};



template<TableId tableId>
static void ComputeFxForTable( const uint64 bucket, uint32 entryCount, const Pairs pairs,
                               const uint32* yIn, const uint64* metaInA, const uint64* metaInB,
                               uint32* yOut, byte* bucketOut, uint64* metaOutA, uint64* metaOutB );

//-----------------------------------------------------------
DiskPlotPhase1::DiskPlotPhase1( DiskPlotContext& cx )
    : _cx( cx )
    //, _diskQueue( cx.workBuffer, cx.diskFlushSize, (uint)(cx.bufferSizeBytes / cx.diskFlushSize) - 1 )
{
    LoadLTargets();

    ASSERT( cx.tmpPath );
    _diskQueue = new DiskBufferQueue( cx.tmpPath, cx.ioHeap, cx.ioHeapSize, cx.ioThreadCount );
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

///
/// F1 Generation
///
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

    const size_t chunkBufferSize  = cx.writeIntervals[(int)TableId::Table1].fxGen;
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
            const uint32 e0  = ( block[0 ] >> bucketShift ) & 0x3F; ASSERT( e0  <= 0b111111u );
            const uint32 e1  = ( block[1 ] >> bucketShift ) & 0x3F; ASSERT( e1  <= 0b111111u );
            const uint32 e2  = ( block[2 ] >> bucketShift ) & 0x3F; ASSERT( e2  <= 0b111111u );
            const uint32 e3  = ( block[3 ] >> bucketShift ) & 0x3F; ASSERT( e3  <= 0b111111u );
            const uint32 e4  = ( block[4 ] >> bucketShift ) & 0x3F; ASSERT( e4  <= 0b111111u );
            const uint32 e5  = ( block[5 ] >> bucketShift ) & 0x3F; ASSERT( e5  <= 0b111111u );
            const uint32 e6  = ( block[6 ] >> bucketShift ) & 0x3F; ASSERT( e6  <= 0b111111u );
            const uint32 e7  = ( block[7 ] >> bucketShift ) & 0x3F; ASSERT( e7  <= 0b111111u );
            const uint32 e8  = ( block[8 ] >> bucketShift ) & 0x3F; ASSERT( e8  <= 0b111111u );
            const uint32 e9  = ( block[9 ] >> bucketShift ) & 0x3F; ASSERT( e9  <= 0b111111u );
            const uint32 e10 = ( block[10] >> bucketShift ) & 0x3F; ASSERT( e10 <= 0b111111u );
            const uint32 e11 = ( block[11] >> bucketShift ) & 0x3F; ASSERT( e11 <= 0b111111u );
            const uint32 e12 = ( block[12] >> bucketShift ) & 0x3F; ASSERT( e12 <= 0b111111u );
            const uint32 e13 = ( block[13] >> bucketShift ) & 0x3F; ASSERT( e13 <= 0b111111u );
            const uint32 e14 = ( block[14] >> bucketShift ) & 0x3F; ASSERT( e14 <= 0b111111u );
            const uint32 e15 = ( block[15] >> bucketShift ) & 0x3F; ASSERT( e15 <= 0b111111u );

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

            queue.WriteBuckets( FileId::Y0, buckets, sizes );
            queue.WriteBuckets( FileId::X , xBuffer, sizes );
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
                queue.WriteFile( FileId::Y0, i, yRemainder, fileBlockSize );
                queue.WriteFile( FileId::X , i, xRemainder, fileBlockSize );
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

        queue.WriteFile( FileId::Y0, i, yBuffer, size );
        queue.WriteFile( FileId::X , i, xBuffer, size );
    }
}


///
/// Forward Propagate Tables
///
//-----------------------------------------------------------
void DiskPlotPhase1::ForwardPropagate()
{
    DiskBufferQueue& ioDispatch  = *_diskQueue;
    ThreadPool&      threadPool  = *_cx.threadPool;
    const uint       threadCount = _cx.threadCount;

    uint   maxBucketCount = 0;
    size_t maxBucketSize  = 0;

    // Find the largest bucket so that we can reserve buffers of its size
    for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
        maxBucketCount = std::max( maxBucketCount, _bucketCounts[i] );

    ASSERT( maxBucketCount <= BB_DP_MAX_ENTRIES_PER_BUCKET );

    maxBucketCount = BB_DP_MAX_ENTRIES_PER_BUCKET;
    maxBucketSize = maxBucketCount * sizeof( uint32 );
    _maxBucketCount =  maxBucketCount;



    // #TODO: We need to have a maximum size here, and just allocate that.
    //        we don't want to allocate per-table as we might not be able to do
    //        so due to fragmentation and the other buffers allocated after this big chunk.
    //        Therefore, it's better to have a set size
    // 
    // #TODO: Create a separate heap... Grab the section out of the working heap we need, and then create
    //        a new heap strictly with the space that will go to IO. No need to have these show up in the 
    //        heap as it will slow down deallocation.
    // 
    // Allocate buffers we need for forward propagation
//     DoubleBuffer bucketBuffers;

    Bucket bucket;
    _bucket = &bucket;

    {
        const size_t fileBlockSize = ioDispatch.BlockSize();

        const size_t ySize       = RoundUpToNextBoundary( maxBucketCount * sizeof( uint32 ) * 2, (int)fileBlockSize );
        const size_t sortKeySize = RoundUpToNextBoundary( maxBucketCount * sizeof( uint32 )    , (int)fileBlockSize );
        const size_t metaSize    = RoundUpToNextBoundary( maxBucketCount * sizeof( uint64 ) * 4, (int)fileBlockSize );
        const size_t pairsLSize  = RoundUpToNextBoundary( maxBucketCount * sizeof( uint32 )    , (int)fileBlockSize );
        const size_t pairsRSize  = RoundUpToNextBoundary( maxBucketCount * sizeof( uint16 )    , (int)fileBlockSize );
        const size_t groupsSize  = RoundUpToNextBoundary( ( maxBucketCount + threadCount * 2 ) * sizeof( uint32), (int)fileBlockSize );

        const size_t totalSize = ySize + sortKeySize + metaSize + pairsLSize + pairsRSize + groupsSize;

        Log::Line( "Reserving %.2lf MiB for forward propagation.", (double)totalSize BtoMB );

        // Temp test:
        bucket.yTmp     = bbvirtalloc<uint32>( ySize    );
        bucket.metaATmp = bbvirtalloc<uint64>( metaSize );
        bucket.metaBTmp = bucket.metaATmp + maxBucketCount;

        bucket.fpBuffer = _cx.heapBuffer;

        byte* ptr = bucket.fpBuffer;

        bucket.y0 = (uint32*)ptr; 
        bucket.y1 = bucket.y0 + maxBucketCount;
        ptr += ySize;

        bucket.sortKey = (uint32*)ptr;
        ptr += sortKeySize;

        bucket.metaA0 = (uint64*)ptr;
        bucket.metaA1 = bucket.metaA0 + maxBucketCount;
        bucket.metaB0 = bucket.metaA1 + maxBucketCount;
        bucket.metaB1 = bucket.metaB0 + maxBucketCount;
        ptr += metaSize;
        ASSERT( ptr == (byte*)( bucket.metaB1 + maxBucketCount ) );


        bucket.pairs.left = (uint32*)ptr;
        ptr += pairsLSize;

        bucket.pairs.right = (uint16*)ptr;
        ptr += pairsRSize;

        bucket.groupBoundaries = (uint32*)ptr;

        // The remainder for the work heap is used to write as fx disk write buffers 
    }

    /// Propagate to each table
    for( TableId table = TableId::Table2; table <= TableId::Table7; table++ )
    {
        const bool isEven = ( (uint)table ) & 1;

        bucket.yFileId     = isEven ? FileId::Y0       : FileId::Y1;
        bucket.metaAFileId = isEven ? FileId::META_A_1 : FileId::META_A_0;
        bucket.metaBFileId = isEven ? FileId::META_B_1 : FileId::META_B_0;

        // Seek buckets to the start and load the first y bucket
       
       
        Log::Line( "Forward propagating to table %d...", (int)table + 1 );
        const auto tableTimer = TimerBegin();

        switch( table )
        {
            case TableId::Table2: ForwardPropagateTable<TableId::Table2>(); break;
            case TableId::Table3: ForwardPropagateTable<TableId::Table3>(); break;
            case TableId::Table4: ForwardPropagateTable<TableId::Table4>(); break;
            case TableId::Table5: ForwardPropagateTable<TableId::Table5>(); break;
            case TableId::Table6: ForwardPropagateTable<TableId::Table6>(); break;
            case TableId::Table7: ForwardPropagateTable<TableId::Table7>(); break;
        }

        const double tableElapsed = TimerEnd( tableTimer );
        Log::Line( "Finished forward propagating table %d in %.2lf seconds.", (int)table + 1, tableElapsed );
    }
}

//-----------------------------------------------------------
template<TableId tableId>
void DiskPlotPhase1::ForwardPropagateTable()
{
    DiskPlotContext& cx          = _cx;
    DiskBufferQueue& ioDispatch  = *_diskQueue;
    Bucket&          bucket      = *_bucket;
    const uint       threadCount = _cx.threadCount;

    constexpr size_t MetaInASize = TableMetaIn<tableId>::SizeA;
    constexpr size_t MetaInBSize = TableMetaIn<tableId>::SizeB;

    // Set the correct file id, given the table (we swap between them for each table)
    {
        const bool isEven = ( (uint)tableId ) & 1;

        bucket.yFileId     = isEven ? FileId::Y0       : FileId::Y1;
        bucket.metaAFileId = isEven ? FileId::META_A_1 : FileId::META_A_0;
        bucket.metaBFileId = isEven ? FileId::META_B_1 : FileId::META_B_0;

        if constexpr ( tableId == TableId::Table2 )
        {
            bucket.metaAFileId = FileId::X;
        }
    }

    // Load first bucket
    ioDispatch.SeekBucket( FileId::Y0, 0, SeekOrigin::Begin );
    ioDispatch.SeekBucket( FileId::Y1, 0, SeekOrigin::Begin );

    if constexpr( tableId == TableId::Table2 )
    {
        ioDispatch.SeekBucket( FileId::X, 0, SeekOrigin::Begin );
    }
    else
    {
        ioDispatch.SeekBucket( FileId::META_A_0, 0, SeekOrigin::Begin );
        ioDispatch.SeekBucket( FileId::META_A_1, 0, SeekOrigin::Begin );
        ioDispatch.SeekBucket( FileId::META_B_0, 0, SeekOrigin::Begin );
        ioDispatch.SeekBucket( FileId::META_B_1, 0, SeekOrigin::Begin );
    }
    ioDispatch.CommitCommands();

    ioDispatch.ReadFile( bucket.yFileId, 0, bucket.y0, _bucketCounts[0] * sizeof( uint32 ) );
    ioDispatch.AddFence( bucket.frontFence );

    ioDispatch.ReadFile( bucket.metaAFileId, 0, bucket.metaA0, _bucketCounts[0] * MetaInASize );

    if constexpr ( MetaInBSize > 0 )
    {
        ioDispatch.ReadFile( bucket.metaBFileId, 0, bucket.metaB0, _bucketCounts[0] * MetaInBSize );
    }
    
    // Fence for metadata
    ioDispatch.AddFence( bucket.frontFence );

    // #TODO: That using a single fence twice can actually cause a race condition.
    // //     Although extremely unlikely, it could be the case that the 2 read commands happen
    //        before this first wait Wait(). Meaning we will be waiting forever on the second one.
    //        Therefore we need to switch to Semaphores for waits. Or whatever other special mechanism
    //        we will use to ensure we can notify the other threads that they should wait on IO as well
    //        whilst the control threads looks waits on a buffer.
    // Dispatch read commands and wait for y to be loaded
    ioDispatch.CommitCommands();
    bucket.frontFence.Wait();

    for( uint bucketIdx = 0; bucketIdx < BB_DP_BUCKET_COUNT; bucketIdx++ )
    {
        Log::Line( " Processing bucket %-2u", bucketIdx );

        const uint entryCount = _bucketCounts[bucketIdx];
        ASSERT( entryCount < _maxBucketCount );

        // Read the next bucket in the background if we're not at the last bucket
        const uint nextBucketIdx = bucketIdx + 1;

        if( nextBucketIdx < BB_DP_BUCKET_COUNT )
        {
            const size_t nextBufferCount = _bucketCounts[nextBucketIdx];

            ioDispatch.ReadFile( bucket.yFileId    , nextBucketIdx, bucket.y1    , nextBufferCount * sizeof( uint32 ) );

            // #TODO: Maybe we should just allocate .5 GiB more for the temp buffers?
            //        for now, try it this way.
            // Don't load the metadata yet, we will use the metadata back buffer as our temporary buffer for sorting
            ioDispatch.CommitCommands();

            ioDispatch.ReadFile( bucket.metaAFileId, nextBucketIdx, bucket.metaA1, nextBufferCount * MetaInASize );

            if constexpr ( MetaInBSize > 0 )
            {
                ioDispatch.ReadFile( bucket.metaBFileId, nextBucketIdx, bucket.metaB1, nextBufferCount * MetaInBSize );
            }

            ioDispatch.AddFence( bucket.backFence );
//             ioDispatch.CommitCommands();
        }
        else
        {
            // Make sure we don't wait at the end of the loop since we don't 
            // have any background bucket loading.
            bucket.backFence.Signal();
        }

        // Sort our current bucket
        {
            Log::Line( "  Sorting bucket." );
            auto timer = TimerBegin();

            uint32* sortKey = bucket.sortKey;

            if constexpr( tableId == TableId::Table2 )
            {
                // No sort key needed for table 1, just sort x along with y
                sortKey = (uint32*)bucket.metaA0;
            }

            uint32* yTemp       = (uint32*)bucket.metaA1;
            uint32* sortKeyTemp = yTemp + entryCount;

//             RadixSort256::SortWithKey<BB_MAX_JOBS>( *cx.threadPool, bucket.y0, yTemp, sortKey, sortKeyTemp, entryCount );
            RadixSort256::Sort<BB_MAX_JOBS>( *cx.threadPool, bucket.y0, yTemp, entryCount );

            double elapsed = TimerEnd( timer );

            // OK to load next (back) metadata buffer now (see comment above)
            if( nextBucketIdx < BB_DP_BUCKET_COUNT )
                ioDispatch.CommitCommands();

            #if _DEBUG
                ASSERT( DbgVerifyGreater( entryCount, bucket.y0 ) );
            #endif

            Log::Line( "  Sorted bucket in %.2lf seconds.", elapsed );
        }

        // Ensure metadata has been loaded on the first bucket
        if( bucketIdx == 0 )
            bucket.frontFence.Wait();

        // Sort metadata with the key
        if constexpr( tableId > TableId::Table2 )
        {
            // #TODO: This
        }

        // Scan for BC groups & match
        GroupInfo groupInfos[BB_MAX_JOBS];
        uint32 totalMatches;
        
//         Pairs pairs[BB_MAX_JOBS];

        {
            // Scan for group boundaries
            const uint32 groupCount = ScanGroups( bucketIdx, bucket.y0, entryCount, bucket.groupBoundaries, BB_DP_MAX_BC_GROUP_PER_BUCKET, groupInfos );
            
            // Produce per-thread matches in meta tmp. It has enough space to hold them.
            // Then move them over to a contiguous buffer.
            uint32* lPairs = (uint32*)bucket.metaATmp;
            uint16* rPairs = (uint16*)( lPairs + BB_DP_MAX_ENTRIES_PER_BUCKET );

            // Match pairs
            const uint32 entriesPerBucket  = (uint32)BB_DP_MAX_ENTRIES_PER_BUCKET;
            const uint32 maxPairsPerThread = entriesPerBucket / threadCount;    // (uint32)( entriesPerBucket / threadCount + BB_DP_XTRA_MATCHES_PER_THREAD );

            for( uint i = 0; i < threadCount; i++ )
            {
                groupInfos[i].pairs.left  = lPairs + i * maxPairsPerThread;
                groupInfos[i].pairs.right = rPairs + i * maxPairsPerThread;
            }
            
            totalMatches = Match( bucketIdx, maxPairsPerThread, bucket.y0, groupInfos );

            // #TODO: Make this multi-threaded... Testing for now
            // Copy matches to a contiguous buffer
            Pairs& pairs = bucket.pairs;
//             pairs.left  = //bbcvirtalloc<uint32>( totalMatches );
//             pairs.right = //bbcvirtalloc<uint16>( totalMatches );

            uint32* lPtr = pairs.left;
            uint16* rPtr = pairs.right;

            for( uint i = 0; i < threadCount; i++ )
            {
                GroupInfo& group = groupInfos[i];
                bbmemcpy_t( lPtr, group.pairs.left, group.entryCount );
                lPtr += group.entryCount;
            }

            for( uint i = 0; i < threadCount; i++ )
            {
                GroupInfo& group = groupInfos[i];
                bbmemcpy_t( rPtr, group.pairs.right, group.entryCount );
                rPtr += group.entryCount;
            }
        }

        // Generate fx values
        GenFxForTable<tableId>( 
            bucketIdx, totalMatches, bucket.pairs,
            bucket.y0, bucket.yTmp, (byte*)bucket.sortKey,    // #TODO: Change this, for now use sort key buffer
            bucket.metaA0, bucket.metaB0,
            bucket.metaATmp, bucket.metaBTmp );
        //for( uint threadIdx = 0; threadIdx < threadCount; threadIdx++ )
        //{
        //    GroupInfo& group = groupInfos[threadIdx];
        //}

        // Ensure the next bucket has finished loading
        bucket.backFence.Wait();

        // Swap are front/back buffers
        std::swap( bucket.y0    , bucket.y1     );
        std::swap( bucket.metaA0, bucket.metaA1 );
        std::swap( bucket.metaB0, bucket.metaB1 );
    }
}

///
/// Group Matching
///
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
uint32 DiskPlotPhase1::Match( uint bucketIdx, uint maxPairsPerThread, const uint32* yBuffer, GroupInfo groupInfos[BB_MAX_JOBS] )
{
    Log::Line( "  Matching groups." );

    auto&      cx          = _cx;
    const uint threadCount = cx.threadCount;

    MTJobRunner<MatchJob> jobs( *cx.threadPool );

    for( uint i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];

        job.yBuffer         = yBuffer;
        job.bucketIdx       = bucketIdx;
        job.maxPairCount    = maxPairsPerThread;
        job.pairCount       = 0;
        job.groupInfo       = &groupInfos[i];
        job.copyLDst        = nullptr;
        job.copyRDst        = nullptr;
    }

    const double elapsed = jobs.Run();

    uint32 matches = jobs[0].pairCount;
    for( uint i = 1; i < threadCount; i++ )
        matches += jobs[i].pairCount;

    Log::Line( "  Found %u matches in %.2lf seconds.", matches, elapsed );
    return matches;
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

    Pairs  pairs     = groupInfo->pairs;
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
    this->pairCount             = pairCount;
    this->groupInfo->entryCount = pairCount;
}

///
/// Fx Generation
///
//-----------------------------------------------------------
template<TableId tableId>
void DiskPlotPhase1::GenFxForTable( uint bucketIdx, uint entryCount, const Pairs pairs,
                                    const uint32* yIn, uint32* yOut, byte* bucketIdOut,
                                    const uint64* metaInA, const uint64* metaInB,
                                    uint64* metaOutA, uint64* metaOutB )
{
    Log::Line( "  Computing Fx..." );
    auto timer = TimerBegin();

    auto& cx = _cx;

    const size_t outMetaSizeA     = TableMetaOut<tableId>::SizeA;
    const size_t outMetaSizeB     = TableMetaOut<tableId>::SizeB;

    const size_t fileBlockSize    = _diskQueue->BlockSize();
    const size_t sizePerEntry     = sizeof( uint32 ) + outMetaSizeA + outMetaSizeB;

    const size_t writeInterval    = cx.writeIntervals[(uint)tableId].fxGen;

    const size_t entriesTotalSize = entryCount * sizePerEntry;
    ASSERT( writeInterval <= entriesTotalSize );

    uint32 entriesPerChunk        = (uint32)( writeInterval / sizePerEntry );
    uint32 chunkCount             = (uint32)( entriesTotalSize / writeInterval );

    const uint32 threadCount      = cx.threadCount;
    const uint32 entriesPerThread = entriesPerChunk / threadCount;

    entriesPerChunk = entriesPerThread * threadCount;
    uint32       trailingEntries  = entryCount - ( entriesPerChunk * chunkCount );

    while( trailingEntries >= entriesPerChunk )
    {
        chunkCount++;
        trailingEntries -= entriesPerChunk;
    }

    // Add trailing entries as a trailing chunk
    const uint32 lastChunkEntries = trailingEntries / threadCount;

    // Remove that from the trailing entries.
    // This guarantees that any trailing entries will be <= threadCount
    trailingEntries -= lastChunkEntries * threadCount;

    MTJobRunner<FxJob> jobs( *cx.threadPool );

    for( uint i = 0; i < threadCount; i++ )
    {
        FxJob& job = jobs[i];

        job.diskQueue            = _diskQueue;
        job.tableId              = tableId;
        job.bucketIdx            = bucketIdx;
        job.entryCount           = entriesPerThread;
        job.chunkCount           = chunkCount;
        job.entriesPerChunk      = entriesPerChunk;
        job.trailingChunkEntries = lastChunkEntries;
        
        const size_t offset = entriesPerThread * i;
        job.pairs      = pairs;
        job.pairs.left  += offset;
        job.pairs.right += offset;

        job.yIn         = yIn;
        job.metaInA     = metaInA;
        job.metaInB     = metaInB;
        job.yOut        = yOut        + offset;
        job.metaOutA    = metaOutA    + offset;
        job.metaOutB    = metaOutB    + offset;
        job.bucketIdOut = bucketIdOut + offset;
    }

    // Calculate Fx
    jobs.Run();

    // #TODO: Calculate trailing entries here (they are less than the thread count)
    //        if we have any.
    if( trailingEntries )
    {
        // Call ComputeFxForTable
    }

    auto elapsed = TimerEnd( timer );
    Log::Line( "  Finished computing Fx in %.4lf seconds.", elapsed );
}

//-----------------------------------------------------------
void FxJob::Run()
{
    ASSERT( this->entriesPerChunk == this->entryCount * this->_jobCount );
    switch( tableId )
    {
        case TableId::Table1: RunForTable<TableId::Table1>(); return;
        case TableId::Table2: RunForTable<TableId::Table2>(); return;
        case TableId::Table3: RunForTable<TableId::Table3>(); return;
        case TableId::Table4: RunForTable<TableId::Table4>(); return;
        case TableId::Table5: RunForTable<TableId::Table5>(); return;
        case TableId::Table6: RunForTable<TableId::Table6>(); return;
        case TableId::Table7: RunForTable<TableId::Table7>(); return;
        
        default:
            ASSERT( 0 );
        break;
    }
}

//-----------------------------------------------------------
template<TableId tableId>
void FxJob::RunForTable()
{
    using TMetaA = TableMetaOut<tableId>::MetaA;
    using TMetaB = TableMetaOut<tableId>::MetaB;

    const uint32   entryCount       = this->entryCount;
    const uint32   chunkCount       = this->chunkCount;
//     const uint32   entriesPerChunk  = this->entriesPerChunk;
    const uint64   bucket           = ( (uint64)this->bucketIdx ) << 32;

    Pairs          lrPairs          = this->pairs;
    const uint64*  inMetaA          = this->metaInA;
    const uint64*  inMetaB          = this->metaInB;
    const uint32*  inY              = this->yIn;
    
    uint32*        outY             = this->yOut;
    byte*          outBucketId      = this->bucketIdOut;
    uint64*        outMetaA         = this->metaOutA;
    uint64*        outMetaB         = this->metaOutB;

    for( uint chunk = 0; chunk < chunkCount; chunk++ )
    {
//         auto timer = TimerBegin();

        ComputeFxForTable<tableId>( bucket, entryCount, lrPairs, inY, inMetaA, inMetaB,
                                    outY, outBucketId, outMetaA, outMetaB );

//         double elapsed = TimerEnd( timer );
//         Trace( "Finished chunk %-2u in %.2lf seconds", chunk, elapsed );
//         this->SyncThreads();
        
        SortToBucket<tableId, TMetaA, TMetaB>(
            entryCount, outBucketId, outY, (TMetaA*)outMetaA, (TMetaB*)outMetaB
        );

        lrPairs.left  += entryCount;
        lrPairs.right += entryCount;
    }

    const uint32 trailingChunkEntries = this->trailingChunkEntries;
    if( trailingChunkEntries )
    {
        ComputeFxForTable<tableId>( bucket, trailingChunkEntries, lrPairs, inY, inMetaA, inMetaB,
                                    outY, outBucketId, outMetaA, outMetaB );

        SortToBucket<tableId, TMetaA, TMetaB>(
            trailingChunkEntries, outBucketId, outY, (TMetaA*)outMetaA, (TMetaB*)outMetaB
        );

        if( this->IsControlThread() )
        {

        }
    }
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"

//-----------------------------------------------------------
template<TableId tableId>
inline 
void ComputeFxForTable( const uint64 bucket, uint32 entryCount, const Pairs pairs, 
                        const uint32* yIn, const uint64* metaInA, const uint64* metaInB, 
                        uint32* yOut, byte* bucketOut, uint64* metaOutA, uint64* metaOutB )
{
    constexpr size_t metaKMultiplierIn  = TableMetaIn <tableId>::Multiplier;
    constexpr size_t metaKMultiplierOut = TableMetaOut<tableId>::Multiplier;

    // Helper consts
    // Table 7 (identified by 0 metadata output) we don't have k + kExtraBits sized y's.
    // so we need to shift by 32 bits, instead of 26.
    constexpr uint extraBitsShift = tableId == TableId::Table7 ? 0 : kExtraBits;

    constexpr uint   shiftBits   = metaKMultiplierOut == 0 ? 0 : kExtraBits;
    constexpr uint   k           = _K;
    constexpr uint32 ySize       = k + kExtraBits;         // = 38
    constexpr uint32 yShift      = 64u - (k + shiftBits);  // = 26 or 32
    constexpr size_t metaSize    = k * metaKMultiplierIn;
    constexpr size_t metaSizeLR  = metaSize * 2;
    constexpr size_t bufferSize  = CDiv( ySize + metaSizeLR, 8u );

    // Bucket for extending y
//     const uint64 bucket = ( (uint64)bucketIdx ) << 32;

    // Meta extraction
    uint64 l0, l1, r0, r1;

    // Hashing
    uint64 input [5];       // y + L + R
    uint64 output[4];       // blake3 hashed output

    static_assert( bufferSize <= sizeof( input ), "Invalid fx input buffer size." );

    blake3_hasher hasher;

    #if _DEBUG
        uint64 prevY = bucket | yIn[pairs.left[0]];
    #endif

    for( uint i = 0; i < entryCount; i++ )
    {
        const uint32 left  = pairs.left[i];
        const uint32 right = left + pairs.right[i];

        const uint64 y     = bucket | yIn[left];

        #if _DEBUG
            ASSERT( y >= prevY );
            prevY = y;
        #endif

        // Extract metadata
        if constexpr( metaKMultiplierIn == 1 )
        {
            l0 = reinterpret_cast<const uint32*>( metaInA )[left ];
            r0 = reinterpret_cast<const uint32*>( metaInA )[right];

            input[0] = Swap64( y  << 26 | l0 >> 6  );
            input[1] = Swap64( l0 << 58 | r0 << 26 );
        }
        else if constexpr( metaKMultiplierIn == 2 )
        {
            l0 = metaInA[left ];
            r0 = metaInA[right];

            input[0] = Swap64( y  << 26 | l0 >> 38 );
            input[1] = Swap64( l0 << 26 | r0 >> 38 );
            input[2] = Swap64( r0 << 26 );
        }
        else if constexpr( metaKMultiplierIn == 3 )
        {
            l0 = metaInA[left];
            l1 = reinterpret_cast<const uint32*>( metaInB )[left ];
            r0 = metaInA[right];
            r1 = reinterpret_cast<const uint32*>( metaInB )[right];
        
            input[0] = Swap64( y  << 26 | l0 >> 38 );
            input[1] = Swap64( l0 << 26 | l1 >> 6  );
            input[2] = Swap64( l1 << 58 | r0 >> 6  );
            input[3] = Swap64( r0 << 58 | r1 << 26 );
        }
        else if constexpr( metaKMultiplierIn == 4 )
        {
            l0 = metaInA[left ];
            l1 = metaInB[left ];
            r0 = metaInA[right];
            r1 = metaInB[right];

            input[0] = Swap64( y  << 26 | l0 >> 38 );
            input[1] = Swap64( l0 << 26 | l1 >> 38 );
            input[2] = Swap64( l1 << 26 | r0 >> 38 );
            input[3] = Swap64( r0 << 26 | r1 >> 38 );
            input[4] = Swap64( r1 << 26 );
        }

        // Hash input
        blake3_hasher_init( &hasher );
        blake3_hasher_update( &hasher, input, bufferSize );
        blake3_hasher_finalize( &hasher, (uint8_t*)output, sizeof( output ) );

        uint64 fx = Swap64( *output ) >> yShift;

        yOut[i] = (uint32)fx;

        if constexpr( tableId != TableId::Table7 )
        {
            // Store the bucket id for this y value
            bucketOut[i] = (byte)( fx >> 32 );
        }

        // Calculate output metadata
        if constexpr( metaKMultiplierOut == 2 && metaKMultiplierIn == 1 )
        {
            metaOutA[i] = l0 << 32 | r0;
        }
        else if constexpr ( metaKMultiplierOut == 2 && metaKMultiplierIn == 3 )
        {
            const uint64 h0 = Swap64( output[0] );
            const uint64 h1 = Swap64( output[1] );

            metaOutA[0] = h0 << ySize | h1 >> 26;
        }
        else if constexpr ( metaKMultiplierOut == 3 )
        {
            const uint64 h0 = Swap64( output[0] );
            const uint64 h1 = Swap64( output[1] );
            const uint64 h2 = Swap64( output[2] );

            metaOutA[i] = h0 << ySize | h1 >> 26;
            reinterpret_cast<uint32*>( metaOutB )[i] = (uint32)( ((h1 << 6) & 0xFFFFFFC0) | h2 >> 58 );
        }
        else if constexpr( metaKMultiplierOut == 4 && metaKMultiplierIn == 2 )
        {
            metaOutA[i] = l0;
            metaOutB[i] = r0;
        }
        else if constexpr ( metaKMultiplierOut == 4 && metaKMultiplierIn != 2 )
        {
            const uint64 h0 = Swap64( output[0] );
            const uint64 h1 = Swap64( output[1] );
            const uint64 h2 = Swap64( output[2] );

            metaOutA[i] = h0 << ySize | h1 >> 26;
            metaOutB[i] = h1 << 38    | h2 >> 26;
        }
    }
}

//-----------------------------------------------------------
template<TableId tableId, typename TMetaA, typename TMetaB>
inline
void FxJob::SortToBucket( uint entryCount, const byte* bucketIndices, const uint32* inY, const TMetaA* inMetaA, const TMetaB* inMetaB )
{
    const bool isEven = static_cast<uint>( tableId ) & 1;

    const FileId yFileId     = isEven ? FileId::Y1       : FileId::Y0;
    const FileId metaAFileId = isEven ? FileId::META_A_0 : FileId::META_A_1;
    const FileId metaBFileId = isEven ? FileId::META_B_0 : FileId::META_B_1;

    const size_t metaSizeA = TableMetaOut<tableId>::SizeA;
    const size_t metaSizeB = TableMetaOut<tableId>::SizeB;

    DiskBufferQueue& queue = *this->diskQueue;

    const size_t fileBlockSize = queue.BlockSize();

    uint32* ySizes       = nullptr;
    uint32* metaASizes   = nullptr;
    uint32* metaBSizes   = nullptr;
    uint32* yBuckets     = nullptr;
    TMetaA* metaABuckets = nullptr;
    TMetaB* metaBBuckets = nullptr;

    uint counts      [BB_DP_BUCKET_COUNT];
    uint pfxSum      [BB_DP_BUCKET_COUNT];
    uint bucketCounts[BB_DP_BUCKET_COUNT];

    // Count our buckets
    memset( counts, 0, sizeof( counts ) );
    for( uint i = 0; i < entryCount; i++ )
    {
        ASSERT( bucketIndices[i] <= ( 0b111111u ) );
        counts[bucketIndices[i]] ++;
    }

    this->bucketCounts = bucketCounts;

    CalculatePrefixSum( counts, pfxSum );

    // Grab a buffer from the queue
    if( this->LockThreads() )
    {
        uint totalCount = 0;
        for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
            totalCount += bucketCounts[i];

        ASSERT( totalCount == this->entriesPerChunk );

        ySizes   = (uint32*)queue.GetBuffer( BB_DP_BUCKET_COUNT * sizeof( uint32 ) );
        yBuckets = (uint32*)queue.GetBuffer( sizeof( uint32 ) * totalCount );

        if constexpr ( metaSizeA > 0 )
        {
            metaASizes   = (uint32*)queue.GetBuffer( ( sizeof( uint32 ) * BB_DP_BUCKET_COUNT ) );
            metaABuckets = (TMetaA*)queue.GetBuffer( sizeof( TMetaA ) * totalCount );
        }

        if constexpr ( metaSizeB > 0 )
        {
            metaBSizes   = (uint32*)queue.GetBuffer( ( sizeof( uint32 ) * BB_DP_BUCKET_COUNT ) );
            metaBBuckets = (TMetaB*)queue.GetBuffer( sizeof( TMetaB ) * totalCount );
        }

        _bucketY     = yBuckets;
        _bucketMetaA = metaABuckets;
        _bucketMetaB = metaBBuckets;
        this->ReleaseThreads();
    }
    else
    {
        // #TODO: We need to wait for release and sleep/block when
        //        if the control thread starts blocking because it
        //        was not able to secure a buffer for writing
        this->WaitForRelease();

        yBuckets     = (uint32*)GetJob( 0 )._bucketY;
        metaABuckets = (TMetaA*)GetJob( 0 )._bucketMetaA;
        metaBBuckets = (TMetaB*)GetJob( 0 )._bucketMetaB;
    }

    // #TODO: Unroll this a bit?
    // Distribute values into buckets at each thread's given offset
    for( uint i = 0; i < entryCount; i++ )
    {
        const uint32 dstIdx = --pfxSum[bucketIndices[i]];

        yBuckets[dstIdx] = yIn[i];

        if constexpr ( metaSizeA > 0 )
        {
            metaABuckets[dstIdx] = inMetaA[i];
        }

        if constexpr ( metaSizeB > 0 )
        {
            metaBBuckets[dstIdx] = inMetaB[i];
        }
    }

    // Write buckets to disk
    if( this->LockThreads() )
    {
        // Calculate the disk block-aligned size
        // #TODO: Don't do this if not using direct IO?
        const uint32* bucketCounts = this->bucketCounts;

        for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
            ySizes[i] = (uint32)( ( bucketCounts[i] * sizeof( uint32 ) ) / fileBlockSize * fileBlockSize );

        queue.WriteBuckets( yFileId, yBuckets, ySizes );

        if constexpr ( metaSizeA > 0 )
        {
            for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
                metaASizes[i] = (uint32)( ( bucketCounts[i] * sizeof( TMetaA ) ) / fileBlockSize * fileBlockSize );

            queue.WriteBuckets( metaAFileId, metaABuckets, metaASizes );
        }

        if constexpr ( metaSizeB > 0 )
        {
            for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
                metaBSizes[i] = (uint32)( ( bucketCounts[i] * sizeof( TMetaB ) ) / fileBlockSize * fileBlockSize );

            queue.WriteBuckets( metaBFileId, metaBBuckets, metaBSizes );
        }

        queue.CommitCommands();

        // #TODO: I don't think we need to wait for SaveBlockRemainders to release our
        //        buffers anymore.. We shouldn't be using them. But check the reasoning in f1gen to be sure.
        SaveBlockRemainders( yFileId, yOut, ySizes, _yRemainderSizes, _yRemainders );
        queue.ReleaseBuffer( ySizes   );
        queue.ReleaseBuffer( yBuckets );
        queue.CommitCommands();

        if constexpr( metaSizeA > 0 )
        {
            SaveBlockRemainders( metaAFileId, metaASizes, metaOutA, _metaARemainderSizes, _metaARemainders );
            queue.ReleaseBuffer( metaASizes   );
            queue.ReleaseBuffer( metaABuckets );
            queue.CommitCommands();
        }

        if constexpr( metaSizeB > 0 )
        {
            SaveBlockRemainders( metaBFileId, metaBSizes, metaOutB, _metaBRemainderSizes, _metaBRemainders );
            queue.ReleaseBuffer( metaBSizes   );
            queue.ReleaseBuffer( metaBBuckets );
            queue.CommitCommands();
        }

        this->ReleaseThreads();
    }
    else
        this->WaitForRelease();
}

//-----------------------------------------------------------
template<typename T>
FORCE_INLINE
void FxJob::SaveBlockRemainders( FileId fileId, const uint32* sizes, const T* buffer, uint32* remainderSizes, DoubleBuffer* remainderBuffers )
{

}


//-----------------------------------------------------------
template<typename TJob>
void BucketJob<TJob>::CalculatePrefixSum( const uint32 counts[BB_DP_BUCKET_COUNT], uint32 pfxSum[BB_DP_BUCKET_COUNT] )
{
    const size_t copySize = sizeof( uint32 ) * BB_DP_BUCKET_COUNT;
    const uint   jobId    = this->JobId();

    this->counts = counts;
    SyncThreads();

    const uint jobCount = this->JobCount();

    // Add up all of the jobs counts
    memcpy( pfxSum, GetJob( 0 ).counts, copySize );

    for( uint t = 1; t < jobCount; t++ )
    {
        const uint* tCounts = GetJob( t ).counts;

        for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
            pfxSum[i] += tCounts[i];
    }

    // If we're the control thread, retain the total bucket count for this chunk
//     uint32 totalCount = 0;
    if( this->IsControlThread() )
    {
        memcpy( this->bucketCounts, pfxSum, copySize );
    }
        
    // #TODO: Only do this for the control thread
//     for( uint j = 0; j < BB_DP_BUCKET_COUNT; j++ )
//         totalCount += pfxSum[j];

    // Calculate the prefix sum for this thread
    for( uint i = 1; i < BB_DP_BUCKET_COUNT; i++ )
        pfxSum[i] += pfxSum[i-1];

    // Subtract the count from all threads after ours
    for( uint t = jobId+1; t < jobCount; t++ )
    {
        const uint* tCounts = GetJob( t ).counts;

        for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
            pfxSum[i] -= tCounts[i];
    }
}


#pragma GCC diagnostic pop

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


