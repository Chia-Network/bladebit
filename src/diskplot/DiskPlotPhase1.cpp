#include "DiskPlotPhase1.h"
#include "pos/chacha8.h"
#include "Util.h"
#include "util/Log.h"


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
    _diskQueue = new DiskBufferQueue( "E:/", cx.workBuffer, cx.bufferSizeBytes, cx.diskFlushSize, cx.diskQueueThreadCount );
}

//-----------------------------------------------------------
void DiskPlotPhase1::Run()
{
    GenF1();
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
    const size_t countsBufferSize = sizeof( uint32 ) * BB_DP_BUCKET_COUNT;

    // "-1" because we reserve 1 chunk buffer for blocks and count buffers
    const size_t chunkBufferSize  = cx.diskFlushSize - countsBufferSize;
    const uint32 blocksPerChunk   = (uint32)( chunkBufferSize / ( kF1BlockSize * 2 ) );
    const uint32 chunkCount       = CDivT( blockCount, blocksPerChunk );                                // How many chunks we need to process
    const uint32 lastChunkBlocks  = blocksPerChunk - ( ( chunkCount * blocksPerChunk ) - blockCount );  // Last chunk might not need to process all blocks

    // Threads operate on a chunk at a time.
    const uint32 threadCount      = pool.ThreadCount();
    const uint32 blocksPerThread  = blocksPerChunk / threadCount;
    
    uint32 trailingBlocks = blocksPerChunk - ( blocksPerThread * threadCount );

    // #TODO: Ensure each thread has at least one block.
    ASSERT( blocksPerThread > 0 );

    uint  x       = 0;
    byte* blocks  = _diskQueue->GetBuffer();
    byte* xBuffer = blocks + blocksPerChunk * kF1BlockSize;

    uint32 bucketCounts[BB_DP_BUCKET_COUNT];

    MTJobRunner<GenF1Job> f1Job( pool );

    for( uint i = 0; i < threadCount; i++ )
    {
        GenF1Job& job = f1Job[i];
        job.key          = key;
        job.chunkCount   = chunkCount;
        job.blockCount   = blocksPerThread;
        job.x            = x;

        job.buffer       = blocks;
        job.xBuffer      = (uint32*)xBuffer;

        job.counts       = nullptr;
        job.bucketCounts = nullptr;
        job.buckets      = nullptr;

        job.diskQueue    = _diskQueue;


        if( trailingBlocks > 0 )
        {
            job.blockCount++;
            trailingBlocks--;
        }

        x       += job.blockCount * entriesPerBlock;
        blocks  += job.blockCount * kF1BlockSize;
        xBuffer += job.blockCount * kF1BlockSize;
    }

    f1Job[0].bucketCounts = bucketCounts;

    Log::Line( "Generating f1..." );
    const double elapsed = f1Job.Run();
    Log::Line( "Finished f1 generation in %.2lf seconds. ", elapsed );
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

    DiskBufferQueue& queue  = *this->diskQueue;
    
    uint32 x = this->x;

    chacha8_ctx chacha;
    chacha8_keysetup( &chacha, key, 256, NULL );

    uint counts[BB_DP_BUCKET_COUNT];
    uint pfxSum[BB_DP_BUCKET_COUNT];

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

        uint32 totalCount = 0;
        // If we're job 0, retain the total bucket count
        if( jobId == 0 )
        {
            memcpy( this->bucketCounts, pfxSum, sizeof( pfxSum ) );

            for( uint j = 0; j < BB_DP_BUCKET_COUNT; j++ )
                totalCount += pfxSum[j];
        }

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
            sizes   = (uint32*)queue.GetBuffer();
            buckets = sizes + BB_DP_BUCKET_COUNT;       // First portion of the buffer is saved for the sizes
            xBuffer = buckets = buckets + totalCount;

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
            const uint32* bucketCounts = this->bucketCounts;

            for( uint j = 0; j < BB_DP_BUCKET_COUNT; j++ )
                sizes[j] = bucketCounts[j] * sizeof( uint32 );

            // If we're not at our last chunk, we need to shave-off
            // any entries that will not align to the file block size and
            // leave them in our buckets for the next run.
            if( i+1 < chunkCount )
            {
                const size_t blockSize = queue.BlockSize();

                for( uint j = 0; j < BB_DP_BUCKET_COUNT; j++ )
                {
                    const size_t blockAlignedSize = sizes[j] / blockSize * blockSize;
                    const size_t remainderEntries = ( sizes[j] - blockAlignedSize ) / sizeof( uint32 );
                    ASSERT( remainderEntries * sizeof( uint32 ) == sizes[j] - blockAlignedSize );

                    if( remainderEntries )
                    {
                        // Need to copy over the remainder as the beginning of the next buffers
                    }
                }
            }

            queue.WriteBuckets( FileId::Y, (byte*)buckets, sizes );
            queue.WriteBuckets( FileId::X, (byte*)xBuffer, sizes );
            queue.ReleaseBuffer( (byte*)sizes );
            queue.CommitCommands();

            this->ReleaseThreads();
        }
        else
            this->WaitForRelease();
    }
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

