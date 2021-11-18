#include "F1GenBucketized.h"
#include "diskplot/DiskBufferQueue.h"
#include "threading/ThreadPool.h"
#include "pos/chacha8.h"

//-----------------------------------------------------------
template<bool WriteToDisk, bool SingleThreaded>
void F1GenBucketized::Generate( 
    chacha8_ctx& chacha, 
    uint32  x,
    byte*   blocks,
    uint32  blockCount,
    uint32  entryCount,
    uint32* buckets,
    uint32* xBuffer,

    // For writing to disk variant
    size_t  fileBlockSize,
    uint32* sizes
    )
{
    const uint64 chachaBlock = ((uint64)x) * _K / kF1BlockSizeBits;

    const uint32 entriesPerBlock   = kF1BlockSize / sizeof( uint32 );
    const uint32 kMinusKExtraBits  = _K - kExtraBits;
    const uint32 bucketShift       = (8u - (uint)kExtraBits);

    const uint32 jobId             = this->JobId();
    const uint32 jobCount          = this->JobCount();

    const size_t fileBlockSize     = queue->BlockSize();

    ASSERT( entryCount <= blockCount * entriesPerBlock );
    
    // Generate chacha blocks
    chacha8_get_keystream( &chacha, chachaBlock, blockCount, blocks );

    // Count how many entries we have per bucket
    uint counts[BB_DP_BUCKET_COUNT];
    uint pfxSum[BB_DP_BUCKET_COUNT];

    memset( counts, 0, sizeof( counts ) );

    const uint32* block = (uint32*)blocks;

    // Count entries per bucket. Only calculate the blocks that have full entries
    const uint32 fullBlockCount  = entryCount / entriesPerBLock;
    const uint32 trailingEntries = blockCount * entriesPerBlock - entryCount;

    for( uint i = 0; i < fullBlockCount; i++ )
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

    // Process trailing entries
    for( uint i = 0; i < trailingEntries; i++ )
        counts[( block[i] >> bucketShift ) & 0x3F] ++;

    // Calculate the prefix sum for our buckets
    if constexpr ( SingleThreaded )
    {
        memcpy( pfxSum, counts, sizeof( counts ) );

        if constexpr ( WriteToDisk )
        {
            // We need to align each count to file block size
            // so that each bucket starts aligned 
            // (we won't write submit these extra false entries)
            pfxSum[i] = RoundUpToNextBoundary( pfxSum[i] * sizeof( uint32 ), (int)fileBlockSize ) / sizeof( uint32 );
        }

        for( uint i = 1; i < BB_DP_BUCKET_COUNT; i++ )
            pfxSum[i] += pfxSum[i-1];
    }
    else
    {
        this->CalculateMultithreadedPredixSum( counts, pfxSum );
    }
    
    // Now we know the offset where we can start storing bucketized y values

    // Grab a buffer from the queue
    if constexpr ( WriteToDisk )
    {
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
            this->WaitForRelease();
            buckets = GetJob( 0 ).buckets;
            xBuffer = GetJob( 0 ).xBuffer;
        }
    }


    // Distribute values into buckets at each thread's given offset
    block = (uint*)blocks;

    for( uint i = 0; i < fullBlockCount; i++ )
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

        // const uint32 refY = 27327;
        // if( buckets[idx0 ] == refY ) BBDebugBreak();
        // if( buckets[idx1 ] == refY ) BBDebugBreak();
        // if( buckets[idx2 ] == refY ) BBDebugBreak();
        // if( buckets[idx3 ] == refY ) BBDebugBreak();
        // if( buckets[idx4 ] == refY ) BBDebugBreak();
        // if( buckets[idx5 ] == refY ) BBDebugBreak();
        // if( buckets[idx6 ] == refY ) BBDebugBreak();
        // if( buckets[idx7 ] == refY ) BBDebugBreak();
        // if( buckets[idx8 ] == refY ) BBDebugBreak();
        // if( buckets[idx9 ] == refY ) BBDebugBreak();
        // if( buckets[idx10] == refY ) BBDebugBreak();
        // if( buckets[idx11] == refY ) BBDebugBreak();
        // if( buckets[idx12] == refY ) BBDebugBreak();
        // if( buckets[idx13] == refY ) BBDebugBreak();
        // if( buckets[idx14] == refY ) BBDebugBreak();
        // if( buckets[idx15] == refY ) BBDebugBreak();

        // if( x + 0  == 2853878795 ) BBDebugBreak();
        // if( x + 1  == 2853878795 ) BBDebugBreak();
        // if( x + 2  == 2853878795 ) BBDebugBreak();
        // if( x + 3  == 2853878795 ) BBDebugBreak();
        // if( x + 4  == 2853878795 ) BBDebugBreak();
        // if( x + 5  == 2853878795 ) BBDebugBreak();
        // if( x + 6  == 2853878795 ) BBDebugBreak();
        // if( x + 7  == 2853878795 ) BBDebugBreak();
        // if( x + 8  == 2853878795 ) BBDebugBreak();
        // if( x + 9  == 2853878795 ) BBDebugBreak();
        // if( x + 10 == 2853878795 ) BBDebugBreak();
        // if( x + 11 == 2853878795 ) BBDebugBreak();
        // if( x + 12 == 2853878795 ) BBDebugBreak();
        // if( x + 13 == 2853878795 ) BBDebugBreak();
        // if( x + 14 == 2853878795 ) BBDebugBreak();
        // if( x + 15 == 2853878795 ) BBDebugBreak();

        block += entriesPerBlock;
        x     += entriesPerBlock;
    }

    // Process trailing entries
    for( uint i = 0; i < trailingEntries; i++ )
    {
        const uint32 y   = Swap32( block[i] );
        const uint32 idx = --pfxSum[y >> kMinusKExtraBits];
        
        buckets[idx] = ( y  << kExtraBits ) | ( ( x + i ) >> kMinusKExtraBits );
        xBuffer[idx] = x + i;
    }

    // Now this chunk can be submitted to the write queue, and we can continue to the next one.
    // After all the chunks have been written, we can read back from disk to sort each bucket
    // #TODO: Move this to its own func
    if constexpr ( WriteToDisk )
    {
        if( this->LockThreads() )
        {
            // #TODO: Don't do this if not using direct IO?
            // #NOTE: We give it the non-block aligned size, but the Queue will 
            //        only write up to the block aligned size. The rest
            //        we write with the remainder buffers.
            for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
                sizes[i] = (uint32)( bucketCounts[i] * sizeof( uint32 ) );

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

            this->ReleaseThreads();
        }
        else
            this->WaitForRelease();
    }
}

//-----------------------------------------------------------
void F1GenBucketized::CalculateMultithreadedPredixSum( 
        uint32 counts[BB_DP_BUCKET_COUNT],
        uint32 pfxSum[BB_DP_BUCKET_COUNT],
        const size_t fileBlockSize
    )
{
    const uint32 jobId    = this->JobId();
    const uint32 jobCount = this->JobCount();

    this->counts = counts;
    this->SyncThreads();

    // Add up all of the jobs counts
    memset( pfxSum, 0, sizeof( uint32 ) * BB_DP_BUCKET_COUNT );

    for( uint i = 0; i < jobCount; i++ )
    {
        const uint* tCounts = GetJob( i ).counts;

        for( uint j = 0; j < BB_DP_BUCKET_COUNT; j++ )
            pfxSum[j] += tCounts[j];
    }

    // #TODO: Only do this for the control thread?
    // uint32 totalCount = 0;
    // for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
    //     totalCount += pfxSum[i];

    // If we're the control thread, retain the total bucket count for this chunk.
    if( this->IsControlThread() )
    {
        uint32* totalBucketCounts = this->totalBucketCounts;

        // Add total bucket counts
        for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
            this->totalBucketCounts[i] += pfxSum[i];
    }

    // #TODO: Only do this if using Direct IO
    // We need to align our bucket totals to the 
    // file block size boundary so that each block buffer
    // is properly aligned for direct io.
    for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
        pfxSum[i] = RoundUpToNextBoundary( pfxSum[i] * sizeof( uint32 ), (int)fileBlockSize ) / sizeof( uint32 );

    // Calculate the prefix sum
    for( uint i = 1; i < BB_DP_BUCKET_COUNT; i++ )
        pfxSum[i] += pfxSum[i-1];

    // Subtract the count from all threads after ours 
    // to get the correct prefix sum for this thread
    for( uint t = jobId+1; t < jobCount; t++ )
    {
        const uint* tCounts = GetJob( t ).counts;

        for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
            pfxSum[i] -= tCounts[i];
    }
}