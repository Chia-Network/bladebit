#include "F1GenBucketized.h"
#include "threading/MTJob.h"
#include "plotdisk/DiskBufferQueue.h"
#include "plotting/PlotTypes.h"
#include "plotting/PackEntries.h"
#include "threading/ThreadPool.h"
#include "pos/chacha8.h"
#include "util/Util.h"

struct F1DiskBucketJob : PrefixSumJob<F1DiskBucketJob>
{
    DiskBufferQueue* diskQueue;
    const byte* key;                        // Chacha key
    uint32      x;                          // Starting x value for this thread
    uint32      numBuckets;                 // How many buckets are we processing?
    uint64      blocksPerBucketPerThread;   // How many blocks to write per bucket per thread
    uint64      blockCount;                 // Total number of blocks for each thread.
    uint32*     totalBucketCounts;          // Total counts per for all buckets. Used by the control thread
    
    byte*       blocks;                     // chacha blocks buffer
    uint64*     entries;                    // Entries in y + x format
    uint64*     writeBuffer;                // Entries encoded in disk format. Set by the control thread
    size_t      writebufferSize;            // Size to allocate for the serialization buffer

    FileId      fileId;

    void Run() override;
    inline void  GetWriteBuffer( uint32*& sizes, const uint64 blockCount, const size_t entrySizeBits );
    inline void  EncodeEntries( const uint64* input, uint64* output, const uint32* pfxSum );
    inline void  WriteBuckets( uint64* entries, const uint32* bucketCounts, uint32* sizes );

};

//-----------------------------------------------------------
size_t F1GenBucketized::GetEntrySizeBits( const uint32 numBuckets, const uint32 k )
{
    ASSERT( numBuckets >= 64 );
    const uint32 bitsSaved    = (uint32)log2( numBuckets ) - kExtraBits;
    const size_t bitsPerEntry = k * 2 - bitsSaved;
    return bitsPerEntry; 
}

///
/// Disk F1
///
//-----------------------------------------------------------
void F1GenBucketized::GenerateF1Disk(
    const byte*      plotId,
    ThreadPool&      pool, 
    uint32           threadCount,
    DiskBufferQueue& diskQueue,
    uint32*          bucketCounts,
    const uint32     numBuckets,
    const FileId     fileId
)
{
    constexpr uint32 k = _K;
    static_assert( k == 32, "K32-only versioin." );

    ASSERT( threadCount <= pool.ThreadCount() );
    ASSERT( numBuckets >= BB_DP_MIN_BUCKET_COUNT );
    ASSERT( numBuckets <= BB_DP_MAX_BUCKET_COUNT );
    
    const uint64 kEntryCount              = 1ull << k;
    const uint64 entriesPerBucket         = kEntryCount / numBuckets;
    const uint32 entriesPerBlock          = (uint32)( kF1BlockSize / (k / 8ull) );
    
    const uint64 blockCount               = kEntryCount / entriesPerBlock;
    const uint64 blocksPerThread          = blockCount / threadCount;
    const uint64 blocksPerBucket          = entriesPerBucket / entriesPerBlock;
    const uint64 blocksPerBucketPerThread = blocksPerBucket / threadCount;
    
    const uint32 bitsSaved        = (uint32)log2( numBuckets ) - kExtraBits;
    // const size_t bitsPerEntry     = k * 2 - bitsSaved;                         // Bits per entry when the entry is serialized
    
    uint64 trailingBlocks = blockCount - blocksPerThread * threadCount;

    // Working buffer allocation size
    const size_t blockBufferSize = RoundUpToNextBoundary( blocksPerBucketPerThread * kF1BlockSize, sizeof( uint64 ) );
    const size_t writeBufferSize = blocksPerBucketPerThread * entriesPerBlock * threadCount * sizeof( uint64 );  // y - extraBits + x

    // Allocate our buffers
    uint64* entries    = (uint64*)diskQueue.GetBuffer( writeBufferSize );
    byte*   blocksRoot = diskQueue.GetBuffer( blockBufferSize * threadCount );
    byte*   blocks     = blocksRoot;

     // Prepare Jobs
    byte key[32] = { 1 };
    memcpy( key + 1, plotId, 31 );

    memset( bucketCounts, 0, sizeof( uint32 ) * numBuckets );

    MTJobRunner<F1DiskBucketJob> jobs( pool );
    uint32 x = 0;

    for( uint i = 0; i < threadCount; i++ )
    {
        F1DiskBucketJob& job = jobs[i];

        job.key                      = key;
        job.x                        = x;
        job.numBuckets               = numBuckets;
        job.blockCount               = blocksPerThread;
        job.blocksPerBucketPerThread = blocksPerBucketPerThread;
        job.blocks                   = blocks;
        job.entries                  = entries;
        job.writeBuffer              = nullptr;
        job.totalBucketCounts        = nullptr;
        job.counts                   = nullptr;

        job.diskQueue                = &diskQueue;
        job.writebufferSize          = writeBufferSize;
        job.fileId                   = fileId;

        if( trailingBlocks )
        {
            job.blockCount += trailingBlocks;
            trailingBlocks--;
        }

        x      += job.blockCount * entriesPerBlock;
        blocks += blockBufferSize;
    }

    ASSERT( trailingBlocks == 0 );

    jobs[0].totalBucketCounts = bucketCounts;

    // Run jobs
    jobs.Run( threadCount );

    // Release our buffers
    {
        Fence fence;

        diskQueue.ReleaseBuffer( entries    );
        diskQueue.ReleaseBuffer( blocksRoot );
        
        diskQueue.SignalFence( fence );
        diskQueue.CommitCommands();
        fence.Wait();
        diskQueue.CompletePendingReleases();
    }
}

//-----------------------------------------------------------
void F1DiskBucketJob::Run()
{
    const uint32 k = _K;
    static_assert( k == 32, "Only k32 supported for now." );

    const uint32 kMinusKExtraBits  = k - kExtraBits;
    const uint32 entriesPerBlock   = (uint32)( kF1BlockSize / (k / 8ull) );
    const uint32 numBuckets        = this->numBuckets;
    const uint32 bitsSaved         = (uint32)log2( numBuckets ) - kExtraBits;
    const uint32 bucketBitShift    = k - ( kExtraBits + bitsSaved );
    const size_t bitsPerEntry      = k * 2 - bitsSaved;
    const uint64 blocksPerBucket   = this->blocksPerBucketPerThread;

    byte*   blocks      = this->blocks;
    uint64* entries     = this->entries;

    uint32* sizes       = nullptr;          // Only used by the control thread   
    uint64  blockCount  = this->blockCount;

    // For distribution into buckets
    uint32 counts        [BB_DP_MAX_BUCKET_COUNT];
    uint32 pfxSum        [BB_DP_MAX_BUCKET_COUNT];
    uint32 totalCounts   [BB_DP_MAX_BUCKET_COUNT];

    uint32* outTotalCounts = this->totalBucketCounts;
    this->totalBucketCounts = totalCounts;
   
    // Start processing blocks
    uint64 x = this->x;

    chacha8_ctx chacha;
    chacha8_keysetup( &chacha, key, 256, NULL );

    for( ;; )
    {
        const uint64 blocksToWrite = std::min( blockCount, blocksPerBucket );
        
        // Drop any threads that are not participating
        if( blocksToWrite < blocksPerBucket )
        {
            uint32 threadCount = this->JobCount();
            for( uint32 i = 0; i < this->JobCount(); i++ )
            {
                if( this->GetJob( i ).blockCount < 1 )
                {
                    if( !this->ReduceThreadCount( i ) )
                        return;

                    break;
                }
            }
        }

        ASSERT( blocksToWrite );

        // Generate chacha blocks
        const uint64 chachaBlock = x * _K / kF1BlockSizeBits;
        chacha8_get_keystream( &chacha, chachaBlock, blocksToWrite, blocks );

        // Count how many entries we have per bucket
        memset( counts, 0, sizeof( uint32 ) * numBuckets );
        
        const int64   entryCount = (int64)( blocksToWrite * entriesPerBlock );
        const uint32* u32Blocks  = (uint32*)blocks;

        for( int64 i = 0; i < entryCount; i++ )
            counts[Swap32( u32Blocks[i] ) >> bucketBitShift] ++;

        memset( totalCounts, 0, numBuckets * sizeof( uint32 ) );
        CalculatePrefixSum( numBuckets, counts, pfxSum, totalCounts );

        if( this->IsControlThread() )
        {
            for( uint32 i = 0; i < numBuckets; i++ )
                outTotalCounts[i] += totalCounts[i];
        }

        // Distribute into buckets
        for( int64 i = 0; i < entryCount; i++ )
        {
            const uint64 y   = Swap32( u32Blocks[i] );
            const uint32 dst = --pfxSum[y >> bucketBitShift];
            entries[dst] = ( y << kExtraBits ) | ( ( x + (uint64)i ) >> kMinusKExtraBits );
        }

        // Grab a buffer to write to
        GetWriteBuffer( sizes, blocksToWrite, bitsPerEntry );
        ASSERT( this->writeBuffer );

        // Convert entries to bits
        // #TODO: We need to save the last offset of the bit we wrote here.
        //        Otherwise we will have incorrect entries when loading the bucket.
        EncodeEntries( entries, (uint64*)this->writeBuffer, pfxSum );

        // Write to disk
        WriteBuckets( (uint64*)this->writeBuffer, totalCounts, sizes );

        x += entriesPerBlock * blocksToWrite;
        blockCount -= blocksToWrite;
        this->blockCount -= blocksToWrite;

        this->writeBuffer = nullptr;
    }
}

//-----------------------------------------------------------
inline void F1DiskBucketJob::EncodeEntries( const uint64* input, uint64* output, const uint32* pfxSum )
{
    const uint32* myCounts    = this->counts;                           // This job's entry counts
    const uint32* totalCounts = this->GetJob( 0 ).totalBucketCounts;    // Total counts for this iteration

    const uint32 numBuckets   = this->numBuckets;
    const size_t bitsPerEntry = F1GenBucketized::GetEntrySizeBits( numBuckets, _K );

    // All buckets must have a minimum set of entries, otherwise do it single-threaded
    const uint32 threadCount = this->JobCount();
    const uint32 threshold   = threadCount * 2;

    for( uint32 bucket = 0; bucket < numBuckets; bucket++ )
    {
        if( totalCounts[bucket] <= threshold )
        {
            if( this->JobId() > 0 )
                return;

            // Encode all buckets single-threaded
            uint64 entryOffset = 0;
            for( uint32 bucket = 0; bucket < numBuckets; bucket++ )
            {
                const uint64 entryCount = totalCounts[bucket];

                EntryPacker::SerializeEntries( input, output, (int64)entryCount, entryOffset, bitsPerEntry );
                
                const uint64 fieldAlignedCount = CDiv( entryCount * bitsPerEntry, 64 );

                entryOffset += entryCount;
                input       += entryCount;
                output      += fieldAlignedCount;
            }

            return;
        }
    }

    for( uint32 bucket = 0; bucket < numBuckets; bucket++ )
    {
        const uint64 entryCount  = myCounts[bucket];
        const uint64 entryOffset = pfxSum  [bucket];

        ASSERT( entryCount > 2 );

        EntryPacker::SerializeEntries( input, output, (int64)entryCount - 2, entryOffset, bitsPerEntry );
        this->SyncThreads();
        EntryPacker::SerializeEntries( input, output, 2, entryOffset + entryCount - 2, bitsPerEntry );
        
        const uint64 fieldAlignedCount = CDiv( totalCounts[bucket] * bitsPerEntry, 64 );
        input  += totalCounts[bucket];
        output += fieldAlignedCount;
    }
}

//-----------------------------------------------------------
inline void F1DiskBucketJob::GetWriteBuffer( uint32*& sizes, const uint64 blockCount, const size_t entrySizeBits )
{
    if( this->IsControlThread() )
    {
        this->LockThreads();

        auto* bits = (uint64*)this->diskQueue->GetBuffer( this->writebufferSize );
        
        sizes  = (uint32*)this->diskQueue->GetBuffer( this->numBuckets * sizeof( uint32 ) );

        for( uint32 i = 0; i < this->JobCount(); i++ )
        {
            auto& job       = _jobs[i];
            job.writeBuffer = bits;
        }

        this->ReleaseThreads();
    }
    else
    {
        this->WaitForRelease();
    }
}

//-----------------------------------------------------------
inline void F1DiskBucketJob::WriteBuckets( uint64* entries, const uint32* bucketCounts, uint32* sizes )
{
    if( this->IsControlThread() )
    {
        const size_t bitsPerEntry = F1GenBucketized::GetEntrySizeBits( numBuckets, _K );

        this->LockThreads();

        for( uint32 i = 0; i < numBuckets; i++ )
            totalBucketCounts[i] += bucketCounts[i];

        for( uint32 i = 0; i < numBuckets; i++ )
            sizes[i] = CDiv( bucketCounts[i] * bitsPerEntry, 64 );

        diskQueue->WriteBuckets( fileId, entries, sizes );
        diskQueue->ReleaseBuffer( entries );
        diskQueue->ReleaseBuffer( sizes );
        diskQueue->CommitCommands();

        this->ReleaseThreads();
    }
    else
        this->WaitForRelease();
}


/*
///
/// F1 Generation
///
//-----------------------------------------------------------
template<bool WriteToDisk, bool SingleThreaded, typename TJob>
void GenerateF1( 
    chacha8_ctx&      chacha, 
    uint32            x,
    byte*             blocks,
    uint32            blockCount,
    uint64            entryCount,
    uint32*           yBuckets,
    uint32*           xBuckets,
    uint32            bucketCounts[BB_DP_BUCKET_COUNT],
    MTJobSyncT<TJob>* job,
    size_t            fileBlockSize
    )
{
    const uint64 chachaBlock = ((uint64)x) * _K / kF1BlockSizeBits;

    const uint32 entriesPerBlock   = kF1BlockSize / sizeof( uint32 );
    const uint32 kMinusKExtraBits  = _K - kExtraBits;
    const uint32 bucketShift       = (8u - (uint)kExtraBits);

    // const uint32 jobId             = job->JobId();
    // const uint32 jobCount          = job->JobCount();

    ASSERT( entryCount <= (uint64)blockCount * entriesPerBlock );
    
    // Generate chacha blocks
    chacha8_get_keystream( &chacha, chachaBlock, blockCount, blocks );

    // Count how many entries we have per bucket
    uint counts[BB_DP_BUCKET_COUNT];
    uint pfxSum[BB_DP_BUCKET_COUNT];

    memset( counts, 0, sizeof( counts ) );

    const uint32* block = (uint32*)blocks;

    // Count entries per bucket. Only calculate the blocks that have full entries
    const uint32 fullBlockCount  = entryCount / entriesPerBlock;
    const uint64 trailingEntries = (uint32)( blockCount * (uint64)entriesPerBlock - entryCount );

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
    for( uint64 i = 0; i < trailingEntries; i++ )
        counts[( block[i] >> bucketShift ) & 0x3F] ++;

    // Calculate the prefix sum for our buckets
    if constexpr ( SingleThreaded )
    {
        memcpy( pfxSum, counts, sizeof( counts ) );

        // Save output bucket counts
        memcpy( bucketCounts, counts, sizeof( counts ) );

        if( WriteToDisk && fileBlockSize )
        {
            // We need to align each count to file block size
            // so that each bucket starts aligned 
            // (we won't write submit these extra false entries)
            for( uint i = 0; i < BB_DP_BUCKET_COUNT; i++ )
                pfxSum[i] = RoundUpToNextBoundary( pfxSum[i] * sizeof( uint32 ), (int)fileBlockSize ) / sizeof( uint32 );
        }

        for( uint i = 1; i < BB_DP_BUCKET_COUNT; i++ )
            pfxSum[i] += pfxSum[i-1];
    }
    else
    {
        static_cast<F1BucketJob<TJob>*>( job )->CalculateMultiThreadedPrefixSum( counts, pfxSum, bucketCounts, fileBlockSize );
    }
    
    // Now we know the offset where we can start distributing
    // y and x values to their respective buckets.
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
        yBuckets[idx0 ] = ( y0  << kExtraBits ) | ( ( x + 0  ) >> kMinusKExtraBits );
        yBuckets[idx1 ] = ( y1  << kExtraBits ) | ( ( x + 1  ) >> kMinusKExtraBits );
        yBuckets[idx2 ] = ( y2  << kExtraBits ) | ( ( x + 2  ) >> kMinusKExtraBits );
        yBuckets[idx3 ] = ( y3  << kExtraBits ) | ( ( x + 3  ) >> kMinusKExtraBits );
        yBuckets[idx4 ] = ( y4  << kExtraBits ) | ( ( x + 4  ) >> kMinusKExtraBits );
        yBuckets[idx5 ] = ( y5  << kExtraBits ) | ( ( x + 5  ) >> kMinusKExtraBits );
        yBuckets[idx6 ] = ( y6  << kExtraBits ) | ( ( x + 6  ) >> kMinusKExtraBits );
        yBuckets[idx7 ] = ( y7  << kExtraBits ) | ( ( x + 7  ) >> kMinusKExtraBits );
        yBuckets[idx8 ] = ( y8  << kExtraBits ) | ( ( x + 8  ) >> kMinusKExtraBits );
        yBuckets[idx9 ] = ( y9  << kExtraBits ) | ( ( x + 9  ) >> kMinusKExtraBits );
        yBuckets[idx10] = ( y10 << kExtraBits ) | ( ( x + 10 ) >> kMinusKExtraBits );
        yBuckets[idx11] = ( y11 << kExtraBits ) | ( ( x + 11 ) >> kMinusKExtraBits );
        yBuckets[idx12] = ( y12 << kExtraBits ) | ( ( x + 12 ) >> kMinusKExtraBits );
        yBuckets[idx13] = ( y13 << kExtraBits ) | ( ( x + 13 ) >> kMinusKExtraBits );
        yBuckets[idx14] = ( y14 << kExtraBits ) | ( ( x + 14 ) >> kMinusKExtraBits );
        yBuckets[idx15] = ( y15 << kExtraBits ) | ( ( x + 15 ) >> kMinusKExtraBits );

        // Store the x that generated this y
        xBuckets[idx0 ] = x + 0 ;
        xBuckets[idx1 ] = x + 1 ;
        xBuckets[idx2 ] = x + 2 ;
        xBuckets[idx3 ] = x + 3 ;
        xBuckets[idx4 ] = x + 4 ;
        xBuckets[idx5 ] = x + 5 ;
        xBuckets[idx6 ] = x + 6 ;
        xBuckets[idx7 ] = x + 7 ;
        xBuckets[idx8 ] = x + 8 ;
        xBuckets[idx9 ] = x + 9 ;
        xBuckets[idx10] = x + 10;
        xBuckets[idx11] = x + 11;
        xBuckets[idx12] = x + 12;
        xBuckets[idx13] = x + 13;
        xBuckets[idx14] = x + 14;
        xBuckets[idx15] = x + 15;

        block += entriesPerBlock;
        x     += entriesPerBlock;
    }

    // Process trailing entries
    for( uint64 i = 0; i < trailingEntries; i++ )
    {
        const uint32 y   = Swap32( block[i] );
        const uint32 idx = --pfxSum[y >> kMinusKExtraBits];
        
        yBuckets[idx] = ( y  << kExtraBits ) | ( ( x + i ) >> kMinusKExtraBits );
        xBuckets[idx] = x + i;
    }
}
*/
