#include "plotdisk/DiskPlotPhase1.h"
#include "util/Util.h"

// Load and sort F7s along with its key, then write back to disk
//-----------------------------------------------------------
void DiskPlotPhase1::SortAndCompressTable7()
{
    Log::Line( "Sorting F7s and writing C tables to plot file..." );
    const auto timer = TimerBegin();

    DiskPlotContext& context = _cx;
    DiskBufferQueue& ioQueue = *context.ioQueue;

    ioQueue.ResetHeap( context.heapSize, context.heapBuffer );

    // Load F7 buckets and sort them
    const uint32 BucketCount = BB_DP_BUCKET_COUNT;

    struct BucketBuffer
    {
        uint32* f7;
        uint32* key;
    };

    uint32 c1NextCheckpoint = 0;  // How many C1 entries to skip until the next checkpoint. If there's any entries here, it means the last bucket wrote a
    uint32 c2NextCheckpoint = 0;  // checkpoint entry and still had entries which did not reach the next checkpoint.
                                  // Ex. Last bucket had 10005 entries, so it wrote a checkpoint at 0 and 10000, then it counted 5 more entries, so
                                  // the next checkpoint would be after 9995 entries.

    // These buffers are small enough on k32 (around 1.6MiB for C1, C2 is negligible), we keep the whole thing in memory,
    // while we write C3 to the actual file
    const uint32 c1Interval  = kCheckpoint1Interval;
    const uint32 c2Interval  = kCheckpoint1Interval * kCheckpoint2Interval;

    const uint64 tableLength = context.entryCounts[(int)TableId::Table7];
    const uint32 c1TotalEntries = (uint32)CDiv( tableLength, (int)c1Interval ) + 1; // +1 because chiapos adds an extra '0' entry at the end
    const uint32 c2TotalEntries = (uint32)CDiv( tableLength, (int)c2Interval ) + 1; // +1 because we add a short-circuit entry to prevent C2 lookup overflows
                                                                                    // #TODO: Remove the extra c2 entry when we support >k^32 entries
    
    const size_t c1TableSizeBytes = c1TotalEntries * sizeof( uint32 );
    const size_t c2TableSizeBytes = c2TotalEntries * sizeof( uint32 );

    uint32* c1Buffer = (uint32*)ioQueue.GetBuffer( c1TableSizeBytes );
    uint32* c2Buffer = (uint32*)ioQueue.GetBuffer( c2TableSizeBytes );
    
    // See Note in LoadNextbucket() regarding the 'prefix region' for c3 overflow that we allocate
    const size_t c3ParkOverflowSize = sizeof( uint32 ) * kCheckpoint1Interval;

    uint32 c3ParkOverflowCount = 0; // Overflow entries from a bucket that did not make it into a C3 park this bucket. Saved for the next bucket.
    uint32 c3ParkOverflow[kCheckpoint1Interval];

    size_t c3TableSizeBytes    = 0; // Total size of the C3 table


    // #TODO: Seek to the C3 table instead of writing garbage data.
    //        For now we write false C1 and C2 tables to get the plot file to the right offset
    ioQueue.WriteFile( FileId::PLOT, 0, c1Buffer, c1TableSizeBytes );
    ioQueue.WriteFile( FileId::PLOT, 0, c2Buffer, c2TableSizeBytes );
    ioQueue.CommitCommands();


    uint32       bucketsLoaded        = 0;
    BucketBuffer buffers[BucketCount] = { 0 };

    Fence readFence;

    // #TODO: Check buffer usage here fits within the minimum heap allocation.
    ioQueue.SeekBucket( FileId::F7       , 0, SeekOrigin::Begin );
    ioQueue.SeekBucket( FileId::SORT_KEY7, 0, SeekOrigin::Begin );
    ioQueue.CommitCommands();

    auto LoadNextBucket = [&]() -> void 
    {
        ASSERT( bucketsLoaded < BucketCount );

        const uint32 bucket       = bucketsLoaded;
        const uint32 bucketLength = context.bucketCounts[(int)TableId::Table7][bucket];
        
        const size_t loadSize     = sizeof( uint32 ) * bucketLength;

        BucketBuffer& bucketBuffer = buffers[bucketsLoaded++];

        // Add enough space on the f7 buffer to keep cross-bucket C3 park's worth of entries.
        // this way we can copy park overflow entries to that prefix region and process the parks normally.
        bucketBuffer.f7 =  (uint32*)( ioQueue.GetBuffer( loadSize + c3ParkOverflowSize, true ) + c3ParkOverflowSize );
        bucketBuffer.key = (uint32*)ioQueue.GetBuffer( loadSize, true );

        ioQueue.ReadFile( FileId::F7       , bucket, bucketBuffer.f7 , bucketLength * sizeof( uint32 ) );
        ioQueue.ReadFile( FileId::SORT_KEY7, bucket, bucketBuffer.key, bucketLength * sizeof( uint32 ) );
        ioQueue.DeleteFile( FileId::SORT_KEY7, bucket );
        ioQueue.SignalFence( readFence, bucketsLoaded );
        ioQueue.CommitCommands();
    };

    // Load first bucket
    LoadNextBucket();

    uint32* c1Writer = c1Buffer;
    uint32* c2Writer = c2Buffer;

    for( uint32 bucket = 0; bucket < BucketCount; bucket++ )
    {
        const uint32 nextBucket = bucket + 1;

        if( bucketsLoaded < BucketCount )
            LoadNextBucket();

        readFence.Wait( nextBucket, _cx.readWaitTime );

        ioQueue.DeleteFile( FileId::F7, bucket );
        ioQueue.CommitCommands();

        // Sort on F7
        BucketBuffer& buffer = buffers[bucket];

        const uint32 bucketLength = context.bucketCounts[(int)TableId::Table7][bucket];
        const size_t allocSize    = sizeof( uint32 ) * bucketLength;

        uint32* keyTmp = (uint32*)ioQueue.GetBuffer( allocSize * 2, true );
        uint32* f7Tmp  = keyTmp + bucketLength;

        RadixSort256::SortWithKey<BB_MAX_JOBS>( *context.threadPool, 
            buffer.f7, f7Tmp, buffer.key, keyTmp, bucketLength );
        
        // Write reverse map
        // #NOTE: We can re-use the temp sort buffer as the map buffer
        //        We instruct WriteReverseMap to release our key buffer as well.
        uint64* map = (uint64*)keyTmp;
        WriteReverseMap( TableId::Table7, bucket, bucketLength, buffer.key, map, nullptr, true );
        ioQueue.ReleaseBuffer( map );
        ioQueue.CommitCommands();


        // At this point we can compress F7 into C tables
        // and write them to the plot file as the first 3 tables
        // write plot header and addresses.
        // We will set the addersses to these tables accordingly.
        const uint32 threadCount = context.cThreadCount;

        // Write C1
        {
            ASSERT( bucketLength > c1NextCheckpoint );

            // #TODO: Do C1 multi-threaded. For now jsut single-thread it...
            for( uint32 i = c1NextCheckpoint; i < bucketLength; i += c1Interval )
                *c1Writer++ = Swap32( buffer.f7[i] );
            
            // Track how many entries we covered in the last checkpoint region
            const uint32 c1Length          = bucketLength - c1NextCheckpoint;
            const uint32 c1CheckPointCount = CDiv( c1Length, (int)c1Interval );

            c1NextCheckpoint = c1CheckPointCount * c1Interval - c1Length;
        }

        // Write C2
        {
            // C2 has so few entries on k=32 that there's no sense in doing it multi-threaded
            static_assert( _K == 32 );

            if( c2NextCheckpoint >= bucketLength )
                c2NextCheckpoint -= bucketLength;   // No entries to write in this bucket
            else
            {
                for( uint32 i = c2NextCheckpoint; i < bucketLength; i += c2Interval )
                    *c2Writer++ = Swap32( buffer.f7[i] );
            
                // Track how many entries we covered in the last checkpoint region
                const uint32 c2Length          = bucketLength - c2NextCheckpoint;
                const uint32 c2CheckPointCount = CDiv( c2Length, (int)c2Interval );

                c2NextCheckpoint = c2CheckPointCount * c2Interval - c2Length;
            }
        }

        // Write C3
        {
            const bool isLastBucket = nextBucket == BucketCount;

            uint32* c3F7           = buffer.f7;
            uint32  c3BucketLength = bucketLength;

            if( c3ParkOverflowCount )
            {
                // Copy our overflow to the prefix region of our f7 buffer
                c3F7 -= c3ParkOverflowCount;
                c3BucketLength += c3ParkOverflowCount;

                memcpy( c3F7, c3ParkOverflow, sizeof( uint32 ) * c3ParkOverflowCount );

                c3ParkOverflowCount = 0;
            }
            
            
            // #TODO: Remove this
            // Dump f7's that have the value of 0xFFFFFFFF for now,
            // this is just for compatibility with RAM bladebit for testing
            // plots against it.
            if( isLastBucket )
            {
                while( c3F7[c3BucketLength-1] == 0xFFFFFFFF )
                    c3BucketLength--;
            }

            // See TableWriter::GetC3ParkCount for details
            uint32 parkCount       = c3BucketLength / kCheckpoint1Interval;
            uint32 overflowEntries = c3BucketLength - ( parkCount * kCheckpoint1Interval );

            // Greater than 1 because the first entry is excluded as it is written in C1 instead.
            if( isLastBucket && overflowEntries > 1 )
            {
                overflowEntries = 0;
                parkCount++;
            }
            else if( overflowEntries && !isLastBucket )
            {
                // Save any entries that don't fill-up a full park for the next bucket
                memcpy( c3ParkOverflow, c3F7 + c3BucketLength - overflowEntries, overflowEntries * sizeof( uint32 ) );
                
                c3ParkOverflowCount = overflowEntries;
                c3BucketLength -= overflowEntries;
            }


            const size_t c3BufferSize = CalculateC3Size() * parkCount;
            byte* c3Buffer = ioQueue.GetBuffer( c3BufferSize );

            // #NOTE: This function uses re-writes our f7 buffer, so ensure it is done after
            //        that buffer is no longer needed.
            const size_t sizeWritten = TableWriter::WriteC3Parallel<BB_MAX_JOBS>( *context.threadPool, 
                                            threadCount, c3BucketLength, c3F7, c3Buffer );
            ASSERT( sizeWritten == c3BufferSize );

            c3TableSizeBytes += sizeWritten;


            // Write the C3 table to the plot file directly
            ioQueue.WriteFile( FileId::PLOT, 0, c3Buffer, c3BufferSize );
            ioQueue.ReleaseBuffer( c3Buffer );
            ioQueue.CommitCommands();
        }

        // Done with the bucket f7 buffer
        ioQueue.ReleaseBuffer( buffer.f7 - kCheckpoint1Interval );
        ioQueue.CommitCommands();
    }

    // Seek back to the begining of the C1 table and
    // write C1 and C2 buffers to file, then seek back to the end of the C3 table

    c1Buffer[c1TotalEntries-1] = 0;          // Chiapos adds a trailing 0
    c2Buffer[c2TotalEntries-1] = 0xFFFFFFFF; // C2 overflow protection

    readFence.Reset( 0 );

    ioQueue.SeekBucket( FileId::PLOT, -(int64)( c1TableSizeBytes + c2TableSizeBytes + c3TableSizeBytes ), SeekOrigin::Current );
    ioQueue.WriteFile( FileId::PLOT, 0, c1Buffer, c1TableSizeBytes );
    ioQueue.WriteFile( FileId::PLOT, 0, c2Buffer, c2TableSizeBytes );
    ioQueue.ReleaseBuffer( c1Buffer );
    ioQueue.ReleaseBuffer( c2Buffer );
    ioQueue.SeekBucket( FileId::PLOT, (int64)c3TableSizeBytes, SeekOrigin::Current );

    ioQueue.SignalFence( readFence );
    ioQueue.CommitCommands();

    // Save C table addresses into the plot context.
    // And set the starting address for the table 1 to be written
    const size_t headerSize = ioQueue.PlotHeaderSize();

    context.plotTablePointers[7] = headerSize;                                      // C1
    context.plotTablePointers[8] = context.plotTablePointers[7] + c1TableSizeBytes; // C2
    context.plotTablePointers[9] = context.plotTablePointers[8] + c2TableSizeBytes; // C3
    context.plotTablePointers[0] = context.plotTablePointers[9] + c3TableSizeBytes; // T1

    // Save sizes
    context.plotTableSizes[7] = c1TableSizeBytes;
    context.plotTableSizes[8] = c2TableSizeBytes;
    context.plotTableSizes[9] = c3TableSizeBytes;

    const double elapsed = TimerEnd( timer );
    Log::Line( "Finished sorting and writing C tables in %.2lf seconds.", elapsed );

    // Wait for all commands to finish
    readFence.Wait();
    ioQueue.CompletePendingReleases();
}