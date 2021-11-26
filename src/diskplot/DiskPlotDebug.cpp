#include "DiskPlotDebug.h"
#include "io/FileStream.h"
#include "threading/ThreadPool.h"
#include "threading/AutoResetSignal.h"
#include "algorithm/RadixSort.h"
#include "Util.h"
#include "util/Log.h"

//-----------------------------------------------------------
void Debug::ValidateYFileFromBuckets( FileId yFileId, ThreadPool& pool, DiskBufferQueue& queue, TableId table, 
                                      uint32 bucketCounts[BB_DP_BUCKET_COUNT] )
{
    const size_t bucketMaxCount  = BB_DP_MAX_ENTRIES_PER_BUCKET;

    uint64  refEntryCount = 0;
    uint64* refEntries    = nullptr;

    // Read the whole reference table into memory
    Log::Line( "Loading reference table..." );
    {
        char path[1024];
        sprintf( path, "%st%d.y.tmp", BB_DP_DBG_REF_DIR, (int)table+1 );

        FileStream refTable;
        FatalIf( !refTable.Open( path, FileMode::Open, FileAccess::Read, FileFlags::LargeFile | FileFlags::NoBuffering ),
                 "Failed to open reference table file %s.", path );

        const size_t blockSize  = refTable.BlockSize();
        const uint64 maxEntries = 1ull << _K;

        const size_t allocSize = (size_t)maxEntries * sizeof( uint64 );

        byte* block = bbvirtalloc<byte>( blockSize );
        refEntries  = bbvirtalloc<uint64>( allocSize );

        // The first block contains the entry count
        FatalIf( !refTable.Read( refEntries, blockSize ), 
                "Failed to read count from reference table file %s with error: %d.", path, refTable.GetError() );

        refEntryCount = *refEntries;
        if( table == TableId::Table1 )
            ASSERT( refEntryCount == maxEntries );

        ASSERT( refEntryCount <= maxEntries );

        const size_t totalSize          = refEntryCount * sizeof( uint64 );
        const size_t blockSizeToRead    = totalSize / blockSize * blockSize;
        const size_t remainder          = totalSize - blockSizeToRead;

        // Read blocks
        size_t sizeToRead = blockSizeToRead;
        byte*  reader     = (byte*)refEntries;
        while( sizeToRead )
        {
            // The rest of the blocks are entries
            const ssize_t sizeRead = refTable.Read( reader, sizeToRead );
            if( sizeRead < 1 )
            {
                const int err = refTable.GetError();
                Fatal( "Failed to read entries from reference table file %s with error: %d.", path, err );
            }

            sizeToRead -= (size_t)sizeRead;
            reader += sizeRead;
        }

        // #TODO: This, for now ignore it?
        if( remainder )
        {
            // if( refTable.Read( block, blockSize ) != (ssize_t)blockSize )
            // {
            //     const int err = refTable.GetError();
            //     Fatal( "Failed to read entries from reference table file %s with error: %d.", path, err );
            // }

            // ASSERT( blockSizeToRead / sizeof( uint64 ) + remainder / sizeof( uint64 ) == refEntryCount );
            // memcpy( reader, block, remainder );
        }

        SysHost::VirtualFree( block );
    }


    // Alloc a buffer for our buckets
    const size_t bucketAllocSize = RoundUpToNextBoundaryT( sizeof( uint32 ) * (size_t)BB_DP_MAX_ENTRIES_PER_BUCKET, queue.BlockSize() );
    uint32* bucketEntries = bbvirtalloc<uint32>( bucketAllocSize );
    uint32* bucketSortTmp = bbvirtalloc<uint32>( bucketAllocSize );

    AutoResetSignal fence;

    // Load the first bucket
    queue.SeekBucket( yFileId, 0, SeekOrigin::Begin );
    queue.ReadFile( yFileId, 0, bucketEntries, bucketCounts[0] * sizeof( uint32 ) );
    queue.AddFence( fence );
    queue.CommitCommands();

    fence.Wait();
    fence.Signal(); // Set as signaled initially since we wait for the next bucket in the loop

    const uint64* refReader = refEntries;

    for( uint bucket = 0; bucket < BB_DP_BUCKET_COUNT; bucket++ )
    {
        Log::Line( "Bucket %u", bucket );

        // Wait for the next bucket to be loaded
        fence.Wait();

        const int64 entryCount = bucketCounts[bucket];
        
        // Sort the bucket
        Log::Line( "  Sorting bucket..." );
        RadixSort256::Sort<BB_MAX_JOBS>( pool, bucketEntries, bucketSortTmp, (uint64)entryCount );

        // Load the next bucket
        const uint nextBucket = bucket + 1;
        
        if( nextBucket < BB_DP_BUCKET_COUNT )
        {
            queue.ReadFile( yFileId, nextBucket, bucketSortTmp, bucketCounts[nextBucket] * sizeof( uint32 ) );
            queue.AddFence( fence );
            queue.CommitCommands();
        }

        // Start validating
        Log::Line( "  Validating entries...");
        const uint64 bucketMask = ((uint64)bucket) << 32;
        
        for( int64 i = 0; i < entryCount; i++, refReader++ )
        {
            const uint64 y      = bucketMask | bucketEntries[i];
            const uint64 yRef   = *refReader;

            // if( y == 112675641563 ) BBDebugBreak();

            const uint32 y32    = bucketEntries[i];
            const uint32 y32Ref = (uint32)yRef;

            FatalIf( y != yRef, 
                    "Failed to validate entry on table %d at bucket position %u:%lld | Global position: %lld.\n"
                    " Expected %llu but got %llu",
                     (int)table+1, bucket, i, (int64)( refReader - refEntries ), yRef, y );
        }

        Log::Line( "  Bucket %u validated successfully!", bucket );

        // Swap bucket buffers
        std::swap( bucketEntries, bucketSortTmp );
    }

    Log::Line( "Table %d y values validated successfully.", (int)table+1 );

    // Restore files to their position, just in case
    queue.SeekBucket( yFileId, 0, SeekOrigin::Begin );
    queue.AddFence( fence );
    queue.CommitCommands();
    fence.Wait();


    SysHost::VirtualFree( refEntries    );
    SysHost::VirtualFree( bucketEntries );
}


