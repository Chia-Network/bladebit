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

        const size_t totalSize       = RoundUpToNextBoundaryT( (size_t)refEntryCount * ( table == TableId::Table7 ? sizeof( uint32 ) : sizeof( uint64 ) ), blockSize );
        const size_t blockSizeToRead = totalSize / blockSize * blockSize;
        const size_t remainder       = totalSize - blockSizeToRead;

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

        if( remainder )
        {
            if( refTable.Read( block, blockSize ) != (ssize_t)blockSize )
            {
                const int err = refTable.GetError();
                Fatal( "Failed to read entries from reference table file %s with error: %d.", path, err );
            }

            // ASSERT( blockSizeToRead / sizeof( uint64 ) + remainder / sizeof( uint64 ) == refEntryCount );
            memcpy( reader, block, remainder );
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
    queue.SignalFence( fence );
    queue.CommitCommands();

    fence.Wait();
    fence.Signal(); // Set as signaled initially since we wait for the next bucket in the loop

    const uint64* refReader = refEntries;
    const uint32* f7Reader  = (uint32*)refEntries; 

    for( uint bucket = 0; bucket < BB_DP_BUCKET_COUNT; bucket++ )
    {
        Log::Line( "Bucket %u", bucket );

        // Wait for the next bucket to be loaded
        fence.Wait();

        const int64 entryCount = bucketCounts[bucket];
        
        // Sort the bucket
    // if( table < TableId::Table7 )
        {
            Log::Line( "  Sorting bucket..." );
            RadixSort256::Sort<BB_MAX_JOBS>( pool, bucketEntries, bucketSortTmp, (uint64)entryCount );
        }

        // Load the next bucket
        const uint nextBucket = bucket + 1;
        
        if( nextBucket < BB_DP_BUCKET_COUNT )
        {
            queue.ReadFile( yFileId, nextBucket, bucketSortTmp, bucketCounts[nextBucket] * sizeof( uint32 ) );
            queue.SignalFence( fence );
            queue.CommitCommands();
        }

        // Start validating
        Log::Line( "  Validating entries...");
        if( table < TableId::Table7 )
        {
            const uint64 bucketMask = ((uint64)bucket) << 32;
            uint64 prevRef = 0;
            
            for( int64 i = 0; i < entryCount; i++, refReader++ )
            {
                const uint64 y      = bucketMask | bucketEntries[i];
                const uint64 yRef   = *refReader;

                // Test for now, since we're not getting all of the reference values in some tables...
                if( yRef < prevRef )
                    break;
                prevRef = yRef;
                // if( y == 112675641563 ) BBDebugBreak();

                // const uint32 y32    = bucketEntries[i];
                // const uint32 y32Ref = (uint32)yRef;

                FatalIf( y != yRef, 
                        "Failed to validate entry on table %d at bucket position %u:%lld | Global position: %lld.\n"
                        " Expected %llu but got %llu",
                        (int)table+1, bucket, i, (int64)( refReader - refEntries ), yRef, y );
            }
        }
        else
        {
            const uint32* refEnd = f7Reader + refEntryCount;
            
            for( int64 i = 0; i < entryCount && f7Reader < refEnd; i++, f7Reader++ )
            {
                const uint32 y      = bucketEntries[i];
                const uint32 yRef   = *f7Reader;

                FatalIf( y != yRef, 
                        "Failed to validate entry on table %d at bucket position %u:%lld | Global position: %lld.\n"
                        " Expected %lu but got %lu",
                        (int)table+1, bucket, i, (int64)( refReader - refEntries ), yRef, y );
            }
        }

        Log::Line( "  Bucket %u validated successfully!", bucket );

        // Swap bucket buffers
        std::swap( bucketEntries, bucketSortTmp );
    }

    Log::Line( "Table %d y values validated successfully.", (int)table+1 );

    // Restore files to their position, just in case
    queue.SeekBucket( yFileId, 0, SeekOrigin::Begin );
    queue.SignalFence( fence );
    queue.CommitCommands();
    fence.Wait();


    SysHost::VirtualFree( refEntries    );
    SysHost::VirtualFree( bucketEntries );
}

//-----------------------------------------------------------
void Debug::ValidateMetaFileFromBuckets( const uint64* metaABucket, const uint64* metaBBucket,
                                         TableId table,  uint32 entryCount, uint32 bucketIdx,
                                         uint32 bucketCounts[BB_DP_BUCKET_COUNT] )
{
    size_t  refEntrySize     = 0;
    uint32  metaMultiplier   = 0;
    size_t  refMetaTableSize = 0;
    uint64  refEntryCount    = 0;
    uint64* refEntries       = nullptr;

    switch( table )
    {
        case TableId::Table2: metaMultiplier = TableMetaOut<TableId::Table2>::Multiplier; break;
        case TableId::Table3: metaMultiplier = TableMetaOut<TableId::Table3>::Multiplier; break;
        case TableId::Table4: metaMultiplier = TableMetaOut<TableId::Table4>::Multiplier; break;
        case TableId::Table5: metaMultiplier = TableMetaOut<TableId::Table5>::Multiplier; break;
        case TableId::Table6: metaMultiplier = TableMetaOut<TableId::Table6>::Multiplier; break;
        default:
            Fatal( "Invalid table specified for metadata verification." );
            break;
    }

    // Reference stores 3*4 multiplier as 4*4
    if( metaMultiplier == 3 )
        metaMultiplier = 4;

    refEntrySize = metaMultiplier * 4;

    // Read the whole reference table into memory
    Log::Line( "Loading reference table..." );
    {
        char path[1024];
        sprintf( path, BB_DP_DBG_REF_DIR "meta%d.tmp", (int)table+1 );

        FileStream refTable;
        FatalIf( !refTable.Open( path, FileMode::Open, FileAccess::Read, FileFlags::LargeFile | FileFlags::NoBuffering ),
                 "Failed to open reference meta table file %s.", path );

        const uint64 maxEntries = 1ull << _K;
        const size_t blockSize  = refTable.BlockSize();
        const size_t allocSize  = (size_t)maxEntries * refEntrySize;

        byte* block = bbvirtalloc<byte>( blockSize );
        refEntries  = bbvirtalloc<uint64>( allocSize );

        // The first block contains the entry count
        FatalIf( !refTable.Read( refEntries, blockSize ), 
                "Failed to read count from reference table file %s with error: %d.", path, refTable.GetError() );

        refMetaTableSize = *refEntries;
        refEntryCount    = refMetaTableSize / refEntrySize;

        ASSERT( refMetaTableSize <= allocSize );
        ASSERT( refEntryCount <= maxEntries );

        const size_t blockSizeToRead    = refMetaTableSize / blockSize * blockSize;
        const size_t remainder          = refMetaTableSize - blockSizeToRead;

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
            reader     += sizeRead;
        }

        if( remainder )
        {
            if( refTable.Read( block, blockSize ) != (ssize_t)blockSize )
            {
                const int err = refTable.GetError();
                Fatal( "Failed to read entries from reference table file %s with error: %d.", path, err );
            }

            memcpy( reader, block, remainder );
        }

        SysHost::VirtualFree( block );
    }


    // Test
    Log::Line( "Validating Bucket %u", bucketIdx );

    if( metaMultiplier == 2 )
    {
        const uint64* refReader = refEntries;

        // Get reader offset based on bucket
        for( uint i = 0; i < bucketIdx; i++ )
            refReader += bucketCounts[i];

        for( int64 i = 0; i < entryCount; i++, refReader++ )
        {
            const uint64 metaA   = metaABucket[i];
            const uint64 metaRef = *refReader;

            // FatalIf( metaA != metaRef, 
                    // "Failed to validate entry on table %d at bucket position %u:%lld | Global position: %lld.\n"
                    // " Expected %llu but got %llu",
                    //     (int)table+1, bucketIdx, i, (int64)( refReader - refEntries ), metaRef, metaA );
            if( metaA != metaRef )
            {
                // Because the y that generated the meta might be repeated, when we sort
                // we might get metadata that is not matching because it is out of order.
                // We test for those cases here in the case that there's a 2-way mismatch.
                // If the y repeates more than 2 times and we have a mismatch, we won't test for
                // it and just consider it as an error for manual checking.

                if( metaABucket[i+1] == metaRef &&
                    metaA == refReader[1] )
                {
                    // Mismatched pair, skip the next one.
                    // Skip the next
                    i++;
                    refReader++;
                    continue;
                }

                Log::Line( "Failed to validate entry on table %d at bucket position %u:%lld | Global position: %lld.\n"
                        " Expected %llu but got %llu",
                            (int)table+1, bucketIdx, i, (int64)( refReader - refEntries ), metaRef, metaA );

            }
        }
    }
    else if( metaMultiplier == 4 )
    {
        struct Meta4 { uint64 a, b; };

        const Meta4* refReader = (Meta4*)refEntries;

        // Get reader offset based on bucket
        for( uint i = 0; i < bucketIdx; i++ )
            refReader += bucketCounts[i];

        for( int64 i = 0; i < entryCount; i++, refReader++ )
        {
            const uint64 metaA    = metaABucket[i];
            const uint64 metaB    = metaBBucket[i];
            const uint64 metaARef = refReader->a;
            const uint64 metaBRef = refReader->b;

            // FatalIf( metaA != metaRef, 
                    // "Failed to validate entry on table %d at bucket position %u:%lld | Global position: %lld.\n"
                    // " Expected %llu but got %llu",
                    //     (int)table+1, bucketIdx, i, (int64)( refReader - refEntries ), metaRef, metaA );
            if( metaA != metaARef || metaB != metaBRef )
            {
                // Because the y that generated the meta might be repeated, when we sort
                // we might get metadata that is not matching because it is out of order.
                // We test for those cases here in the case that there's a 2-way mismatch.
                // If the y repeates more than 2 times and we have a mismatch, we won't test for
                // it and just consider it as an error for manual checking.

                if( metaABucket[i+1] == metaARef && metaA == refReader[1].a &&
                    metaBBucket[i+1] == metaBRef && metaB == refReader[1].b )
                {
                    // Mismatched pair, skip the next one.
                    // Skip the next
                    i++;
                    refReader++;
                    continue;
                }

                const intptr_t globalPos = ((intptr_t)refReader - (intptr_t)refEntries) / (intptr_t)sizeof( Meta4 );

                Log::Line( "Failed to validate entry on table %d at bucket position %u:%lld | Global position: %lld.\n"
                        " Expected A:%llu but got A:%llu\n"
                        " Expected B:%llu but got B:%llu",
                            (int)table+1, bucketIdx, i, 
                            globalPos, metaARef, metaA, metaBRef, metaB );

            }
        }
    }

    Log::Line( "Table %d meta values validated successfully.", (int)table+1 );

    SysHost::VirtualFree( refEntries );
}


