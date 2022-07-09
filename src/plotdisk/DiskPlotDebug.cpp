#include "DiskPlotDebug.h"
#include "io/FileStream.h"
#include "threading/ThreadPool.h"
#include "threading/AutoResetSignal.h"
#include "algorithm/RadixSort.h"
#include "util/Util.h"
#include "util/Log.h"
#include "jobs/IOJob.h"
#include "DiskPlotContext.h"
#include "DiskBufferQueue.h"
#include "plotdisk/k32/FpMatchBounded.inl"

#define BB_DBG_WRITE_LP_BUCKET_COUNTS 1

// #define BB_DBG_READ_LP_BUCKET_COUNTS 1

//-----------------------------------------------------------
void Debug::ValidateYFileFromBuckets( FileId yFileId, ThreadPool& pool, DiskBufferQueue& queue, TableId table, 
                                      uint32 bucketCounts[BB_DP_BUCKET_COUNT] )
{
    // const size_t bucketMaxCount  = BB_DP_MAX_ENTRIES_PER_BUCKET;

    // uint64  refEntryCount = 0;
    // uint64* refEntries    = nullptr;

    // // Read the whole reference table into memory
    // Log::Line( "Loading reference table..." );
    // {
    //     char path[1024];
    //     sprintf( path, "%st%d.y.tmp", BB_DP_DBG_REF_DIR, (int)table+1 );

    //     FileStream refTable;
    //     FatalIf( !refTable.Open( path, FileMode::Open, FileAccess::Read, FileFlags::LargeFile | FileFlags::NoBuffering ),
    //              "Failed to open reference table file %s.", path );

    //     const size_t blockSize  = refTable.BlockSize();
    //     const uint64 maxEntries = 1ull << _K;

    //     const size_t allocSize = (size_t)maxEntries * sizeof( uint64 );

    //     byte* block = bbvirtalloc<byte>( blockSize );
    //     refEntries  = bbvirtalloc<uint64>( allocSize );

    //     // The first block contains the entry count
    //     FatalIf( !refTable.Read( refEntries, blockSize ), 
    //             "Failed to read count from reference table file %s with error: %d.", path, refTable.GetError() );

    //     refEntryCount = *refEntries;
    //     if( table == TableId::Table1 )
    //         ASSERT( refEntryCount == maxEntries );

    //     ASSERT( refEntryCount <= maxEntries );

    //     const size_t totalSize       = RoundUpToNextBoundaryT( (size_t)refEntryCount * ( table == TableId::Table7 ? sizeof( uint32 ) : sizeof( uint64 ) ), blockSize );
    //     const size_t blockSizeToRead = totalSize / blockSize * blockSize;
    //     const size_t remainder       = totalSize - blockSizeToRead;

    //     // Read blocks
    //     size_t sizeToRead = blockSizeToRead;
    //     byte*  reader     = (byte*)refEntries;
    //     while( sizeToRead )
    //     {
    //         // The rest of the blocks are entries
    //         const ssize_t sizeRead = refTable.Read( reader, sizeToRead );
    //         if( sizeRead < 1 )
    //         {
    //             const int err = refTable.GetError();
    //             Fatal( "Failed to read entries from reference table file %s with error: %d.", path, err );
    //         }

    //         sizeToRead -= (size_t)sizeRead;
    //         reader += sizeRead;
    //     }

    //     if( remainder )
    //     {
    //         if( refTable.Read( block, blockSize ) != (ssize_t)blockSize )
    //         {
    //             const int err = refTable.GetError();
    //             Fatal( "Failed to read entries from reference table file %s with error: %d.", path, err );
    //         }

    //         // ASSERT( blockSizeToRead / sizeof( uint64 ) + remainder / sizeof( uint64 ) == refEntryCount );
    //         memcpy( reader, block, remainder );
    //     }

    //     SysHost::VirtualFree( block );
    // }


    // // Alloc a buffer for our buckets
    // const size_t bucketAllocSize = RoundUpToNextBoundaryT( sizeof( uint32 ) * (size_t)BB_DP_MAX_ENTRIES_PER_BUCKET, queue.BlockSize() );
    // uint32* bucketEntries = bbvirtalloc<uint32>( bucketAllocSize );
    // uint32* bucketSortTmp = bbvirtalloc<uint32>( bucketAllocSize );

    // Fence fence;

    // // Load the first bucket
    // queue.SeekBucket( yFileId, 0, SeekOrigin::Begin );
    // queue.ReadFile( yFileId, 0, bucketEntries, bucketCounts[0] * sizeof( uint32 ) );
    // queue.SignalFence( fence );
    // queue.CommitCommands();

    // fence.Wait();
    // fence.Signal(); // Set as signaled initially since we wait for the next bucket in the loop

    // const uint64* refReader = refEntries;
    // const uint32* f7Reader  = (uint32*)refEntries; 

    // for( uint bucket = 0; bucket < BB_DP_BUCKET_COUNT; bucket++ )
    // {
    //     Log::Line( "Bucket %u", bucket );

    //     // Wait for the next bucket to be loaded
    //     fence.Wait();

    //     const int64 entryCount = bucketCounts[bucket];
        
    //     // Sort the bucket
    // // if( table < TableId::Table7 )
    //     {
    //         Log::Line( "  Sorting bucket..." );
    //         RadixSort256::Sort<BB_MAX_JOBS>( pool, bucketEntries, bucketSortTmp, (uint64)entryCount );
    //     }

    //     // Load the next bucket
    //     const uint nextBucket = bucket + 1;
        
    //     if( nextBucket < BB_DP_BUCKET_COUNT )
    //     {
    //         queue.ReadFile( yFileId, nextBucket, bucketSortTmp, bucketCounts[nextBucket] * sizeof( uint32 ) );
    //         queue.SignalFence( fence );
    //         queue.CommitCommands();
    //     }

    //     // Start validating
    //     Log::Line( "  Validating entries...");
    //     if( table < TableId::Table7 )
    //     {
    //         const uint64 bucketMask = ((uint64)bucket) << 32;
    //         uint64 prevRef = 0;
            
    //         for( int64 i = 0; i < entryCount; i++, refReader++ )
    //         {
    //             const uint64 y      = bucketMask | bucketEntries[i];
    //             const uint64 yRef   = *refReader;

    //             // Test for now, since we're not getting all of the reference values in some tables...
    //             if( yRef < prevRef )
    //                 break;
    //             prevRef = yRef;
    //             // if( y == 112675641563 ) BBDebugBreak();

    //             // const uint32 y32    = bucketEntries[i];
    //             // const uint32 y32Ref = (uint32)yRef;

    //             FatalIf( y != yRef, 
    //                     "Failed to validate entry on table %d at bucket position %u:%lld | Global position: %lld.\n"
    //                     " Expected %llu but got %llu",
    //                     (int)table+1, bucket, i, (int64)( refReader - refEntries ), yRef, y );
    //         }
    //     }
    //     else
    //     {
    //         const uint32* refEnd = f7Reader + refEntryCount;
            
    //         for( int64 i = 0; i < entryCount && f7Reader < refEnd; i++, f7Reader++ )
    //         {
    //             const uint32 y      = bucketEntries[i];
    //             const uint32 yRef   = *f7Reader;

    //             FatalIf( y != yRef, 
    //                     "Failed to validate entry on table %d at bucket position %u:%lld | Global position: %lld.\n"
    //                     " Expected %lu but got %lu",
    //                     (int)table+1, bucket, i, (int64)( refReader - refEntries ), yRef, y );
    //         }
    //     }

    //     Log::Line( "  Bucket %u validated successfully!", bucket );

    //     // Swap bucket buffers
    //     std::swap( bucketEntries, bucketSortTmp );
    // }

    // Log::Line( "Table %d y values validated successfully.", (int)table+1 );

    // // Restore files to their position, just in case
    // queue.SeekBucket( yFileId, 0, SeekOrigin::Begin );
    // queue.SignalFence( fence );
    // queue.CommitCommands();
    // fence.Wait();


    // SysHost::VirtualFree( refEntries    );
    // SysHost::VirtualFree( bucketEntries );
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
    
    Log::Line( "Validating metadata for table %d...", (int)table+1 );

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

// Ensure we have an ordered lookup index
//-----------------------------------------------------------
void Debug::ValidateLookupIndex( TableId table, ThreadPool& pool, DiskBufferQueue& queue, const uint32 bucketCounts[BB_DP_BUCKET_COUNT] )
{
    uint64 totalEntries = 0;
    for( uint32 i = 0; i < BB_DP_BUCKET_COUNT; i++ )
        totalEntries += bucketCounts[i];

    // Lod buckets into a single contiguous bucket
    uint32* indices = bbcvirtalloc<uint32>( RoundUpToNextBoundary( totalEntries, (int)queue.BlockSize() ) * 2 );

    Log::Line( "Loading table %d lookup indices...", (int)table );
    {
        const FileId fileId = TableIdToSortKeyId( table );

        uint32* readPtr = indices;

        queue.SeekBucket( fileId, 0, SeekOrigin::Begin );
        queue.CommitCommands();

        for( uint32 bucket = 0; bucket < BB_DP_BUCKET_COUNT; bucket++ )
        {
            queue.ReadFile( fileId, bucket, readPtr, bucketCounts[bucket] * sizeof( uint32 ) );
            readPtr += bucketCounts[bucket];
        }

        Fence fence;
        queue.SignalFence( fence );
        queue.SeekBucket( fileId, 0, SeekOrigin::Begin );
        queue.CommitCommands();
        fence.Wait();
    }

    // Check counts per bucket, before sort
    uint32 bcounts[BB_DP_BUCKET_COUNT];
    {
        memset( bcounts, 0, sizeof( bcounts ) );

        const uint32* reader = indices;

        for( uint32 bucket = 0; bucket < BB_DP_BUCKET_COUNT; bucket++ )
        {
            const uint32 bucketEntryCount = bucketCounts[bucket];

            for( uint32 i = 0; i < bucketEntryCount; i++ )
            {
                const uint b = reader[i] >> ( 32 - kExtraBits );
                ASSERT( b < BB_DP_BUCKET_COUNT );
                bcounts[b]++;
            }

            reader += bucketEntryCount;
        }
    }

    // Sort the entries
    Log::Line( "Sorting indices..." );
    RadixSort256::Sort<BB_MAX_JOBS>( pool, indices, indices + totalEntries, totalEntries );

    // Ensure all indices are ordered
    Log::Line( "Validating indices..." );
    for( uint64 i = 0; i < totalEntries; i++ )
    {
        ASSERT( indices[i] == i );
    }
    Log::Line( "** Validation success **" );

    // Check distribution into buckets
    uint32 counts[BB_DP_BUCKET_COUNT];
    memset( counts, 0, sizeof( counts ) );

    for( uint64 i = 0; i < totalEntries; i++ )
    {
        const uint bucket = indices[i] >> ( 32 - kExtraBits );
        ASSERT( bucket < BB_DP_BUCKET_COUNT );
        counts[bucket]++;
    }

    // Cleanup
    SysHost::VirtualFree( indices );
}

//-----------------------------------------------------------
void Debug::ValidateLinePoints( DiskPlotContext& context, TableId table, uint32 bucketCounts[BB_DPP3_LP_BUCKET_COUNT] )
{
    // Log::Line( "Validating Line points for table %u", table+1 );

    // // Load the reference table
    // uint64  refLPCount    = 0;
    // uint64* refLinePoints = bbcvirtalloc<uint64>( 1ull << _K );

    // byte* blockBuffer = nullptr;

    // {
    //     Log::Line( " Loading reference values..." );
    //     char path[1024];
    //     sprintf( path, "%slp.t%u.tmp", BB_DP_DBG_REF_DIR, table );

    //     FileStream file;
    //     FatalIf( !file.Open( path, FileMode::Open, FileAccess::ReadWrite, FileFlags::NoBuffering | FileFlags::LargeFile  ),
    //         "Failed to open reference LP table @ %s", path );
        
    //     blockBuffer = bbvirtalloc<byte>( file.BlockSize() );

    //     // Read entry count
    //     FatalIf( file.Read( blockBuffer, file.BlockSize() ) != (ssize_t)file.BlockSize(),
    //         "Failed to read reference LP table entry count @ %s", path );
        
    //     refLPCount = *(uint64*)blockBuffer;
    //     int err = 0;
    //     FatalIf( !IOJob::ReadFromFile( file, (byte*)refLinePoints,
    //         refLPCount * sizeof( uint64 ), blockBuffer, file.BlockSize(), err ),
    //         "Failed to read LP table with error: %d : %s", err, path );
    // }

    // bool readBuckets = false;
    // #if BB_DBG_READ_LP_BUCKET_COUNTS
    // {
    //     char path[1024];
    //     sprintf( path, "%slp_t%u_bucket_counts.tmp", BB_DP_DBG_TEST_DIR, table+1 );

    //     FileStream file;
    //     if( file.Exists( path ) )
    //     {
    //         FatalIf( !file.Open( path, FileMode::Open, FileAccess::Read, FileFlags::None ),
    //             "Failed to open file %s for reading.", path );

    //         const size_t readSize = sizeof( uint32 ) * BB_DPP3_LP_BUCKET_COUNT;
    //         FatalIf( (ssize_t)readSize != file.Read( bucketCounts, readSize ),
    //             "Failed to read bucket counts for LP table file %s with error: %d.", path, file.GetError() );

    //         readBuckets = true;
    //     }
    // }
    // #endif
    // #if BB_DBG_WRITE_LP_BUCKET_COUNTS
    // if( !readBuckets )
    // {
    //     // Write This table's LP bucket counts
    //     Log::Line( " Writing table %d LP bucket counts...", table );
        
    //     char path[1024];
    //     sprintf( path, "%slp_t%u_bucket_counts.tmp", BB_DP_DBG_TEST_DIR, table+1 );

    //     FileStream file;
    //     FatalIf( !file.Open( path, FileMode::OpenOrCreate, FileAccess::Write, FileFlags::None ),
    //         "Failed to open file %s for writing.", path );

    //     const size_t writeSize = sizeof( uint32 ) * BB_DPP3_LP_BUCKET_COUNT;
    //     FatalIf( (ssize_t)writeSize != file.Write( bucketCounts, writeSize ),
    //         "Failed to write bucket counts for LP table file %s with error: %d.", path, file.GetError() );
    // }
    // #endif


    // const FileId fileId = TableIdToLinePointFileId( table );
    // DiskBufferQueue& ioQueue = *context.ioQueue;
    // ioQueue.SeekBucket( fileId, 0, SeekOrigin::Begin );
    // ioQueue.CommitCommands();

    // // Should be enough for an allocation size
    // uint64  lpBucketSize = ( ( 1ull << _K ) / BB_DP_BUCKET_COUNT ) * 2;
    // uint64* linePoints   = bbcvirtalloc<uint64>( lpBucketSize );
    // uint64* lpTemp       = bbcvirtalloc<uint64>( lpBucketSize );

    // Fence readFence;
    // readFence.Reset( 0 );

    // const uint64* refLP = refLinePoints;
    // uint64 totalCount = 0;

    // Log::Line( " Validating buckets..." );
    // for( uint32 bucket = 0; bucket < BB_DPP3_LP_BUCKET_COUNT; bucket++ )
    // {
    //     uint64 entryCount = bucketCounts[bucket];

    //     Log::Write( " Bucket %2u... ", bucket ); Log::Flush();

    //     ioQueue.ReadFile( fileId, bucket, linePoints, entryCount * sizeof( uint64 ) );
    //     ioQueue.SignalFence( readFence );
    //     ioQueue.CommitCommands();

    //     readFence.Wait();

    //     // Sort the bucket
    //     RadixSort256::Sort<BB_MAX_JOBS, uint64, 7>( *context.threadPool, linePoints, lpTemp, entryCount );

    //     // Cap the entry count if we're above the ref count
    //     // (bladebit ram is dropping 1 entry for table 7 for some reason, have to check why.)
    //     if( totalCount + entryCount > refLPCount )
    //         entryCount -= ( ( totalCount + entryCount ) - refLPCount );

    //     ASSERT( totalCount + entryCount <= refLPCount );
    //     ASSERT( entryCount <= lpBucketSize );

    //     // Compare values
    //     uint64* lpReader = lpTemp;
    //     for( int64 i = 0; i < (int64)entryCount; i++ )
    //     {
    //         const uint64 ref = refLP   [i];
    //         const uint64 lp  = lpReader[i];
    //         ASSERT( lp == ref );
    //     }

    //     refLP += entryCount;
    //     totalCount += entryCount;

    //     Log::Line( "OK" );
    // }

    // ASSERT( totalCount == refLPCount );
    // Log::Line( "Table %u validated successfully!", table+1 );

    // ioQueue.SeekBucket( fileId, 0, SeekOrigin::Begin );
    // ioQueue.CommitCommands();

    // SysHost::VirtualFree( refLinePoints );
    // SysHost::VirtualFree( blockBuffer   );
    // SysHost::VirtualFree( linePoints    );
    // SysHost::VirtualFree( lpTemp        );
}

//-----------------------------------------------------------
// void ValidatePark7( DiskBufferQueue& ioQueue, uint64 park7Size )


//-----------------------------------------------------------
void Debug::WriteTableCounts( const DiskPlotContext& cx )
{
    // Write bucket counts
    FileStream bucketCounts, tableCounts, backPtrBucketCounts;

    if( bucketCounts.Open( BB_DP_DBG_TEST_DIR BB_DP_DBG_READ_BUCKET_COUNT_FNAME, FileMode::Create, FileAccess::Write ) )
    {
        if( bucketCounts.Write( cx.bucketCounts, sizeof( cx.bucketCounts ) ) != sizeof( cx.bucketCounts ) )
            Log::Error( "Failed to write to bucket counts file." );
    }
    else
        Log::Error( "Failed to open bucket counts file." );

    if( tableCounts.Open( BB_DP_DBG_TEST_DIR BB_DP_TABLE_COUNTS_FNAME, FileMode::Create, FileAccess::Write ) )
    {
        if( tableCounts.Write( cx.entryCounts, sizeof( cx.entryCounts ) ) != sizeof( cx.entryCounts ) )
            Log::Error( "Failed to write to table counts file." );
    }
    else
        Log::Error( "Failed to open table counts file." );

    if( backPtrBucketCounts.Open( BB_DP_DBG_TEST_DIR BB_DP_DBG_PTR_BUCKET_COUNT_FNAME, FileMode::Create, FileAccess::Write ) )
    {
        if( backPtrBucketCounts.Write( cx.ptrTableBucketCounts, sizeof( cx.ptrTableBucketCounts ) ) != sizeof( cx.ptrTableBucketCounts ) )
            Log::Error( "Failed to write to back pointer bucket counts file." );
    }
    else
        Log::Error( "Failed to open back pointer bucket counts file." );
}

//-----------------------------------------------------------
bool Debug::ReadTableCounts( DiskPlotContext& cx )
{
    #if( BB_DP_DBG_SKIP_PHASE_1 || BB_DP_P1_SKIP_TO_TABLE || BB_DP_DBG_SKIP_TO_C_TABLES )

        FileStream bucketCounts, tableCounts, backPtrBucketCounts;

        if( bucketCounts.Open( BB_DP_DBG_TEST_DIR BB_DP_DBG_READ_BUCKET_COUNT_FNAME, FileMode::Open, FileAccess::Read ) )
        {
            if( bucketCounts.Read( cx.bucketCounts, sizeof( cx.bucketCounts ) ) != sizeof( cx.bucketCounts ) )
            {
                Log::Error( "Failed to read from bucket counts file." );
                return false;
            }
        }
        else
        {
            Log::Error( "Failed to open bucket counts file." );
            return false;
        }

        if( tableCounts.Open( BB_DP_DBG_TEST_DIR BB_DP_TABLE_COUNTS_FNAME, FileMode::Open, FileAccess::Read ) )
        {
            if( tableCounts.Read( cx.entryCounts, sizeof( cx.entryCounts ) ) != sizeof( cx.entryCounts ) )
            {
                Log::Error( "Failed to read from table counts file." );
                return false;
            }
        }
        else
        {
            Log::Error( "Failed to open table counts file." );
            return false;
        }

        if( backPtrBucketCounts.Open( BB_DP_DBG_TEST_DIR BB_DP_DBG_PTR_BUCKET_COUNT_FNAME, FileMode::Open, FileAccess::Read ) )
        {
            if( backPtrBucketCounts.Read( cx.ptrTableBucketCounts, sizeof( cx.ptrTableBucketCounts ) ) != sizeof( cx.ptrTableBucketCounts ) )
            {
                Fatal( "Failed to read from pointer bucket counts file." );
            }
        }
        else
        {
            Fatal( "Failed to open pointer bucket counts file." );
        }

        return true;
    #else
        return false;
    #endif
}

//-----------------------------------------------------------
void Debug::DumpDPUnboundedY( const TableId table, const uint32 bucket, const DiskPlotContext& context, const Span<uint64> y )
{
    if( y.Length() < 1 || table < TableId::Table2 )
        return;

    char path[1024];
    sprintf( path, "%st%d.y-dp-unbounded.tmp", BB_DP_DBG_REF_DIR, (int)table+1 );
    
    const FileMode mode = bucket > 0 ? FileMode::Open : FileMode::Create;
    
    FileStream file;
    FatalIf( !file.Open( path, mode, FileAccess::Write ),
        "Failed to open '%s' for writing.", path );

    FatalIf( !file.Seek( (int64)file.Size(), SeekOrigin::Begin ), "Failed to seek file '%s'.", path );
    
    size_t sizeWrite = y.Length() * sizeof( uint64 );

    Span<uint64> yWrite = y;
    while( sizeWrite )
    {
        const ssize_t written = file.Write( yWrite.Ptr(), sizeWrite );
        FatalIf( written <= 0, "Failed to write with eerror %d to file '%s'.", file.GetError(), path );

        sizeWrite -= (size_t)written;

        yWrite = yWrite.Slice( (size_t)written/sizeof( uint64 ) );
    }
}

//-----------------------------------------------------------
void Debug::LoadDPUnboundedY( const TableId table, Span<uint64>& inOutY )
{
    // It's written unaligned for now, so we can determine the length by its size
    char path[1024];
    sprintf( path, "%st%d.y-dp-unbounded.tmp", BB_DP_DBG_REF_DIR, (int)table+1 );
    
    Log::Line( " Loading reference disk-plot Y table at '%s'.", path );

    FileStream file;
    FatalIf( !file.Open( path, FileMode::Open, FileAccess::Read ), 
        "Failed to open file '%s'.", path );

    const uint64 entryCount = file.Size() / sizeof( uint64 );
    ASSERT( entryCount > 0 );

    if( inOutY.values == nullptr )
        inOutY = bbcvirtallocboundednuma_span<uint64>( entryCount );
    else
    {
        FatalIf( entryCount > inOutY.Length(), "Y buffer too small." );
    }

    void* block = bbvirtallocbounded( file.BlockSize() );

    int err;
    FatalIf( !IOJob::ReadFromFile( file, inOutY.Ptr(), file.Size(), block, file.BlockSize(), err ),
        "Error %d when reading from file '%s'.", err, path );

    bbvirtfreebounded( block );
}
