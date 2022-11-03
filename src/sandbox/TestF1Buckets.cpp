#include "Sandbox.h"
#include "plotdisk/jobs/F1GenBucketized.h"
#include "plotdisk/DiskPlotConfig.h"
#include "util/Util.h"
#include "util/Log.h"
#include "plotdisk/DiskPlotDebug.h"
#include "io/FileStream.h"
#include "algorithm/RadixSort.h"

//-----------------------------------------------------------
void TestF1Buckets( ThreadPool& pool, const byte plotId[32], const byte memoId[128] )
{
    const size_t totalEntries    = 1ull << _K;
    const size_t totalBlocks     = totalEntries * sizeof( uint32 ) / kF1BlockSize;
    const size_t entriesPerBlock = kF1BlockSize / sizeof( uint32 );
    
    ASSERT( totalEntries == totalBlocks * entriesPerBlock );
    
    byte*   blocks   = bbvirtalloc<byte>( totalEntries * sizeof( uint32 ) );
    uint32* yBuckets = bbcvirtalloc<uint32>( totalEntries * 2 );
    uint32* xBuckets = yBuckets + totalEntries;

    uint32 bucketCounts[BB_DP_BUCKET_COUNT];

    Log::Line( "Generating in-memory bucketized F1..." );
    F1GenBucketized::GenerateF1Mem( plotId, pool, pool.ThreadCount(), blocks, yBuckets, xBuckets, bucketCounts );
    SysHost::VirtualFree( blocks ); blocks = nullptr;
    
    uint32* yBucketTmp = xBuckets;

    // Now validate it
    uint64  refEntryCount = 0;
    uint64* refEntries    = nullptr;

    // Read the whole reference table into memory
    Log::Line( "Loading reference table..." );
    {
        char path[1024];
        sprintf( path, "%st%d.y.tmp", BB_DP_DBG_REF_DIR, 1 );

        FileStream refTable;
        FatalIf( !refTable.Open( path, FileMode::Open, FileAccess::Read, FileFlags::LargeFile | FileFlags::NoBuffering ),
                 "Failed to open reference table file %s.", path );

        const size_t blockSize  = refTable.BlockSize();
        const uint64 maxEntries = 1ull << _K;

        const size_t allocSize = RoundUpToNextBoundaryT( (size_t)maxEntries * sizeof( uint64 ), blockSize );

        refEntries = bbvirtalloc<uint64>( allocSize );

        // The first block contains the entry count
        FatalIf( refTable.Read( refEntries, blockSize ) != blockSize, 
                "Failed to read count from reference table file %s with error: %d.", path, refTable.GetError() );

        refEntryCount = *refEntries;

        ASSERT( refEntryCount <= maxEntries );

        size_t sizeToRead = allocSize;
        byte*  reader     = (byte*)refEntries;
        while( sizeToRead )
        {
            // The rest of the blocks are entries
            const ssize_t sizeRead = refTable.Read( reader, sizeToRead );
            FatalIf( sizeRead <= 0, "Failed to read entries from reference table file %s with error: %d.", path, refTable.GetError() );

            sizeToRead -= (size_t)sizeRead;
            reader += sizeRead;
        }
    }

    uint32* yReader   = yBuckets;
    const uint64* refReader = refEntries;

    for( uint bucket = 0; bucket < BB_DP_BUCKET_COUNT; bucket++ )
    {
        Log::Line( "Bucket %u", bucket+1 );

        const int64 entryCount = bucketCounts[bucket];
        
        // Sort the bucket
        Log::Line( "  Sorting bucket..." );
        RadixSort256::Sort<BB_MAX_JOBS>( pool, yReader, yBucketTmp, (uint64)entryCount );

        // Start validating
        Log::Line( "  Validating entries...");
        const uint64 bucketMask = ((uint64)bucket) << 32;
        
        for( int64 i = 0; i < entryCount; i++, yReader++, refReader++ )
        {
            const uint64 y      = bucketMask | *yReader;
            const uint64 yRef   = *refReader;

            const uint32 y32    = *yReader;
            const uint32 y32Ref = (uint32)yRef;

            FatalIf( y != yRef, "Failed to validate entry on table %d at bucket position %u:%lld | Global position: %lld",
                     (int)1, bucket, i, (int64)( refReader - refEntries ) );
        }

        Log::Line( "  Bucket %u validated successfully!", bucket+1 );
    }
}