#include "io/FileStream.h"
#include "diskplot/DiskPlotConfig.h"
#include "diskplot/jobs/IOJob.h"
#include "algorithm/RadixSort.h"

static uint32 bucketCounts[7][BB_DP_BUCKET_COUNT];
static uint32 ptrTableBucketCounts[7][BB_DP_BUCKET_COUNT];
static uint64 tableEntryCounts[7];

uint64* LoadLookupMap( TableId table, uint32 bucket, uint64& outEntryCount )
{
    char path[512];
    snprintf( path, sizeof( path ), "/mnt/p5510a/disk_tmp/table_%d_map_%d.tmp", table+1, bucket );

    const uint64 tableLength = tableEntryCounts[(int)table];
    uint64 entryCount  = ( 1ull << _K ) / BB_DP_BUCKET_COUNT;
    
    if( bucket == BB_DP_BUCKET_COUNT-1 )
        entryCount = tableLength - ( entryCount * (BB_DP_BUCKET_COUNT-1));

    outEntryCount = entryCount;

    FileStream file;

    FatalIf( !file.Open( path, FileMode::Open,  FileAccess::Read, FileFlags::LargeFile | FileFlags::NoBuffering ), "Failed to open map table." );
    uint64* ptr = bbcvirtalloc<uint64>( RoundUpToNextBoundaryT( (size_t)entryCount * sizeof( uint64 ), file.BlockSize() ) );

    byte* blockBuffer = bbvirtalloc<byte>( file.BlockSize() );

    int err = 0;
    FatalIf( !IOWriteJob::ReadFromFile( file, (byte*)ptr, entryCount * sizeof( uint64 ), blockBuffer, file.BlockSize(), err ), "Failed to read map table." );

    SysHost::VirtualFree( blockBuffer );
    return ptr;
}

void ReadEntryCounts()
{
    FileStream backPtrBucketCounts;
    if( backPtrBucketCounts.Open( BB_DP_DBG_TEST_DIR BB_DP_DBG_PTR_BUCKET_COUNT_FNAME, FileMode::Open, FileAccess::Read ) )
    {
        if( backPtrBucketCounts.Read( ptrTableBucketCounts, sizeof( ptrTableBucketCounts ) ) != sizeof( ptrTableBucketCounts ) )
            Fatal( "Failed to read from pointer bucket counts file." );
    }
    else
    {
        Fatal( "Failed to open pointer bucket counts file." );
    }

    FileStream bucketCountsFile;

    if( bucketCountsFile.Open( BB_DP_DBG_TEST_DIR BB_DP_DBG_READ_BUCKET_COUNT_FNAME, FileMode::Open, FileAccess::Read ) )
    {
        if( bucketCountsFile.Read( bucketCounts, sizeof( bucketCounts ) ) != sizeof( bucketCounts ) )
            Fatal( "Failed to read from bucket counts file." );
    }
    else
    {
        Fatal( "Failed to open bucket counts file." );
    }

    for( TableId table = TableId::Table1; table <= TableId::Table7; table = table+1 )
    {
        uint64 tableEntryCount = 0;

        for( uint bucket = 0; bucket < BB_DP_BUCKET_COUNT; bucket++ )
            tableEntryCount += bucketCounts[(int)table][bucket];

        tableEntryCounts[(int)table] = tableEntryCount;
    }
}

void TestLookupMaps()
{
    ReadEntryCounts();
    
    Log::Line( "Loading table..." );
    const TableId table       = TableId::Table6;
    const uint32 targetBucket = 1;

    uint64 entryCount;
    uint64* entries = LoadLookupMap( table, targetBucket, entryCount );

    uint64* tmpTable  = bbcvirtalloc<uint64>( entryCount );

    ThreadPool pool( 64 );

    Log::Line( "Sorting table..." );
    RadixSort256::Sort<64, uint64, 4>( pool, entries, tmpTable, entryCount );

    const uint32 bucket = ((uint32)*entries) >> 26;
    ASSERT( bucket == targetBucket );

    uint32* map  = (uint32*)tmpTable;
    uint32* orig = map + entryCount;

    {
        uint64*       r     = entries;
        const uint64* end   = r + entryCount;
        uint32*       wMap  = map;
        uint32*       wOrig = orig;

        do {
            uint64 e = *r++;
            *wMap++  = (uint32)(e >> 32);
            *wOrig++ = (uint32)e;

        } while( r < end );
    }

    Log::Line( "Validating origin indices..." );
    {
        for( uint64 i = 1; i < entryCount; i++ )
            ASSERT( orig[i] > orig[i-1] );
    }

    Log::Line( "Done." );
    
}