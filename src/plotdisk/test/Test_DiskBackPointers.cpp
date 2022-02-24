#include "io/FileStream.h"
#include "plotdisk/DiskPlotConfig.h"
#include "plotdisk/jobs/IOJob.h"
#include "util/Log.h"

void LoadTable( TableId table, uint32*& lPtr, uint16*& rPtr, uint64 entryCount )
{
    char lPath[512];
    char rPath[512];
    snprintf( lPath, sizeof( lPath ), "/mnt/p5510a/disk_tmp/table_%u_l_0.tmp", table+1 );
    snprintf( rPath, sizeof( rPath ), "/mnt/p5510a/disk_tmp/table_%u_r_0.tmp", table+1 );

    FileStream lFile, rFile;
        
    FatalIf( !lFile.Open( lPath, FileMode::Open,  FileAccess::Read, FileFlags::LargeFile | FileFlags::NoBuffering ), "Failed to open L table." );
    FatalIf( !rFile.Open( rPath, FileMode::Open,  FileAccess::Read, FileFlags::LargeFile | FileFlags::NoBuffering ), "Failed to open R table." );

    lPtr = bbcvirtalloc<uint32>( RoundUpToNextBoundaryT( (size_t)entryCount * sizeof( uint32 ), lFile.BlockSize() ) );
    rPtr = bbcvirtalloc<uint16>( RoundUpToNextBoundaryT( (size_t)entryCount * sizeof( uint16 ), rFile.BlockSize() ) );

    ASSERT( lFile.BlockSize() == rFile.BlockSize() );
    byte* blockBuffer = bbvirtalloc<byte>( lFile.BlockSize() );

    int err = 0;
    FatalIf( !IOJob::ReadFromFile( lFile, (byte*)lPtr, entryCount * sizeof( uint32 ), blockBuffer, lFile.BlockSize(), err ), "Failed to read L table." );
    FatalIf( !IOJob::ReadFromFile( rFile, (byte*)rPtr, entryCount * sizeof( uint16 ), blockBuffer, lFile.BlockSize(), err ), "Failed to read R table." );

    SysHost::VirtualFree( blockBuffer );
}

void TestDiskBackPointers()
{
    uint32 bucketCounts[7][BB_DP_BUCKET_COUNT];
    uint32 ptrTableBucketCounts[7][BB_DP_BUCKET_COUNT];

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
    }

    uint64 tableLengths[7];
    memset( tableLengths, 0, sizeof( tableLengths ) );

    for( int table = (int)TableId::Table7; table > (int)TableId::Table1; table-- )
        for( int bucket = 0; bucket < BB_DP_BUCKET_COUNT; bucket++ )
            tableLengths[table] += ptrTableBucketCounts[table][bucket];

    // Load table 2
    uint64  tableLength = tableLengths[(int)TableId::Table2];
    uint32* lPtr        = nullptr;
    uint16* rPtr        = nullptr;

    Log::Line( "Loading table..." );
    LoadTable( TableId::Table2, lPtr, rPtr, tableLength );

    Log::Line( "Table loaded." );
}