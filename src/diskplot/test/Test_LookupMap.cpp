#include "io/FileStream.h"
#include "diskplot/DiskPlotConfig.h"
#include "diskplot/jobs/IOJob.h"
#include "algorithm/RadixSort.h"
#include "util/BitField.h"

static uint32 bucketCounts[7][BB_DP_BUCKET_COUNT];
static uint32 ptrTableBucketCounts[7][BB_DP_BUCKET_COUNT];
static uint64 tableEntryCounts[7];

uint64* LoadLookupMap( TableId table, uint32 bucket, uint64& outEntryCount, uint64* map = nullptr )
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

    if( map == nullptr )
        map = bbcvirtalloc<uint64>( RoundUpToNextBoundaryT( (size_t)entryCount * sizeof( uint64 ), file.BlockSize() ) );

    byte* blockBuffer = bbvirtalloc<byte>( file.BlockSize() );

    int err = 0;
    FatalIf( !IOWriteJob::ReadFromFile( file, (byte*)map, entryCount * sizeof( uint64 ), blockBuffer, file.BlockSize(), err ), "Failed to read map table." );

    SysHost::VirtualFree( blockBuffer );
    return map;
}

void StripLookupMap( ThreadPool& pool, uint64 entryCount, uint64* map, uint64* tmp, uint32* outMap )
{
    RadixSort256::Sort<64, uint64, 4>( pool, map, tmp, entryCount );

    const uint64* end = map + entryCount;
    
    do {
        *outMap++ = (uint32)( (*map++) >> 32);

    } while( map < end );
}

void LoadPairsTable( TableId table, uint32*& lPtr, uint16*& rPtr )
{
    const uint64 entryCount = tableEntryCounts[(int)table];

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
    FatalIf( !IOWriteJob::ReadFromFile( lFile, (byte*)lPtr, entryCount * sizeof( uint32 ), blockBuffer, lFile.BlockSize(), err ), "Failed to read L table." );
    FatalIf( !IOWriteJob::ReadFromFile( rFile, (byte*)rPtr, entryCount * sizeof( uint16 ), blockBuffer, lFile.BlockSize(), err ), "Failed to read R table." );

    SysHost::VirtualFree( blockBuffer );
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

void PruneTablesMemory()
{
    const size_t bitFieldSize = 1 GB + 100 MB;

    uint64* bitFields[2];
    bitFields[0] = bbvirtalloc<uint64>( bitFieldSize );
    bitFields[1] = bbvirtalloc<uint64>( bitFieldSize );
    
    uint64 bucketLength64 = ( 1ull << _K ) / BB_DP_BUCKET_COUNT;

    uint64* mapBuffer = bbcvirtalloc<uint64>( bucketLength64 * (BB_DP_BUCKET_COUNT + 1) );
    uint64* tmpMap    = bbcvirtalloc<uint64>( bucketLength64 * 2 );
    uint32* sortedMap = bbcvirtalloc<uint32>( bucketLength64 * (BB_DP_BUCKET_COUNT + 1) );
    
    ThreadPool pool( 64 );
    
    for( TableId table = TableId::Table7; table > TableId::Table2; table = table-1 )
    {
        Log::Line( "[Prunning table %u]", table );

        const uint64 tableEntryCount = tableEntryCounts[(int)table];

        BitField leftMarks ( bitFields[0] );
        BitField rightMarks( bitFields[1] );

        // Load map
        uint32* map = nullptr;
        if( table < TableId::Table7 )
        {
            Log::Line( " Loading map..." );

            uint64* mapRead = mapBuffer;
            uint64* tmp     = tmpMap;

            uint32* rSortedMap = map = sortedMap;

            uint64 mapEntryCount = 0;

            for( uint32 bucket = 0; bucket < BB_DP_BUCKET_COUNT; bucket++ )
            {
                uint64 bucketEntryCount;
                LoadLookupMap( table, bucket, bucketEntryCount, mapRead );
                
                StripLookupMap( pool, bucketEntryCount, mapRead, tmpMap, rSortedMap );

                mapRead       += bucketEntryCount;
                rSortedMap    += bucketEntryCount;
                mapEntryCount += bucketEntryCount;
            }

            ASSERT( mapEntryCount = tableEntryCount );
        }

        // Load pairs
        Log::Line( " Loading pairs..." );
        uint32* left  = nullptr;
        uint16* right = nullptr;
        LoadPairsTable( table, left, right );

        // Change pairs from bucket-relative to absolute indices
        // (Do as a separate step for ease of inspection of the next step)
        // Log::Line( " Setting pairs to absolute indices..." );
        // {
        //     uint64 lTableOffset = 0;

        //     uint32* l = left;
        //     uint16* r = right;

        //     for( int32 bucket = 0; bucket < (int32)BB_DP_BUCKET_COUNT; bucket++ )
        //     {
        //         const int64 pairBucketLength = (int64)ptrTableBucketCounts[(int)table][bucket];
                
        //         for( int64 i = 0; i < pairBucketLength; i++ )
        //         {
        //             l[i] += lTableOffset;
        //             r[i] += l[i];

        //             const uint64 l = (uint64)left[i] + lTableOffset;
        //             const uint64 r = l + right[i];

        //             leftMarks.Set( l );
        //             leftMarks.Set( r ); 
        //         }

        //         l += pairBucketLength;
        //         r += pairBucketLength;

        //         lTableOffset += bucketCounts[(int)table-1][bucket];
        //     }
        // }
        
        // Mark table
        Log::Line( " Marking table..." );
        {
            const int64 length = (int64)tableEntryCount;

            uint64 lTableOffset = 0;

            int64 i = 0;
            for( int32 bucket = 0; bucket < (int32)BB_DP_BUCKET_COUNT; bucket++ )
            {
                const int64 pairBucketLength = (int64)ptrTableBucketCounts[(int)table][bucket];

                for( const int64 end = i+pairBucketLength; i < end; i++ )
                {
                    if( table < TableId::Table7 )
                    {
                        const uint64 rTableIdx = map[i];
                        if( !rightMarks.Get( rTableIdx ) )
                            continue;
                    }

                    const uint64 l = (uint64)left[i] + lTableOffset;
                    const uint64 r = l + right[i];

                    leftMarks.Set( l );
                    leftMarks.Set( r ); 
                }

                lTableOffset += bucketCounts[(int)table-1][bucket];
            }
        }

        // Validate
        Log::Line( " Validating..." );
        {
            const int64 lTableLength = (int64)tableEntryCounts[(int)table-1];
            int64 markedEntryCount = 0;

            for( int64 i = 0; i < lTableLength; i++ )
            {
                if( leftMarks.Get( i ) )
                    markedEntryCount++;
            }

            Log::Line( " Table %u: %llu / %llu : %.2lf%%", table, markedEntryCount, lTableLength,
                ((double)markedEntryCount / lTableLength ) * 100.0 );
            Log::Line( "" );
        }

        SysHost::VirtualFree( left  );
        SysHost::VirtualFree( right );

        // Swap and zero-out left bit field
        std::swap( bitFields[0], bitFields[1] );
        memset( bitFields[0], 0, bitFieldSize );
    }
}

void TestLookupMaps()
{
    ReadEntryCounts();
    PruneTablesMemory();
    return;
    PruneTablesMemory();
    
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