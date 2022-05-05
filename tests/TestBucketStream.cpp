#include "TestUtil.h"
#include "io/BucketStream.h"
#include "io/HybridStream.h"    // #TODO: Use/Implement a MemoryStream instead.

const uint32 k            = 24;
const uint32 entryCount   = 1ull << k;

const size_t cacheSize    = entryCount * sizeof( uint32 );
void*        cache        = bbmalloc<void>( cacheSize  );
uint32*      bucketBuffer = bbcalloc<uint32>( entryCount );

//-----------------------------------------------------------
template<uint32 _numBuckets>
void TestBuckets()
{
    const uint32 entriesPerBucket = entryCount / _numBuckets;
    const uint32 entriesPerSlice  = entriesPerBucket / _numBuckets;
    ASSERT( entriesPerSlice * _numBuckets == entriesPerBucket );

    Log::Line( "Testing %u buckets", _numBuckets );
    Log::Line( "  Entries/bucket: %u.", entriesPerBucket );
    Log::Line( "  Entries/slice : %u.", entriesPerSlice );

#if PLATFORM_IS_WINDOWS
    const char* path = "nul";
#else
    const char* path = "/dev/null";
#endif

    HybridStream memStream;
    FatalIf( !memStream.Open( cache, cacheSize, path, FileMode::Create, FileAccess::ReadWrite, FileFlags::LargeFile ),
        "Failed to open test file." );

    BucketStream stream( memStream, entriesPerSlice * sizeof( uint32 ), _numBuckets );

    uint32 sliceSizes[_numBuckets];
    for( uint32 slice = 0; slice < _numBuckets; slice++ )
        sliceSizes[slice] = entriesPerSlice * sizeof( uint32 );

    auto WriteSlice = [&]( const uint32 slice ) {

        uint32* writer = bucketBuffer;

        for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
        {
            const uint32 entryOffset = bucket * entriesPerBucket + slice * entriesPerSlice;

            for( uint32 i = 0; i < entriesPerSlice; i++ )
                writer[i] = entryOffset + i;

            writer += entriesPerSlice;
        }

        stream.WriteBucketSlices( bucketBuffer, sliceSizes );
    };

    // Write our entries in sequential mode (entries will contain valuies 0..numEntries-1)
    for( uint32 slice = 0; slice < _numBuckets; slice++ )
        WriteSlice( slice );

    // Read it back, still in sequential mode
    ENSURE( stream.Seek( 0, SeekOrigin::Begin ) );

    for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
    {
        stream.ReadBucket( entriesPerBucket * sizeof( uint32 ), bucketBuffer );

        uint32 e = bucket * entriesPerBucket;

        // Validate read entries
        for( uint32 i = 0; i < entriesPerBucket; i++, e++ )
        {
            ENSURE( e == bucketBuffer[i] );
        }

        // Write in interleaved mode now. This will write all slices
        // to the space reserved for the bucket we just read
        WriteSlice( bucket );
    }

    // Now read in interleaved mode
    ENSURE( stream.Seek( 0, SeekOrigin::Begin ) );

    for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
    {
        stream.ReadBucket( entriesPerBucket * sizeof( uint32 ), bucketBuffer );

        uint32 e = bucket * entriesPerBucket;

        // Validate entries
        for( uint32 i = 0; i < entriesPerBucket; i++, e++ )
        {
            ENSURE( e == bucketBuffer[i] );
        }

        // Write in sequential mode again
        WriteSlice( bucket );
    }

    // Finally, read it back, still in sequential mode again and validate
    ENSURE( stream.Seek( 0, SeekOrigin::Begin ) );

    for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
    {
        stream.ReadBucket( entriesPerBucket * sizeof( uint32 ), bucketBuffer );

        uint32 e = bucket * entriesPerBucket;

        // Validate read entries
        for( uint32 i = 0; i < entriesPerBucket; i++, e++ )
        {
            ENSURE( e == bucketBuffer[i] );
        }
    }
}

//-----------------------------------------------------------
TEST_CASE( "bucket-stream", "[unit-core]" )
{
    SysHost::InstallCrashHandler();

    TestBuckets<64>();
    TestBuckets<128>();
    TestBuckets<256>();
    TestBuckets<512>();
    TestBuckets<1024>();
}