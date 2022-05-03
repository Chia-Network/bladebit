#include "TestUtil.h"
#include "io/BucketStream.h"
#include "io/HybridStream.h"    // #TODO: Use/Implement a MemoryStream instead.

const uint32 k                = 24;
const uint32 entryCount       = 1ull << k;

//-----------------------------------------------------------
template<uint32 _numBuckets>
void TestBuckets( ThreadPool& pool, uint32* entries, void* cache, const size_t cacheSize )
{
    const uint32 entriesPerBucket = entryCount / _numBuckets;
    const uint32 entriesPerSlice  = entriesPerBucket / _numBuckets;
    ASSERT( entriesPerSlice * _numBuckets == entriesPerBucket );

    // Generate entries in sequence
    AnonMTJob::Run( pool, [=]( AnonMTJob* self ){

        uint32 count, offset, end;
        GetThreadOffsets( self, entryCount, count, offset, end );

        for( uint32 i = offset; i < end; i++ )
            entries[i] = i;
    });

#if PLATFORM_IS_WINDOWS
    const char* path = "nul";
#else
    const char* path = "/dev/null";
#endif

    HybridStream memStream;
    FatalIf( !memStream.Open( cache, cacheSize, path, FileMode::Create, FileAccess::ReadWrite, FileFlags::LargeFile ),
        "Failed to open test file." );

    BucketStream stream( memStream, entriesPerSlice * sizeof( uint32 ), _numBuckets );

    // Write our entries (which are in ordinal sequence) sequential mode
    for( uint32 bucket = 0; bucket = _numBuckets; bucket++ )
    {

    }
}

//-----------------------------------------------------------
TEST_CASE( "bucket-stream", "[unit-core]" )
{
    SysHost::InstallCrashHandler();




    uint32* entries = bbcalloc<uint32>( entryCount );



//    ThreadPool pool( SysHost::GetLogicalCPUCount() );
//
//    AnonMTJob::Run( pool, [=]( AnonMTJob* self ){
//
//        uint64 count, offset, end;
//        GetThreadOffsets( self, entryCount, count, offset, end );
//    });
}