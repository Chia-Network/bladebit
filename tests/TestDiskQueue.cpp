#include "TestUtil.h"
#include "plotting/DiskQueue.h"
#include "plotting/DiskBucketBuffer.h"
#include "plotting/DiskBuffer.h"
#include "util/VirtualAllocator.h"

constexpr uint32 bucketCount      = 64;
constexpr uint32 entriesPerBucket = 1 << 16;
constexpr uint32 entriesPerSlice  = entriesPerBucket / bucketCount;

static void WriteBucketSlices( DiskBucketBuffer* buf, uint32 bucket, uint32 mask, Span<size_t> sliceSizes );

//-----------------------------------------------------------
TEST_CASE( "disk-slices", "[disk-queue]" )
{
    const char* tempPath = GetEnv( "bb_queue_path", "/Users/harito/.sandbox/plot" );

    DiskQueue queue( tempPath );

    auto buf = std::unique_ptr<DiskBucketBuffer>( DiskBucketBuffer::Create( 
        queue, "slices.tmp", bucketCount, sizeof( uint32 ) * entriesPerSlice,
        FileMode::Create, FileAccess::ReadWrite ) );

    ENSURE( buf.get() );

    {
        VirtualAllocator allocator{};
        buf->ReserveBuffers( allocator );
    }

    size_t _sliceSizes[bucketCount] = {};
    for( uint32 i = 0; i < bucketCount; i++ )
        _sliceSizes[i] = entriesPerSlice * sizeof( uint32 );

    Span<size_t> sliceSizes( _sliceSizes, bucketCount );

    // Write a whole "table"'s worth of data
    for( uint32 b = 0; b < bucketCount; b++ )
    {
        WriteBucketSlices( buf.get(), b, 0, sliceSizes  );
    }

    // Read back
    buf->Swap();
    const uint32 secondMask = 0xF0000000;

    {
        buf->ReadNextBucket();

        for( uint32 b = 0; b < bucketCount; b++ )
        {
            buf->TryReadNextBucket();

            auto input = buf->GetNextReadBufferAs<uint32>();

            const uint32 readMask = b << 16;

            ENSURE( input.Length() == entriesPerBucket );

            // Validate
            for( uint32 i = 0; i < input.Length(); i++ )
            {
                ENSURE( input[i] == (readMask | i ) );
            }

            // Write new bucket
            WriteBucketSlices( buf.get(), b, secondMask, sliceSizes );
        }
    }

    // Read again and validate the second match
    buf->Swap();
    {
        buf->ReadNextBucket();

        for( uint32 b = 0; b < bucketCount; b++ )
        {
            buf->TryReadNextBucket();

            auto input = buf->GetNextReadBufferAs<uint32>();

            const uint32 readMask = secondMask | (b << 16);

            ENSURE( input.Length() == entriesPerBucket );

            // Validate
            for( uint32 i = 0; i < input.Length(); i++ )
            {
                ENSURE( input[i] == (readMask | i ) );
            }
        }
    }

    buf->Swap();
    Log::Line( "Ok" );
}

//-----------------------------------------------------------
TEST_CASE( "disk-buckets", "[disk-queue]" )
{
    const char* tempPath = GetEnv( "bb_queue_path", "/Users/harito/.sandbox/plot" );
    DiskQueue queue( tempPath );

    auto buffer = std::unique_ptr<DiskBuffer>( DiskBuffer::Create( 
        queue, "bucket.tmp",
        bucketCount, sizeof( uint32 ) * entriesPerBucket,
        FileMode::Create, FileAccess::ReadWrite ) );

    ENSURE( buffer );
    {
        VirtualAllocator allocator{};
        buffer->ReserveBuffers( allocator );
    }

    // Write bucket
    {
        for( uint32 b = 0; b < bucketCount; b++ )
        {
            auto bucket = buffer->GetNextWriteBufferAs<uint32>();

            for( uint32 i = 0; i < entriesPerBucket; i++ )
                bucket[i] = b * entriesPerBucket + i;

            buffer->Submit( entriesPerBucket * sizeof( uint32 ) );
        }
    }

    // Read back bucket
    buffer->Swap();

    {
        buffer->ReadNextBucket();
        for( uint32 b = 0; b < bucketCount; b++ )
        {
            buffer->TryReadNextBucket();

            auto bucket = buffer->GetNextReadBufferAs<uint32>();

            // Validate
            ENSURE( bucket.Length() == entriesPerBucket );
            for( uint32 i = 0; i < entriesPerBucket; i++ )
            {
                ENSURE( bucket[i] == b * entriesPerBucket + i );
            }
        }
    }

    Log::Line( "Ok" );
}

//-----------------------------------------------------------
void WriteBucketSlices( DiskBucketBuffer* buf, uint32 bucket, uint32 writeMask, Span<size_t> sliceSizes )
{
    const uint32 base = entriesPerSlice * bucket;

    auto slices = buf->GetNextWriteBufferAs<uint32>();

    for( uint32 slice = 0; slice < bucketCount; slice++ )
    {
        const uint32 mask = writeMask | (slice << 16);

        for( uint32 i = 0; i < entriesPerSlice; i++ )
            slices[i] = mask | (base + i);

        slices = slices.Slice( buf->GetSliceStride() / sizeof( uint32 ) );
    }

    // Submit next buffer
    buf->Submit();
}


