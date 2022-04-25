#include "BucketStream.h"
#include "util/Util.h"

//-----------------------------------------------------------
BucketStream::BucketStream( IStream& baseStream, const size_t bucketMaxSize, const uint32 numBuckets )
    : _baseStream     ( baseStream )
    , _sliceSizes     ( bbcalloc<Slice*>( numBuckets ), numBuckets )
    , _bucketCapacity ( bucketMaxSize )
    , _numBuckets     ( numBuckets )
{
    Slice* sliceBuffer = bbcalloc<Slice>( numBuckets * numBuckets );

    for( uint32 i = 0; i < numBuckets; i++ )
    {
        _sliceSizes[i] = sliceBuffer;
        _sliceSizes[i] = 0;

        sliceBuffer += numBuckets;
    }
}

//-----------------------------------------------------------
BucketStream::~BucketStream()
{
    free( _sliceSizes[0] );
    free( _sliceSizes.Ptr() );
}

//-----------------------------------------------------------
void BucketStream::WriteBucketSlices( const void* slices, const uint32* sliceSizes )
{
    const byte* sliceBytes = (byte*)slices;

    IStream&     stream    = _baseStream;
    const size_t blockSize = stream.BlockSize();

    if( _mode == Sequential )
    {
        // Write slices across all buckets
        for( uint32 i = 0; i < _numBuckets; i++ )
        {
            const int64 offset = i * (int64)_bucketCapacity + (int64)_sliceSizes[i][0].size;
            PanicIf( !stream.Seek( offset, SeekOrigin::Begin ), "Base stream failed to seek." );

            const size_t size = sliceSizes[i];
            ASSERT( ((size_t)offset - _bucketCapacity * i) + size < _bucketCapacity );

            // #TODO: Loop write
            PanicIf( stream.Write( sliceBytes, size ) != (ssize_t)size, "Failed to write slice to base stream." );
            _sliceSizes[i][0].size += size;
        }
    }
    else
    {
        // Write the whole thing then seek to the start of the next bucket
        size_t totalSize = sliceSizes[0];
        for( uint32 i = 1; i < _numBuckets; i++ )
            totalSize += sliceSizes[i];

        ASSERT( totalSize <= _bucketCapacity );

        PanicIf( stream.Write( sliceBytes, totalSize ) != (ssize_t)totalSize, "Failed to write bucket to base stream." );

        // Save size written
        for( uint32 i = 0; i < _numBuckets; i++ )
            _sliceSizes[_bucket][i] = sliceSizes[i];
    }

    _bucket++;
}

//-----------------------------------------------------------
void BucketStream::ReadBucket( const size_t size, void* readBuffer )
{
    ASSERT( _bucket < _numBuckets );
    ASSERT( size );
    ASSERT( size == _sliceSizes[_bucket][0] );
    ASSERT( readBuffer );

    if( _mode == Sequential )
    {
        // Offset to the start of the bucket
        const int64 offset = (int64)_bucket * (int64)_bucketCapacity;
        PanicIf( !_baseStream.Seek( offset, SeekOrigin::Begin ), "Failed to seek to bucket %u.", (uint32)_bucket );

        // Whole bucket ready to read
        PanicIf( _baseStream.Read( readBuffer, size ) != (ssize_t)size, "Failed to read bucket %u.", (uint32)_bucket );
    }
    else
    {
        // Read all slices
        byte* writer = (byte*)readBuffer;

        for( uint32 i = 0; i < _numBuckets; i++ )
        {
            const size_t size = _sliceSizes[_bucket][i];

            

            writer += size;
        }
    }

    _bucket++;
}

//-----------------------------------------------------------
ssize_t BucketStream::Read( void* buffer, size_t size )
{
    ASSERT( 0 );
    return 0;       // Can only read bucket
}

//-----------------------------------------------------------
ssize_t BucketStream::Write( const void* buffer, size_t size )
{
    ASSERT( 0 );
    return 0;       // Can only write bucket
}

//-----------------------------------------------------------
bool BucketStream::Seek( int64 offset, SeekOrigin origin )
{
    return origin == SeekOrigin::Begin && offset == 0;
}

//-----------------------------------------------------------
bool BucketStream::Flush()
{
    return false;
}

//-----------------------------------------------------------
inline size_t BucketStream::BlockSize() const
{
    return _baseStream.BlockSize();
}

//-----------------------------------------------------------
ssize_t BucketStream::Size()
{
    return 0;
}

//-----------------------------------------------------------
bool BucketStream::Truncate( const ssize_t length )
{
    return false;
}

//-----------------------------------------------------------
int BucketStream::GetError()
{
    return 0;
}

