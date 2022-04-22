#include "BucketStream.h"
#include "util/Util.h"

//-----------------------------------------------------------
BucketStream::BucketStream( IStream& baseStream, const size_t sliceSize, const uint32 numBuckets )
    : _baseStream     ( baseStream )
    , _blockBuffers   ( bbvirtallocnuma<byte>( baseStream.BlockSize() * (size_t)numBuckets * numBuckets ) ) // Virtual alloc aligned.
    , _slicePositions     ( bbcalloc<size_t*>( numBuckets ), numBuckets )
    , _blockRemainders( bbcalloc<size_t>( (size_t)numBuckets * numBuckets ) )
    , _sliceSize      ( sliceSize )
    , _bucketCapacity ( sliceSize * numBuckets )
    , _numBuckets     ( numBuckets )
{
    size_t* sliceBuffer = bbcalloc<size_t>( numBuckets * numBuckets );

    for( uint32 i = 0; i < numBuckets; i++ )
    {
        _slicePositions[i] = sliceBuffer;
        _slicePositions[i] = 0;

        sliceBuffer += numBuckets;
    }

    const uint32 numSlices = numBuckets * numBuckets;
    for( uint32 i = 0; i < numSlices; i++ )
        _blockRemainders[i] = 0;
}

//-----------------------------------------------------------
BucketStream::~BucketStream()
{
    free( _slicePositions[0] );
    free( _slicePositions.Ptr() );
    free( _blockRemainders );
    bbvirtfree( _blockBuffers );
}

//-----------------------------------------------------------
void BucketStream::WriteSlices( const void* slices, const uint32* sliceSizes )
{
    const byte* sliceBytes = (byte*)slices;

    IStream&     stream    = _baseStream;
    const size_t blockSize = stream.BlockSize();


    if( _mode == Sequential )
    {
        // Each bucket contains all the slices
        for( uint32 i = 0; i < _numBuckets; i++ )
        {
            int64 offset = (int64)_slicePositions[i][0];
            PanicIf( !stream.Seek( offset, SeekOrigin::Begin ), "Base stream failed to seek." );

            size_t size = sliceSizes[i];
            ASSERT( ((size_t)offset - _bucketCapacity * i) + size < _bucketCapacity );

            // #TODO: save Remainders
            stream.Write( sliceBytes, size )

            _slicePositions[i][0] += size;
        }
    }
    else
    {

    }
    for( uint32 i = 0; i < _numBuckets; i++ )
    {
        stream.Seek( offset )
        const size_t size = sliceSizes[i];

        slices += size;
    }
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

