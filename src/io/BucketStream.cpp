#include "BucketStream.h"
#include "util/Util.h"

 //-----------------------------------------------------------
 BucketStream::BucketStream( IStream& baseStream, const size_t sliceSize, const uint32 numBuckets )
     : _baseStream     ( baseStream )
     , _slices         ( bbcalloc<size_t*>( numBuckets ), numBuckets )
     , _sliceCapacity  ( sliceSize )
     , _bucketCapacity ( sliceSize * numBuckets )
     , _numBuckets     ( numBuckets )
 {
    const size_t sliceCount = numBuckets * (size_t)numBuckets;
    size_t* sliceBuffer = bbcalloc<size_t>( sliceCount );
    ZeroMem( sliceBuffer, sliceCount );

    for( uint32 i = 0; i < numBuckets; i++ )
    {
        _slices[i] = sliceBuffer;
        sliceBuffer += numBuckets;
    }
 }

 //-----------------------------------------------------------
 BucketStream::~BucketStream()
 {
     free( _slices[0] );
     free( _slices.Ptr() );
 }

 //-----------------------------------------------------------
 void BucketStream::WriteBucketSlices( const void* slices, const uint32* sliceSizes )
 {
    const byte* sliceBytes = (byte*)slices;

    IStream& stream = _baseStream;

    if( _mode == Sequential )
    {
        // Write slices across all buckets. That is, each bucket contains its own slices
        for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
        {
            const int64 offset = (int64)( bucket * _bucketCapacity + _slice * _sliceCapacity ); //bucket * (int64)_bucketCapacity + (int64)_slices[i][0].size;
            PanicIf( !stream.Seek( offset, SeekOrigin::Begin ), "Base stream failed to seek." );

            const size_t size = sliceSizes[bucket];
            ASSERT( size <= _sliceCapacity );

            // #TODO: Loop write, or use IOJob
            PanicIf( stream.Write( sliceBytes, size ) != (ssize_t)size, "Failed to write slice to base stream." );

            _slices[bucket][_slice] += size;
            sliceBytes += size;
        }

        _slice++;
    }
    else
    {
        // Interleaved writes. This means that we will fill each bucket
        // with slices that belong to all buckets, not just he bukcket's own slices.
        // ie. Bucket 0 will have slices from bucket, 0, 1, 2... and so on.

        // #NOTE: We can't just write the whole bucket-full because we need to have each slice
        //        separated by its full slice capacity so that we don't overwrite any data
        //        during the next sequential write.
        const size_t bucketOffset = _slice * _bucketCapacity;
        for( uint32 slice = 0; slice < _numBuckets; slice++ )
        {
            const int64  offset = (int64)( bucketOffset + slice * _sliceCapacity );
            PanicIf( !stream.Seek( offset, SeekOrigin::Begin ), "Base stream failed to seek." );

            const size_t size = sliceSizes[slice];
            ASSERT( size <= _sliceCapacity );

            // #TODO: Loop write, or use IOJob
            PanicIf( stream.Write( sliceBytes, size ) != (ssize_t)size, "Failed to write slice to base stream." );
            _slices[_bucket][slice] += size;
            sliceBytes += size;
        }

        _bucket++;
     }
 }

 //-----------------------------------------------------------
 void BucketStream::ReadBucket( const size_t size, void* readBuffer )
 {
    ASSERT( _bucket < _numBuckets );
    ASSERT( size );
    ASSERT( size == _slices[_bucket][0] );
    ASSERT( readBuffer );

    byte* buffer = (byte*)readBuffer;
    IStream& stream = _baseStream;

    if( _mode == Sequential )
    {
        // Read all slices from the same bucket
        const size_t _bucketStart = _bucket * _bucketCapacity;

        for( uint32 slice = 0; slice < _numBuckets; slice++ )
        {
            const int64 offset = (int64)(_bucketStart + slice * _sliceCapacity );
            PanicIf( !stream.Seek( offset, SeekOrigin::Begin ), "Failed to seek for reading." );

            // #TODO: Loop read, or use IOJob
            const size_t  sliceSize = _slices[_bucket][slice];
            const ssize_t sizeRead  = stream.Read( buffer, sliceSize );
            PanicIf( sizeRead != (ssize_t)sliceSize, "Failed to read slice %u of bucket %u.", slice, (uint32)_bucket );

            buffer += sliceSize;
        }

        _bucket++;
     }
    else
    {
        // Read a whole bucket's worth of bytes by reading slices spread across all buckets
        for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
        {
            const int64 offset = (int64)( bucket * _bucketCapacity + _slice * _sliceCapacity );
            PanicIf( !_baseStream.Seek( offset, SeekOrigin::Begin ), "Failed to seek for reading." );

            // #TODO: Loop read, or use IOJob
            const size_t  sliceSize = _slices[bucket][_slice];
            const ssize_t sizeRead  = _baseStream.Read( buffer, sliceSize );
            PanicIf( sizeRead != (ssize_t)sliceSize, "Failed to read slice %u of bucket %u.", (uint32)_slice, bucket );

            buffer += sliceSize;
        }

        _slice++;
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
    if(origin == SeekOrigin::Begin && offset == 0)
    {
        _bucket = _slice = 0;
        return true;
    }

    return false;
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

