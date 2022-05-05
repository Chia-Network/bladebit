#include "BucketStream.h"
#include "util/Util.h"

 //-----------------------------------------------------------
 BucketStream::BucketStream( IStream& baseStream, const size_t sliceSize, const uint32 numBuckets )
    : _baseStream       ( baseStream )
    , _sequentialSlices ( bbcalloc<size_t*>( numBuckets ), numBuckets )
    , _interleavedSlices( bbcalloc<size_t*>( numBuckets ), numBuckets )
    , _sliceCapacity    ( sliceSize )
    , _bucketCapacity   ( sliceSize * numBuckets )
    , _numBuckets       ( numBuckets )
{
    const size_t sliceCount = numBuckets * (size_t)numBuckets;

    size_t* seqSliceBuffer = bbcalloc<size_t>( sliceCount );
    size_t* intSliceBuffer = bbcalloc<size_t>( sliceCount );
    ZeroMem( seqSliceBuffer, sliceCount );
    ZeroMem( intSliceBuffer, sliceCount );

    for( uint32 i = 0; i < numBuckets; i++ )
    {
        _sequentialSlices [i] = seqSliceBuffer;
        _interleavedSlices[i] = intSliceBuffer;

        seqSliceBuffer += numBuckets;
        intSliceBuffer += numBuckets;
    }
 }

 //-----------------------------------------------------------
 BucketStream::~BucketStream()
 {
     free( _sequentialSlices [0] );
     free( _interleavedSlices[0] );
     free( _sequentialSlices .Ptr() );
     free( _interleavedSlices.Ptr() );
 }

 //-----------------------------------------------------------
 void BucketStream::WriteBucketSlices( const void* slices, const uint32* sliceSizes )
 {
    const byte* sliceBytes = (byte*)slices;

    IStream& stream = _baseStream;

    if( GetWriteMode() == Sequential )
    {
        // Write slices across all buckets. That is, each bucket contains its own slices
        for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
        {
            const int64 offset = (int64)( bucket * _bucketCapacity + _writeSlice * _sliceCapacity ); //bucket * (int64)_bucketCapacity + (int64)_slices[i][0].size;
            PanicIf( !stream.Seek( offset, SeekOrigin::Begin ), "Base stream failed to seek." );

            const size_t size = sliceSizes[bucket];
            ASSERT( size <= _sliceCapacity );

            // #TODO: Loop write, or use IOJob
            PanicIf( stream.Write( sliceBytes, size ) != (ssize_t)size, "Failed to write slice to base stream." );

            _sequentialSlices[bucket][_writeSlice] = size;
            sliceBytes += size;
        }

        _writeSlice++;
    }
    else
    {
        // Interleaved writes. This means that we will fill each bucket
        // with slices that belong to all buckets, not just he bukcket's own slices.
        // ie. Bucket 0 will have slices from bucket, 0, 1, 2... and so on.

        // #NOTE: We can't just write the whole bucket-full because we need to have each slice
        //        separated by its full slice capacity so that we don't overwrite any data
        //        during the next sequential write.
        const size_t bucketOffset = _writeBucket * _bucketCapacity;

        for( uint32 slice = 0; slice < _numBuckets; slice++ )
        {
            const int64 offset = (int64)( bucketOffset + slice * _sliceCapacity );
            PanicIf( !stream.Seek( offset, SeekOrigin::Begin ), "Base stream failed to seek." );

            const size_t size = sliceSizes[slice];
            ASSERT( size <= _sliceCapacity );

            // #TODO: Loop write, or use IOJob
            PanicIf( stream.Write( sliceBytes, size ) != (ssize_t)size, "Failed to write slice to base stream." );
            _interleavedSlices[_writeBucket][slice] = size;
            sliceBytes += size;
        }

        _writeBucket++;
     }
 }

 //-----------------------------------------------------------
 void BucketStream::ReadBucket( const size_t size, void* readBuffer )
 {
    ASSERT( _readBucket < _numBuckets );
    ASSERT( size );
    ASSERT( readBuffer );
    // #TODO: Remove the read size... We don't need that here...

    byte* buffer = (byte*)readBuffer;
    IStream& stream = _baseStream;

    if( GetReadMode() == Sequential )
    {
        // Read all slices from the same bucket
        const size_t _bucketStart = _readBucket * _bucketCapacity;

        for( uint32 slice = 0; slice < _numBuckets; slice++ )
        {
            const int64 offset = (int64)(_bucketStart + slice * _sliceCapacity );
            PanicIf( !stream.Seek( offset, SeekOrigin::Begin ), "Failed to seek for reading." );

            // #TODO: Loop read, or use IOJob
            const size_t  sliceSize = _sequentialSlices[_readBucket][slice];
            const ssize_t sizeRead  = stream.Read( buffer, sliceSize );
            PanicIf( sizeRead != (ssize_t)sliceSize, "Failed to read slice %u of bucket %u.", slice, (uint32)_readBucket );

            buffer += sliceSize;
        }

        _readBucket++;
     }
    else
    {
        // Read a whole bucket's worth of bytes by reading slices spread across all buckets
        for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
        {
            const int64 offset = (int64)( bucket * _bucketCapacity + _readSlice * _sliceCapacity );
            PanicIf( !_baseStream.Seek( offset, SeekOrigin::Begin ), "Failed to seek for reading." );

            // #TODO: Loop read, or use IOJob
            const size_t  sliceSize = _interleavedSlices[bucket][_readSlice];
            const ssize_t sizeRead  = _baseStream.Read( buffer, sliceSize );
            PanicIf( sizeRead != (ssize_t)sliceSize, "Failed to read slice %u of bucket %u.", (uint32)_readSlice, bucket );

            buffer += sliceSize;
        }

        _readSlice++;
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
    if( origin == SeekOrigin::Begin && offset == 0 )
    {
        _readBucket  = 0;
        _writeBucket = 0;

        SwitchMode();
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

