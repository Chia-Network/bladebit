#pragma once
#include "util/Util.h"
#include "util/BitView.h"


template<uint32 _numBuckets>
class BitBucketWriter
{
    struct BitBucket
    {
        size_t count;
        byte*  buffer;
    };

    DiskBufferQueue* _queue                          = nullptr;
    uint64*          _remainderFields  [_numBuckets] = { nullptr };   // Left-over block buffers
    uint64           _remainderBitCount[_numBuckets] = { 0 };
    BitBucket        _buckets          [_numBuckets] = { 0 };
    FileId           _fileId                         = FileId::None;

public:

    //-----------------------------------------------------------   
    inline BitBucketWriter()
    {}

    //-----------------------------------------------------------
    inline BitBucketWriter( DiskBufferQueue& queue, const FileId fileId, byte* blockBuffers )
        : _queue       ( &queue )
        , _fileId      ( fileId )
    {
        SetBlockBuffers( blockBuffers );
    }

    //-----------------------------------------------------------
    inline BitBucketWriter( const BitBucketWriter<_numBuckets>& other )
    {
        memcpy( this, &other, sizeof( other ) );
    }

    //-----------------------------------------------------------
    // inline BitBucketWriter( BitBucketWriter<_numBuckets>&& other )
    // {
    //     memcpy( this, &other, sizeof( other ) );
    //     memset( *other, 0, sizeof( other ));
    // }

    //-----------------------------------------------------------
    inline void SetFileId( const FileId id ) { _fileId = id; }

    //-----------------------------------------------------------
    inline void BeginWriteBuckets( const uint64 bucketBitSizes[_numBuckets] )
    {
        const size_t fsBlockSize     = _queue->BlockSize( _fileId );
        const size_t fsBlockSizeBits = fsBlockSize * 8;
        ASSERT( fsBlockSize > 0 );

        size_t allocSize = 0;

        for( uint32 i = 0; i < _numBuckets; i++ )
            allocSize += CDivT( (size_t)bucketBitSizes[i] + (size_t)_remainderBitCount[i], fsBlockSizeBits ) * fsBlockSizeBits / 8;

        byte* bucketBuffers = _queue->GetBuffer( allocSize, fsBlockSize, true );

        BeginWriteBuckets( bucketBitSizes, bucketBuffers );
    }

    //-----------------------------------------------------------
    inline void BeginWriteBuckets( const uint64 bucketBitSizes[_numBuckets], byte* bucketBuffers )
    {
        ASSERT( bucketBuffers );
        
        const size_t fsBlockSizeBits = _queue->BlockSize( _fileId ) * 8;
        ASSERT( fsBlockSizeBits > 0 );

        // Initialize our BitWriters
        byte* fields = bucketBuffers;
        for( uint32 i = 0; i < _numBuckets; i++ )
        {
            const uint64 leftOverBitCount  = _remainderBitCount[i];
            const size_t totalBitCount     = bucketBitSizes[i] + leftOverBitCount;
            const size_t bufferBitCapacity = CDivT( totalBitCount, fsBlockSizeBits ) * fsBlockSizeBits;
            ASSERT( bufferBitCapacity / 64 * 64 == bufferBitCapacity );
            
            _buckets[i].count  = totalBitCount;
            _buckets[i].buffer = fields;
            
            if( leftOverBitCount > 0 )
            {
                const uint64 nFields = CDiv( leftOverBitCount, 64 );
                memcpy( fields, _remainderFields[i], nFields * sizeof( uint64 ) );
            }

            fields += bufferBitCapacity / 8;
        }
    }

    //-----------------------------------------------------------
    inline void Submit()
    {
        const size_t fsBlockSize     = _queue->BlockSize( _fileId );
        const size_t fsBlockSizeBits = fsBlockSize * 8;

        // Save any overflow bits
        for( uint32 i = 0; i < _numBuckets; i++ )
        {
            BitBucket& bucket = _buckets[i];
            
            const size_t bitCount      = bucket.count;
            const size_t bitsToWrite   = bitCount / fsBlockSizeBits * fsBlockSizeBits;
            const size_t remainderBits = bitCount - bitsToWrite;
            
            const size_t bytesToWrite  = bitsToWrite / 8;

            ASSERT( bitsToWrite / fsBlockSizeBits * fsBlockSizeBits == bitsToWrite );

            if( remainderBits )
            {
                // Copy left-over fields
                const size_t remainderFieldCount = CDiv( remainderBits, 64 );
                memcpy( _remainderFields[i], bucket.buffer + bytesToWrite, remainderFieldCount * sizeof( uint64 ) );
            }

            if( bytesToWrite )
                _queue->WriteFile( _fileId, i, bucket.buffer, bytesToWrite );

            _remainderBitCount[i] = remainderBits;
        }
        _queue->CommitCommands();
    }

    //-----------------------------------------------------------
    inline void SubmitAndRelease()
    {
        this->Submit();
        _queue->ReleaseBuffer( _buckets[0].buffer );
        _queue->CommitCommands();
    }

    //-----------------------------------------------------------
    inline void SubmitLeftOvers()
    {
        const size_t fsBlockSize = _queue->BlockSize( _fileId );
        ASSERT( fsBlockSize );

        for( uint32 i = 0; i < _numBuckets; i++ )
        {
            if( _remainderBitCount[i] > 0 )
                _queue->WriteFile( _fileId, i, _remainderFields[i], fsBlockSize );
        }
        _queue->CommitCommands();
    }

    //-----------------------------------------------------------
    inline BitWriter GetWriter( const uint32 bucket, const uint64 bitOffset )
    {
        ASSERT( bucket < _numBuckets );

        BitBucket& b = _buckets[bucket];
        return BitWriter( (uint64*)b.buffer, b.count, _remainderBitCount[bucket] + bitOffset );
    }

    //-----------------------------------------------------------
    inline size_t RemainderBits( const uint32 bucket )
    {
        ASSERT( bucket < _numBuckets );
        return _remainderBitCount[bucket];
    }

private:

    //-----------------------------------------------------------
    inline void SetBlockBuffers( byte* blockBuffers )
    {
        const size_t fsBlockSize = _queue->BlockSize( _fileId );
        ASSERT( fsBlockSize > 0 );

        for( uint32 i = 0; i < _numBuckets; i++ )
        {
            _remainderFields[i]  = (uint64*)blockBuffers;
            *_remainderFields[i] = 0;
            blockBuffers += fsBlockSize;
        }
    }
};
