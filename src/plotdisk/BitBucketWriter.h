#pragma once
#include "util/Util.h"
#include "util/BitView.h"


template<uint32 _numBuckets>
class BitBucketWriter
{
    DiskBufferQueue& _queue;
    uint64*          _remainderFields  [_numBuckets];   // Left-over block buffers
    uint64           _remainderBitCount[_numBuckets] = { 0 };
    BitWriter        _writers          [_numBuckets];
    FileId           _fileId;

public:

    //-----------------------------------------------------------
    inline BitBucketWriter( DiskBufferQueue& queue, const FileId fileId, byte* blockBuffers )
        : _queue       ( queue  )
        , _fileId      ( fileId )
    {
        const size_t fsBlockSize = _queue.BlockSize( _fileId );
        ASSERT( fsBlockSize > 0 );

        for( uint32 i = 0; i < _numBuckets; i++ )
        {
            _remainderFields[i]  = (uint64*)blockBuffers;
            *_remainderFields[i] = 0;
            blockBuffers += fsBlockSize;
        }
    }

    //-----------------------------------------------------------
    inline void BeginWriteBuckets( const uint64 bucketBitSizes[_numBuckets] )
    {
        const size_t fsBlockSize     = _queue.BlockSize( _fileId );
        const size_t fsBlockSizeBits = fsBlockSize * 8;
              size_t allocSize       = 0;

        ASSERT( fsBlockSize > 0 );

        for( uint32 i = 0; i < _numBuckets; i++ )
            allocSize += CDiv( bucketBitSizes[i] + _remainderBitCount[i], fsBlockSizeBits ) * fsBlockSizeBits / 8;

        byte* bucketBuffers = _queue.GetBuffer( allocSize, fsBlockSize, true );

        // Initialize our BitWriters
        uint64* fields = (uint64*)bucketBuffers;
        for( uint32 i = 0; i < _numBuckets; i++ )
        {
            const uint64 leftOverBitCount = _remainderBitCount[i];
            const size_t totalBitCount    = bucketBitSizes[i] + leftOverBitCount;
            const size_t bufferBitSize    = CDiv( totalBitCount, fsBlockSizeBits ) * fsBlockSizeBits;
            const size_t bufferFieldCount = bufferBitSize / 64;
            ASSERT( bufferBitSize / 64 * 64 == bufferBitSize );

            _writers[i] = BitWriter( fields, bufferBitSize, leftOverBitCount );
            
            if( leftOverBitCount > 0 )
            {
                const uint64 nFields = CDiv( leftOverBitCount, 64 );
                memcpy( fields, _remainderFields[i], nFields * sizeof( 64 ) );
            }

            // Update with the remainder bits based on the size of the bucket in bits
            const uint64 newLeftOverBits = totalBitCount - totalBitCount / fsBlockSizeBits * fsBlockSizeBits;
            _remainderBitCount[i] = newLeftOverBits;

            fields += bufferFieldCount;
        }
    }

    //-----------------------------------------------------------
    inline BitWriter GetWriter( const uint32 bucket, const uint64 bitOffset )
    {
        ASSERT( bucket < _numBuckets );

        BitWriter writer( _writers[bucket] );
        writer.Bump( bitOffset );

        return writer;
    }

    //-----------------------------------------------------------
    inline void Submit()
    {
        const size_t fsBlockSize     = _queue.BlockSize( _fileId );
        const size_t fsBlockSizeBits = fsBlockSize * 8;

        // Save any overflow bits (we already recorded the overflow bit count in BeginWriteBuckets)
        for( uint32 i = 0; i < _numBuckets; i++ )
        {
            BitWriter& writer = _writers[i];

            // const size_t writableBits 
            const size_t remainderBits = _remainderBitCount[i];
            const size_t bitsToWrite   = writer.Capacity() - fsBlockSizeBits;
            const size_t bytesToWrite  = bitsToWrite / 8;

            ASSERT( bitsToWrite / fsBlockSizeBits * fsBlockSizeBits == bitsToWrite );

            if( remainderBits )
            {
                // Copy fields needed
                const size_t remainderFieldCount = CDiv( remainderBits, 64 );
                memcpy( _remainderFields[i], ((byte*)writer.Fields()) + bytesToWrite, remainderFieldCount * sizeof( uint64 ) );
            }

            if( bytesToWrite )
                _queue.WriteFile( _fileId, i, writer.Fields(), bytesToWrite );
        }

        _queue.ReleaseBuffer( _writers[0].Fields() );
        _queue.CommitCommands();
    }

    //-----------------------------------------------------------
    inline void SubmitLeftOvers()
    {
        const size_t fsBlockSize = _queue.BlockSize( _fileId );
        ASSERT( fsBlockSize );

        for( uint32 i = 0; i < _numBuckets; i++ )
        {
            if( _remainderBitCount[i] > 0 )
                _queue.WriteFile( _fileId, i, _remainderFields[i], fsBlockSize );
        }
        _queue.CommitCommands();
    }
};
