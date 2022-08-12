#pragma once
#include "threading/Fence.h"
#include "util/StackAllocator.h"
#include "plotdisk/BitBucketWriter.h"

template<uint32 _numBuckets, bool _overflow>
class MapWriter
{
public:
    static constexpr uint32 _k             = _K;
    static constexpr uint32 BucketBits     = bblog2( _numBuckets );
    static constexpr uint32 ExtraBucket    = _overflow ? 1 : 0;
    static constexpr uint32 AddressBitSize = _k + ExtraBucket;
    static constexpr uint32 EntryBitSize   = _k - BucketBits + AddressBitSize;  // ( origin address | final address ) ( k-log2(buckets) | 32 )

    using MapBitBucketWriter = BitBucketWriter<_numBuckets+ExtraBucket>;
    using Job                = AnonPrefixSumJob<uint32>;

public:

    //-----------------------------------------------------------
    MapWriter() {}

    //-----------------------------------------------------------
    MapWriter( DiskBufferQueue& ioQueue, const FileId fileId, 
               IAllocator& allocator, const uint64 maxEntries, const size_t blockSize,
               Fence& writeFence, Duration& writeWaitTime )
        : _ioQueue      ( &ioQueue )
        , _bucketWriter ( ioQueue, fileId, AllocBucketWriterBuffer( allocator, blockSize ) )
        , _writeFence   ( &writeFence )
        , _writeWaitTime( &writeWaitTime )
    {
        AllocateWriteBuffers( maxEntries, allocator, blockSize );
    }

    // #NOTE: Use only to determine allocation size
    //-----------------------------------------------------------
    MapWriter( const uint64 maxEntries, IAllocator& allocator, const size_t blockSize )
    {
        AllocBucketWriterBuffer( allocator, blockSize );
        AllocateWriteBuffers( maxEntries, allocator, blockSize );
    }

    //-----------------------------------------------------------
    void Write( ThreadPool& pool, const uint32 threadCount,
                const uint32 bucket, const int64 entryCount, const uint64 mapOffset, 
                const uint64* map, uint64* outMap, uint64 outMapBucketCounts[_numBuckets+ExtraBucket] )
    {
        // Write the key as a map to the final entry location.
        // We do this by writing into buckets to final sorted (current) location
        // into its original bucket, with the offset in that bucket.
        using Job = AnonPrefixSumJob<uint32>;

        uint32 _totalCounts   [_numBuckets+ExtraBucket]; uint32* totalCounts    = _totalCounts;
        uint64 _totalBitCounts[_numBuckets+ExtraBucket]; uint64* totalBitCounts = _totalBitCounts;

        Job::Run( pool, threadCount, [=]( Job* self ) {
            
            const uint32 bucketBits   = BucketBits;
            const uint32 bucketShift  = _k - bucketBits;
            const uint32 bitSize      = EntryBitSize;
            const uint32 encodeShift  = AddressBitSize;
            const uint32 numBuckets   = _numBuckets + ExtraBucket;


            int64 count, offset, end;
            GetThreadOffsets( self, entryCount, count, offset, end );

            uint32 counts[numBuckets] = { 0 };
            uint32 pfxSum[numBuckets];

            const uint64* inIdx    = map + offset;
            const uint64* inIdxEnd = map + end;

            // Count buckets
            do {
                const uint64 b = *inIdx >> bucketShift;
                ASSERT( b < numBuckets );
                counts[b]++;
            } while( ++inIdx < inIdxEnd );

            self->CalculatePrefixSum( numBuckets, counts, pfxSum, totalCounts );

            // Convert map entries from source index to reverse map
            const uint64 tableOffset = mapOffset + (uint64)offset;

            const uint64* reverseMap    = map + offset;
                  uint64* outMapBuckets = outMap; 

            for( int64 i = 0; i < count; i++ )
            {
                const uint64 origin = reverseMap[i];            ASSERT( origin < 0x100000000 + (1ull<<_k) / _numBuckets );
                const uint64 b      = origin >> bucketShift;    ASSERT( b <= _numBuckets );
                const uint32 dstIdx = --pfxSum[b];              ASSERT( (int64)dstIdx < entryCount );

                const uint64 finalIdx = tableOffset + (uint64)i;
                ASSERT( finalIdx < ( 1ull << encodeShift ) );

                outMapBuckets[dstIdx] = (origin << encodeShift) | finalIdx;
            }

            auto&   bitWriter = _bucketWriter;
            uint64* bitCounts = totalBitCounts;

            if( self->BeginLockBlock() )
            {
                // Convert counts to bit sizes
                for( uint32 i = 0; i < numBuckets; i++ )
                    bitCounts[i] = (uint64)totalCounts[i] * bitSize;

                byte* writeBuffer = GetWriteBuffer( bucket );

                // Wait for the buffer to be available first
                if( bucket > 1 )
                    _writeFence->Wait( bucket - 2, *_writeWaitTime );

                bitWriter.BeginWriteBuckets( bitCounts, writeBuffer );
            }
            self->EndLockBlock();


            // Bit-compress/pack each bucket entries (except the overflow bucket)
            uint64 bitsWritten = 0;

            for( uint32 i = 0; i < _numBuckets; i++ )
            {
                if( counts[i] < 1 )
                {
                    self->SyncThreads();
                    continue;
                }

                const uint64 writeOffset = pfxSum[i];
                const uint64 bitOffset   = writeOffset * bitSize - bitsWritten;
                bitsWritten += bitCounts[i];

                ASSERT( bitOffset + counts[i] * bitSize <= bitCounts[i] );

                BitWriter writer = bitWriter.GetWriter( i, bitOffset );

                const uint64* mapToWrite    = outMapBuckets + writeOffset;
                const uint64* mapToWriteEnd = mapToWrite + counts[i]; 

                // Compress a couple of entries first, so that we don't get any simultaneaous writes to the same fields
                const uint64* mapToWriteEndPass1 = mapToWrite + std::min( counts[i], 2u ); 
                ASSERT( counts[i] > 2 );

                while( mapToWrite < mapToWriteEndPass1 )
                    writer.Write( *mapToWrite++, bitSize );

                self->SyncThreads();

                while( mapToWrite < mapToWriteEnd )
                    writer.Write( *mapToWrite++, bitSize );
            }

            // Write the overflow bucket and then write to disk
            if( self->BeginLockBlock() )
            {
                if constexpr ( _overflow )
                {
                    const uint32 overflowBucket = _numBuckets;
                    const uint64 overflowCount  = totalCounts[overflowBucket];

                    if( overflowCount )
                    {
                        const uint64 writeOffset = pfxSum[overflowBucket];
                        ASSERT( writeOffset * bitSize - bitsWritten == 0 );
                        
                        BitWriter writer = bitWriter.GetWriter( overflowBucket, 0 );

                        const uint64* mapToWrite  = outMapBuckets + writeOffset;
                        const uint64* mapWriteEnd = mapToWrite + overflowCount;
                        while( mapToWrite < mapWriteEnd )
                            writer.Write( *mapToWrite++, bitSize );
                    }
                }

                bitWriter.Submit();
                _ioQueue->SignalFence( *_writeFence, bucket );
                _ioQueue->CommitCommands();
            }
            self->EndLockBlock();
        });

        for( int32 b = 0; b <= (int32)_numBuckets; b++ )
            outMapBucketCounts[b] += totalCounts[b];
    }

    //-----------------------------------------------------------
    template<typename TMapIn>
    void WriteJob( Job* self, const uint32 bucket, const uint64 mapOffset,
        const Span<TMapIn> indices,
              Span<uint64> mapOut, 
              uint64       outMapBucketCounts  [_numBuckets+ExtraBucket],
              uint32       totalCounts         [_numBuckets+ExtraBucket],
              uint64       sharedTotalBitCounts[_numBuckets+ExtraBucket] )  // Must be 1 instance shared across all jobs
    {
        const uint32 bucketBits  = BucketBits;
        const uint32 bucketShift = _k - bucketBits;
        const uint32 bitSize     = EntryBitSize;
        const uint32 encodeShift = AddressBitSize;
        const uint32 numBuckets  = _numBuckets + ExtraBucket;

        uint32 counts[numBuckets] = { 0 };
        uint32 pfxSum[numBuckets];

        uint32 count, offset, _;
        GetThreadOffsets( self, (uint32)indices.Length(), count, offset, _ );

        auto inIdx = indices.Slice( offset, count );

        // Count buckets
        for( size_t i = 0; i < inIdx.Length(); i++ )
        {
            const TMapIn b = inIdx[i] >> bucketShift;     ASSERT( b < numBuckets );
            counts[b]++;
        }

        self->CalculatePrefixSum( numBuckets, counts, pfxSum, totalCounts );

        const uint64 tableOffset = mapOffset + (uint64)offset;

        // Convert entries from source index to reverse map
        for( size_t i = 0; i < inIdx.Length(); i++ )
        {
            const uint64 origin = inIdx[i];               ASSERT( origin < 0x100000000 + (1ull<<_k) / _numBuckets );
            const uint64 b      = origin >> bucketShift;  ASSERT( b < numBuckets );
            const uint32 dstIdx = --pfxSum[b];            ASSERT( dstIdx < indices.Length() );
            const uint64 finalIdx = tableOffset + (uint64)i;
            
            mapOut[dstIdx] = (origin << encodeShift) | finalIdx;

            ASSERT( finalIdx < ( 1ull << encodeShift ) );
        }

        // Write 
        auto&   bitWriter = _bucketWriter;
        uint64* bitCounts = sharedTotalBitCounts;

        if( self->BeginLockBlock() )
        {
            // Convert counts to bit sizes
            for( uint32 i = 0; i < numBuckets; i++ )
                bitCounts[i] = (uint64)totalCounts[i] * bitSize;

            byte* writeBuffer = GetWriteBuffer( bucket );

            // Wait for the buffer to be available first
            if( bucket > 1 )
                _writeFence->Wait( bucket - 2, *_writeWaitTime );

            bitWriter.BeginWriteBuckets( bitCounts, writeBuffer );
        }
        self->EndLockBlock();


        // Bit-compress/pack each bucket entries (except the overflow bucket)
        uint64 bitsWritten = 0;

        for( uint32 i = 0; i < _numBuckets; i++ )
        {
            if( counts[i] < 1 )
            {
                self->SyncThreads();
                continue;
            }

            const uint64 writeOffset = pfxSum[i];
            const uint64 bitOffset   = writeOffset * bitSize - bitsWritten;
            bitsWritten += bitCounts[i];

            ASSERT( bitOffset + counts[i] * bitSize <= bitCounts[i] );

            BitWriter writer = bitWriter.GetWriter( i, bitOffset );

            const uint64* mapToWrite    = mapOut.Ptr() + writeOffset;
            const uint64* mapToWriteEnd = mapToWrite + counts[i]; 

            // Compress a couple of entries first, so that we don't get any simultaneaous writes to the same fields
            const uint64* mapToWriteEndPass1 = mapToWrite + std::min( counts[i], 2u ); 
            ASSERT( counts[i] > 2 );

            while( mapToWrite < mapToWriteEndPass1 )
                writer.Write( *mapToWrite++, bitSize );

            self->SyncThreads();

            while( mapToWrite < mapToWriteEnd )
                writer.Write( *mapToWrite++, bitSize );
        }

        // Write the overflow bucket and then write to disk
        if( self->BeginLockBlock() )
        {
            if constexpr( _overflow )
            {
                Fatal( "Unimplemented" );   // #TODO: Fix compiler errors
                const uint32 overflowBucket = _numBuckets;
                const uint64 overflowCount  = totalCounts[overflowBucket];

                if( overflowCount )
                {
                    const uint64 writeOffset = pfxSum[overflowBucket];
                    ASSERT( writeOffset * bitSize - bitsWritten == 0 );
                    
                    BitWriter writer = bitWriter.GetWriter( overflowBucket, 0 );

                    const uint64* mapToWrite  = mapOut.Ptr() + writeOffset;
                    const uint64* mapWriteEnd = mapToWrite + overflowCount;
                    while( mapToWrite < mapWriteEnd )
                        writer.Write( *mapToWrite++, bitSize );
                }
            }

            bitWriter.Submit();
            _ioQueue->SignalFence( *_writeFence, bucket );
            _ioQueue->CommitCommands();
        }
        self->EndLockBlock();
    }

    //-----------------------------------------------------------
    void SubmitFinalBits()
    {
        _bucketWriter.SubmitLeftOvers();
        _ioQueue->SignalFence( *_writeFence, _numBuckets );
        _ioQueue->CommitCommands();
    }

private:
    //-----------------------------------------------------------
    inline void AllocateWriteBuffers( const uint64 maxEntries, IAllocator& allocator, const size_t blockSize  )
    {
        const size_t writeBufferSize = RoundUpToNextBoundary( CDiv( maxEntries * EntryBitSize, 8 ), (int)blockSize );

        _writebuffers[0] = allocator.AllocT<byte>( writeBufferSize, blockSize );
        _writebuffers[1] = allocator.AllocT<byte>( writeBufferSize, blockSize );
    }

    //-----------------------------------------------------------
    inline byte* AllocBucketWriterBuffer( IAllocator& allocator, const size_t blockSize )
    {
        return (byte*)allocator.CAlloc( _numBuckets+ExtraBucket, blockSize, blockSize );
    }

    //-----------------------------------------------------------
    inline byte* GetWriteBuffer( const uint32 bucket )
    {
        return _writebuffers[bucket & 1];
    }

private:
    DiskBufferQueue*   _ioQueue         = nullptr;
    MapBitBucketWriter _bucketWriter;
    byte*              _writebuffers[2] = { nullptr };
    Fence*             _writeFence      = nullptr;
    Duration*          _writeWaitTime   = nullptr;
};

