#pragma once
#include "util/Util.h"
#include "util/StackAllocator.h"
#include "pos/chacha8.h"
#include "threading/MTJob.h"
#include "plotdisk/DiskPlotInfo.h"
#include "plotdisk/DiskPlotContext.h"
#include "plotdisk/DiskPlotConfig.h"
#include "plotdisk/DiskBufferQueue.h"

template<uint32 _numBuckets>
class K32BoundedF1
{
    using Job = AnonPrefixSumJob<uint32>;

    static constexpr uint32 _k                = 32;
    static constexpr uint64 _kEntryCount      = 1ull << _k;
    static constexpr uint32 _entriesPerBucket = (uint32)( _kEntryCount / _numBuckets );
    static constexpr uint32 _entriesPerBlock  = kF1BlockSize / sizeof( uint32 );
    static constexpr uint32 _blocksPerBucket  = _entriesPerBucket * sizeof( uint32 ) / kF1BlockSize;

public:
    //-----------------------------------------------------------
    K32BoundedF1( DiskPlotContext& context, IAllocator& allocator )
        : _context( context )
        , _ioQueue( *context.ioQueue )
        , _writeFence( context.fencePool->RequireFence() )
    {
        const uint32 threadCount = context.f1ThreadCount;

        // Get the maximum block count per thread
        uint32 blockCount, _;
        GetThreadOffsets( threadCount-1, threadCount, _blocksPerBucket, blockCount, _, _ );

        const uint32 blockBufferSize = blockCount * threadCount * _entriesPerBlock;
        _blockBuffer = allocator.CAllocSpan<uint32>( blockBufferSize );

        _yEntries[0] = allocator.CAlloc<uint32>( _entriesPerBucket );
        _yEntries[1] = allocator.CAlloc<uint32>( _entriesPerBucket );
        _xEntries[0] = allocator.CAlloc<uint32>( _entriesPerBucket );
        _xEntries[1] = allocator.CAlloc<uint32>( _entriesPerBucket );
    }

    //-----------------------------------------------------------
    void Run()
    {
        Job::Run( *_context.threadPool, _context.f1ThreadCount, [=]( Job* self ) {

            byte key[BB_PLOT_ID_LEN] = { 1 };
            memcpy( key + 1, _context.plotId, BB_PLOT_ID_LEN-1 );

            chacha8_ctx chacha;
            chacha8_keysetup( &chacha, key, 256, nullptr );

            uint32 blockCount, blockOffset, _;
            GetThreadOffsets( self, _blocksPerBucket, blockCount, blockOffset, _ );

            auto blocks = _blockBuffer.Slice( blockOffset * _entriesPerBlock, blockCount * _entriesPerBlock );

            for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
            {
                // Calculate f1 blocks
                chacha8_get_keystream( &chacha, blockOffset, blockCount, (byte*)blocks.Ptr() );

                // Write blocks to disk buckets
                WriteToBuckets( bucket, self, blocks.Ptr(), blockCount, blockOffset * _entriesPerBlock );

                blockOffset += _blocksPerBucket; // Offset to block start at next bucket
            }
        });
    }

private:
    //-----------------------------------------------------------
    void WriteToBuckets( const uint32 bucket, Job* self, const uint32* blocks, const uint32 blockCount, const uint32 xStart )
    {
        const uint32 entriesPerBlock  = kF1BlockSize / sizeof( uint32 );
        const uint32 entryCount       = blockCount * entriesPerBlock;
        const uint32 bucketBits       = bblog2( _numBuckets );
        const uint32 bucketBitShift   = _k - bucketBits;
        const uint32 kMinusKExtraBits = _k - kExtraBits;

        // Distribute to buckets
        uint32 counts     [_numBuckets] = { 0 };
        uint32 pfxSum     [_numBuckets];
        uint32 totalCounts[_numBuckets];

//        memset( counts, 0, sizeof( counts ) );

        for( uint32 i = 0; i < entryCount; i++ )
            counts[Swap32( blocks[i] ) >> bucketBitShift]++;

        self->CalculatePrefixSum( _numBuckets, counts, pfxSum, totalCounts );

        const uint32 yBits = _k + kExtraBits - bucketBits;
        const uint32 yMask = (uint32)(( 1ull << yBits ) - 1);

        Span<uint32> yEntries, xEntries, elementCounts;
        GetNextBuffer( self, bucket, yEntries, xEntries, elementCounts );

        // Distribute slices to buckets
        for( uint32 i = 0; i < entryCount; i++ )
        {
                  uint32 y   = Swap32( blocks[i] );
            const uint32 dst = --pfxSum[y >> bucketBitShift];
            const uint32 x   = xStart + i;

            y = ( ( y << kExtraBits ) | ( x >> kMinusKExtraBits ) ) & yMask;

            yEntries[dst] = y;
            xEntries[dst] = x;
        }

        // Write to disk
        if( self->BeginLockBlock() )
        {
            memcpy( elementCounts.Ptr(), totalCounts, sizeof( totalCounts ) );

            _ioQueue.WriteBucketElementsT( FileId::FX1  , yEntries.Ptr(), elementCounts.Ptr() );
            _ioQueue.WriteBucketElementsT( FileId::META1, xEntries.Ptr(), elementCounts.Ptr() );
            _ioQueue.SignalFence( _writeFence, bucket+1 );
            _ioQueue.CommitCommands();

            for( uint32 i = 0; i < _numBuckets; i++ )
                _context.bucketCounts[(int)TableId::Table1][i] += totalCounts[i];
        }
        self->EndLockBlock();
    }

    //-----------------------------------------------------------
    void GetNextBuffer( Job* self, const uint32 bucket, Span<uint32>& yEntries, Span<uint32>& xEntries, Span<uint32>& totalCounts )
    {
        if( bucket >= 2 && _writeFence.Value() < bucket-1 )
        {
            if( self->BeginLockBlock() )
                _writeFence.Wait( bucket - 1, _context.p1TableWaitTime[(int)TableId::Table1] );

            self->EndLockBlock();
        }

        const uint32 entriesPerBucket = (uint32)( _kEntryCount / _numBuckets );
        yEntries    = Span<uint32>( _yEntries[bucket & 1], entriesPerBucket );
        xEntries    = Span<uint32>( _xEntries[bucket & 1], entriesPerBucket );
        totalCounts = Span<uint32>( _elementCounts[bucket & 1], _numBuckets );
    }

private:
    DiskPlotContext&    _context;
    DiskBufferQueue&    _ioQueue;
    Fence&              _writeFence;
    Span<uint32>        _blockBuffer;

    // I/O buffers
    uint32* _yEntries     [2];
    uint32* _xEntries     [2];
    uint32  _elementCounts[2][_numBuckets] = {};
};