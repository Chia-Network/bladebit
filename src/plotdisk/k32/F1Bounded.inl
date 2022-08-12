#pragma once
#include "util/Util.h"
#include "util/StackAllocator.h"
#include "pos/chacha8.h"
#include "threading/MTJob.h"
#include "plotdisk/DiskPlotInfo.h"
#include "plotdisk/DiskPlotContext.h"
#include "plotdisk/DiskPlotConfig.h"
#include "plotdisk/DiskBufferQueue.h"

#if _DEBUG
    #include "plotdisk/DiskPlotDebug.h"
    #include "algorithm/RadixSort.h"
    void DbgValidateF1( DiskPlotContext& context );
#endif


template<uint32 _numBuckets>
class K32BoundedF1
{
    using Job = AnonPrefixSumJob<uint32>;

    static constexpr uint32 _k                      = 32;
    static constexpr uint64 _kEntryCount            = 1ull << _k;
    static constexpr uint32 _entriesPerBucket       = (uint32)( _kEntryCount / _numBuckets );
    static constexpr uint32 _entriesPerBlock        = kF1BlockSize / sizeof( uint32 );
    static constexpr uint32 _blocksPerBucket        = _entriesPerBucket * sizeof( uint32 ) / kF1BlockSize;
    static constexpr uint32 _maxEntriesPerSlice     = (uint32)((_entriesPerBucket / _numBuckets) * BB_DP_ENTRY_SLICE_MULTIPLIER);
    
public:
    //-----------------------------------------------------------
    K32BoundedF1( DiskPlotContext& context, IAllocator& allocator )
        : _context   ( context )
        , _ioQueue   ( *context.ioQueue )
        , _writeFence( context.fencePool->RequireFence() )
    {
        static_assert( (uint64)_blocksPerBucket * _entriesPerBlock * _numBuckets == _kEntryCount );
        
        const uint32 threadCount = context.f1ThreadCount;

        // We need to pad our slices to block size
        const uint32 blockSize               = (uint32)_ioQueue.BlockSize( FileId::FX0 );
        const uint32 entriesPerSliceAligned  = RoundUpToNextBoundaryT( _maxEntriesPerSlice, blockSize ) + blockSize / sizeof( uint32 ); // Need an extra block for when we offset the entries
        const uint32 entriesPerBucketAligned = entriesPerSliceAligned * _numBuckets;                                                    // in subsequent slices
        ASSERT( entriesPerBucketAligned >= _entriesPerBucket );

        #if _DEBUG
            _maxEntriesPerIOBucket = entriesPerBucketAligned;
        #endif


        // Get the maximum block count per thread (use the last thread id to get the maximum)
        uint32 blockCount, _;
        GetThreadOffsets( threadCount-1, threadCount, _blocksPerBucket, blockCount, _, _ );

        const uint32 blockBufferSize = blockCount * threadCount * _entriesPerBlock;
        ASSERT( blockCount * threadCount >= _blocksPerBucket );

        _blockBuffer = allocator.CAllocSpan<uint32>( blockBufferSize );

        _yEntries[0] = allocator.CAllocSpan<uint32>( entriesPerBucketAligned, context.tmp2BlockSize );
        _yEntries[1] = allocator.CAllocSpan<uint32>( entriesPerBucketAligned, context.tmp2BlockSize );
        _xEntries[0] = allocator.CAllocSpan<uint32>( entriesPerBucketAligned, context.tmp2BlockSize );
        _xEntries[1] = allocator.CAllocSpan<uint32>( entriesPerBucketAligned, context.tmp2BlockSize );

        _offsets = allocator.CAllocSpan<Span<uint32>>( threadCount );
        for( uint32 i = 0; i < _offsets.Length(); i++ )
        {
            _offsets[i] = allocator.CAllocSpan<uint32>( _numBuckets );
            _offsets[i].ZeroOutElements();
        }
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

        Fence& fence = _context.fencePool->RequireFence();
        _ioQueue.SignalFence( fence, 1 );
        _ioQueue.CommitCommands();
        fence.Wait( 1, _context.p1TableWaitTime[(int)TableId::Table1] );

        _context.entryCounts[(int)TableId::Table1] = 1ull << _k;

        #if _DEBUG
        {
            uint64 tableEntryCount = 0;
            for( uint32 i = 0; i < _numBuckets; i++ )
                tableEntryCount += _context.bucketCounts[(int)TableId::Table1][i];

            ASSERT( tableEntryCount == _context.entryCounts[(int)TableId::Table1] );
        }
        #endif

        _context.fencePool->RestoreAllFences();

        #if ( _DEBUG && BB_DP_DBG_VALIDATE_F1 )
            DbgValidateF1( _context );
        #endif
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
        const uint32 fsBlockSize      = (uint32)_ioQueue.BlockSize( FileId::FX0 );

        // Distribute to buckets
        uint32 counts[_numBuckets] = {};
        uint32 pfxSum[_numBuckets];
        
        // Count bucket entries
        for( uint32 i = 0; i < entryCount; i++ )
            counts[Swap32( blocks[i] ) >> bucketBitShift]++;
        
        // Prefix sum
        Span<uint32> offsets = _offsets[self->JobId()];
        
        // Grab next buffer
        Span<uint32> yEntries, xEntries, elementCounts, alignedElementCounts;
        GetNextBuffer( self, bucket, yEntries, xEntries, elementCounts, alignedElementCounts );

        // Calculate prefix sum
        self->CalculateBlockAlignedPrefixSum<uint32>( _numBuckets, fsBlockSize, counts, pfxSum, elementCounts.Ptr(), offsets.Ptr(), alignedElementCounts.Ptr() );

        // Distribute slices to buckets
        const uint32 yBits = _k + kExtraBits - bucketBits;
        const uint32 yMask = (uint32)(( 1ull << yBits ) - 1);

        for( uint32 i = 0; i < entryCount; i++ )
        {
                  uint32 y   = Swap32( blocks[i] );
            const uint32 dst = --pfxSum[y >> bucketBitShift];
            const uint32 x   = xStart + i;
            ASSERT( dst < _maxEntriesPerIOBucket );

            yEntries[dst] = ( ( (uint64)y << kExtraBits ) | ( x >> kMinusKExtraBits ) ) & yMask;
            xEntries[dst] = x;
        }

        // Write to disk (and synchronize threads)
        if( self->BeginLockBlock() )
        {
            _ioQueue.WriteBucketElementsT( FileId::FX0  , true, yEntries.Ptr(), alignedElementCounts.Ptr(), elementCounts.Ptr() );
            _ioQueue.WriteBucketElementsT( FileId::META0, true, xEntries.Ptr(), alignedElementCounts.Ptr(), elementCounts.Ptr() );
            _ioQueue.SignalFence( _writeFence, bucket+2 );
            _ioQueue.CommitCommands();

            for( uint32 i = 0; i < _numBuckets; i++ )
                _context.bucketCounts[(int)TableId::Table1][i] += elementCounts[i];
        }
        self->EndLockBlock();
    }

    //-----------------------------------------------------------
    void GetNextBuffer( Job* self, const uint32 bucket, 
                        Span<uint32>& yEntries, Span<uint32>& xEntries,
                        Span<uint32>& elementCounts, Span<uint32>& alignedElementCounts )
    {
        const uint32 bucketIdx = bucket & 1; // % 2

        // if( bucket >= 2 && _writeFence.Value() < bucket-1 )
        if( bucket >= 2 )
        {
            // #TODO: Figure out if we can avoid the lock if already signaled.
            //        Like what's commented out above. However, we have to make sure that
            //        the signal is properly visible to all threads
            if( self->BeginLockBlock() )
            {
                _writeFence.Wait( bucket, _context.p1TableWaitTime[(int)TableId::Table1] );
            }
            self->EndLockBlock();
        }

        yEntries = _yEntries[bucketIdx];
        xEntries = _xEntries[bucketIdx];

        if( self->IsControlThread() )
        {
            elementCounts        = Span<uint32>( _elementCounts       [bucketIdx], _numBuckets );
            alignedElementCounts = Span<uint32>( _alignedElementCounts[bucketIdx], _numBuckets );
        }
    }

private:
    DiskPlotContext&    _context;
    DiskBufferQueue&    _ioQueue;
    Fence&              _writeFence;
    Span<uint32>        _blockBuffer;

    // I/O buffers
    Span<uint32> _yEntries[2];
    Span<uint32> _xEntries[2];
    uint32       _elementCounts       [2][_numBuckets] = {};
    uint32       _alignedElementCounts[2][_numBuckets] = {};
    
    Span<Span<uint32>> _offsets;

#if _DEBUG
    uint32 _maxEntriesPerIOBucket;
#endif
};


#if _DEBUG

//-----------------------------------------------------------
void DbgValidateF1( DiskPlotContext& context )
{
    Log::Line( "[DEBUG: Validating y and x]" );

    auto& ioQueue = *context.ioQueue;
    ioQueue.SeekBucket( FileId::FX0, 0, SeekOrigin::Begin );
    ioQueue.SeekBucket( FileId::META0, 0, SeekOrigin::Begin );
    ioQueue.CommitCommands();

    const uint64 entryCount = context.entryCounts[(int)TableId::Table1];
    
    Span<uint64> yReference = bbcvirtallocboundednuma_span<uint64>( entryCount );
    Span<uint32> xReference = bbcvirtallocboundednuma_span<uint32>( entryCount );
    Span<uint64> yBuffer    = bbcvirtallocboundednuma_span<uint64>( entryCount );
    Span<uint32> xBuffer    = bbcvirtallocboundednuma_span<uint32>( entryCount );
    Span<uint64> tmpBuffer  = bbcvirtallocboundednuma_span<uint64>( entryCount );
    Span<uint32> tmpBuffer2 = bbcvirtallocboundednuma_span<uint32>( entryCount );

    // Load reference values
    Log::Line( " Loading reference values..." );
    FatalIf( !Debug::LoadRefTableByName( "y.t1.tmp", yReference ), "Failed to load reference y table." );
    ASSERT( yReference.Length() == entryCount );
    FatalIf( !Debug::LoadRefTableByName( "x.t1.tmp", xReference ), "Failed to load reference x table." );
    ASSERT( xReference.Length() == entryCount );
    ASSERT( yReference.Length() == xReference.Length() );
    
    // Load our values
    Log::Line( " Loading our values..." );
    Fence& fence = context.fencePool->RequireFence();

    {
        // const size_t blockSizeEntries = context.tmp2BlockSize / sizeof( uint32 );
        // const uint32 bucketSize       = (uint32)RoundUpToNextBoundaryT( (size_t)entryCount / 2, blockSizeEntries );
        
        Span<uint64> yReader = yBuffer;
        Span<uint32> xReader = xBuffer;

        const uint32 numBuckets = context.numBuckets;
        for( uint32 bucket = 0; bucket < numBuckets; bucket++ )
        {
            Span<uint32> yBucket = tmpBuffer.template As<uint32>();
            Span<uint32> xBucket = tmpBuffer2;

            ioQueue.ReadBucketElementsT( FileId::FX0  , true, yBucket );
            ioQueue.ReadBucketElementsT( FileId::META0, true, xBucket );
            ioQueue.SignalFence( fence );
            ioQueue.CommitCommands();
            fence.Wait();

            ASSERT( yBucket.Length() && yBucket.Length() == xBucket.Length() );

            const uint32 k          = 32;
            const uint32 bucketBits = bblog2( numBuckets );
            const uint32 yBits      = k + kExtraBits - bucketBits;
            const uint64 yMask      = ((uint64)bucket) << yBits;
            
            for( size_t i = 0; i < yBucket.Length(); i++ )
                yReader[i] = yMask | yBucket[i];
            
            xBucket.CopyTo( xReader );

            // Sort bucket
            RadixSort256::SortWithKey<BB_DP_MAX_JOBS>( *context.threadPool, 
                yReader.Ptr(), tmpBuffer.Ptr(), xReader.Ptr(), tmpBuffer2.Ptr(), yBucket.Length() );

            yReader = yReader.Slice( yBucket.Length() );
            xReader = xReader.Slice( xBucket.Length() );
        }

        ASSERT( yReader.Length() == xReader.Length() );
    }

    // Sort
    // Log::Line( " Sorting..." );
    // RadixSort256::SortWithKey<BB_DP_MAX_JOBS>( *context.threadPool, yBuffer.Ptr(), tmpBuffer.Ptr(), xBuffer.Ptr(), tmpBuffer2.Ptr(), yBuffer.Length() );

    // Compare
    Log::Line( " Comparing values..." );
    for( size_t i = 0; i < yBuffer.Length(); i++ )
    {
        const uint64 y  = yBuffer[i];
        const uint64 yr = yReference[i];
        const uint32 x  = xBuffer[i];
        const uint32 xr = xReference[i];

        ASSERT( y == yr );

        if( x != xr )
        {
            if( xBuffer[i] == xReference[i+1] ) // 2-way match?
            {
                i++;
                continue;
            }
            else if( xBuffer[i] == xReference[i+2] &&   // 3 way match?
                     xBuffer[i+2] == xReference[i] && 
                     xBuffer[i+1] == xReference[i+1])  
            {
                i+=2;
                continue;
            }
            ASSERT( false );
        }
    }
    // ASSERT( yReference.EqualElements( yBuffer ) );
    // ASSERT( xReference.EqualElements( xBuffer ) );

    Log::Line( " Finished." );
    Log::Line( "" );

    // Cleanup
    context.fencePool->RestoreAllFences();
    bbvirtfreebounded( yReference.Ptr() );
    bbvirtfreebounded( xReference.Ptr() );
    bbvirtfreebounded( yBuffer.Ptr() );
    bbvirtfreebounded( xBuffer.Ptr() );
    bbvirtfreebounded( tmpBuffer.Ptr() );
    bbvirtfreebounded( tmpBuffer2.Ptr() );
}

#endif

