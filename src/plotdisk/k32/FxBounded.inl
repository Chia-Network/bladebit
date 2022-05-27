#pragma once
#include "plotdisk/DiskPlotContext.h"
#include "plotdisk/DiskPlotConfig.h"
#include "plotdisk/DiskBufferQueue.h"
#include "plotdisk/BitBucketWriter.h"
#include "plotdisk/MapWriter.h"
#include "util/StackAllocator.h"
#include "FpMatchBounded.inl"
#include "b3/blake3.h"

#if _DEBUG
    #include "algorithm/RadixSort.h"
#endif

typedef uint32 K32Meta1;
typedef uint64 K32Meta2;
struct K32Meta3 { uint32 m0, m1, m2; };
struct K32Meta4 { uint64 m0, m1; };
struct K32NoMeta {};

template<TableId rTable>
struct K32MetaType{};

template<> struct K32MetaType<TableId::Table1>{ using In = K32NoMeta; using Out = K32Meta1;  };
template<> struct K32MetaType<TableId::Table2>{ using In = K32Meta1;  using Out = K32Meta2;  };
template<> struct K32MetaType<TableId::Table3>{ using In = K32Meta2;  using Out = K32Meta4;  };
template<> struct K32MetaType<TableId::Table4>{ using In = K32Meta4;  using Out = K32Meta4;  };
template<> struct K32MetaType<TableId::Table5>{ using In = K32Meta4;  using Out = K32Meta3;  };
template<> struct K32MetaType<TableId::Table6>{ using In = K32Meta3;  using Out = K32Meta2;  };
template<> struct K32MetaType<TableId::Table7>{ using In = K32Meta2;  using Out = K32NoMeta; };

template<TableId rTable> struct K32TYOut { using Type = uint64; };
template<>               struct K32TYOut<TableId::Table7> { using Type = uint32; };

template<TableId rTable, uint32 _numBuckets>
class DiskPlotFxBounded
{
    using TMetaIn  = typename K32MetaType<rTable>::In;
    using TMetaOut = typename K32MetaType<rTable>::Out;
    using Job     = AnonPrefixSumJob<uint32>;
    static constexpr uint32 _k               = 32;
    static constexpr uint64 _maxTableEntries = (1ull << _k) - 1;
    static constexpr uint32 _bucketBits      = bblog2( _numBuckets );

    static constexpr uint32 _pairsMaxDelta   = 512;
    static constexpr uint32 _pairsRightBits  = bblog2( _pairsMaxDelta );
    static constexpr uint32 _pairBitSize     = _k - _bucketBits + _pairsRightBits;


public:

    //-----------------------------------------------------------
    DiskPlotFxBounded( DiskPlotContext& context )
        : _context       ( context )
        , _ioQueue       ( *context.ioQueue )
        , _yReadFence    ( context.fencePool ? context.fencePool->RequireFence() : *(Fence*)nullptr )
        , _metaReadFence ( context.fencePool ? context.fencePool->RequireFence() : *(Fence*)nullptr )
        , _indexReadFence( context.fencePool ? context.fencePool->RequireFence() : *(Fence*)nullptr )
        , _fxWriteFence  ( context.fencePool ? context.fencePool->RequireFence() : *(Fence*)nullptr )
        , _pairWriteFence( context.fencePool ? context.fencePool->RequireFence() : *(Fence*)nullptr )
        , _mapWriteFence ( context.fencePool ? context.fencePool->RequireFence() : *(Fence*)nullptr )
    {
        const TableId lTable = rTable - 1;
        _yId   [0] = FileId::FX0    + (FileId)((int)lTable & 1);
        _idxId [0] = FileId::INDEX0 + (FileId)((int)lTable & 1);
        _metaId[0] = FileId::META0  + (FileId)((int)lTable & 1);

        _yId   [1] = FileId::FX0    + (FileId)((int)rTable & 1);
        _idxId [1] = FileId::INDEX0 + (FileId)((int)rTable & 1);
        _metaId[1] = FileId::META0  + (FileId)((int)rTable & 1);
    }

    //-----------------------------------------------------------
    ~DiskPlotFxBounded()
    {
    }

    //-----------------------------------------------------------
    static size_t GetRequiredHeapSize( const size_t t1BlockSize, const size_t t2BlockSize, const uint32 threadCount )
    {
        DiskPlotContext cx = {};

        DiskPlotFxBounded<TableId::Table4, _numBuckets> instance( cx );

        DummyAllocator allocator;
        instance.AllocateBuffers( allocator, t1BlockSize, t2BlockSize, threadCount, true );
        const size_t requiredSize = allocator.Size();

        return requiredSize;
    }

    //-----------------------------------------------------------
    void AllocateBuffers( IAllocator& allocator, const size_t t1BlockSize, const size_t t2BlockSize, const uint32 threadCount, const bool dryRun )
    {
        const uint64 kEntryCount      = 1ull << _k;
        const uint64 entriesPerBucket = (uint64)( kEntryCount / _numBuckets * BB_DP_XTRA_ENTRIES_PER_BUCKET );
        _entriesPerBucket = entriesPerBucket;

        // Read buffers
        _yBuffers[0]     = allocator.CAllocSpan<uint32>( entriesPerBucket, t2BlockSize );
        _yBuffers[1]     = allocator.CAllocSpan<uint32>( entriesPerBucket, t2BlockSize );

        _metaBuffers[0]  = allocator.CAllocSpan<K32Meta4>( entriesPerBucket, t2BlockSize );
        _metaBuffers[1]  = allocator.CAllocSpan<K32Meta4>( entriesPerBucket, t2BlockSize );

        _indexBuffers[0] = allocator.CAllocSpan<uint32>( entriesPerBucket, t2BlockSize );
        _indexBuffers[1] = allocator.CAllocSpan<uint32>( entriesPerBucket, t2BlockSize );


        // Work buffers
        _yTmp       = allocator.CAllocSpan<uint64>  ( entriesPerBucket, t2BlockSize );
        _metaTmp[0] = allocator.CAllocSpan<K32Meta4>( entriesPerBucket, t2BlockSize );
        _metaTmp[1] = allocator.CAllocSpan<K32Meta4>( entriesPerBucket, t2BlockSize );
        _sortKey    = allocator.CAllocSpan<uint32>  ( entriesPerBucket );
        _pairBuffer = allocator.CAllocSpan<Pair>    ( entriesPerBucket );

        _sliceCountY[0]           = allocator.CAllocSpan<uint32>( _numBuckets );
        _sliceCountY[1]           = allocator.CAllocSpan<uint32>( _numBuckets );
        _sliceCountMeta[0]        = allocator.CAllocSpan<uint32>( _numBuckets );
        _sliceCountMeta[1]        = allocator.CAllocSpan<uint32>( _numBuckets );
        _alignedSliceCountY[0]    = allocator.CAllocSpan<uint32>( _numBuckets );
        _alignedSliceCountY[1]    = allocator.CAllocSpan<uint32>( _numBuckets );
        _alignedsliceCountMeta[0] = allocator.CAllocSpan<uint32>( _numBuckets );
        _alignedsliceCountMeta[1] = allocator.CAllocSpan<uint32>( _numBuckets );

        _offsetsY    = allocator.CAllocSpan<uint32*>( threadCount );
        _offsetsMeta = allocator.CAllocSpan<uint32*>( threadCount );

        for( uint32 i = 0; i < _offsetsY.Length(); i++ )
        {
            Span<uint32> offsetsY    = allocator.CAllocSpan<uint32>( _numBuckets );
            Span<uint32> offsetsMeta = allocator.CAllocSpan<uint32>( _numBuckets );

            if( !dryRun )
            {
                offsetsY   .ZeroOutElements();
                offsetsMeta.ZeroOutElements();

                _offsetsY   [i] = offsetsY   .Ptr();
                _offsetsMeta[i] = offsetsMeta.Ptr();
            }
        }

        // Write buffers
        _yWriteBuffer     = allocator.CAllocSpan<uint32  >( entriesPerBucket, t2BlockSize );
        _indexWriteBuffer = allocator.CAllocSpan<uint32  >( entriesPerBucket, t2BlockSize );
        _metaWriteBuffer  = allocator.CAllocSpan<TMetaOut>( entriesPerBucket, t2BlockSize );

        _map = allocator.CAllocSpan<uint64>( entriesPerBucket );

        const size_t pairsWriteSize = CDiv( entriesPerBucket * _pairBitSize, 8 );
        _pairsWriteBuffer = allocator.AllocT<byte>( pairsWriteSize, t1BlockSize );
        byte* pairBlocks  = (byte*)allocator.CAlloc( 1, t1BlockSize, t1BlockSize );
        
        if( !dryRun )
        {
            _pairBitWriter = BitBucketWriter<1>( _ioQueue, FileId::T1 + (FileId)rTable, pairBlocks );
            
            // _mapWriter = MapWriter<_numBuckets, false>( _ioQueue, FileId::MAP2 + (FileId)(rTable-1), allocator, 
            //                                             entriesPerBucket, t1BlockSize, _mapWriteFence, _tableIOWait );
        }
        else
        {
            // _mapWriter = MapWriter<_numBuckets, false>( entriesPerBucket, allocator, t1BlockSize );
        }
        // _pairsL = allocator.CAllocSpan<uint32>( entriesPerBucket );
        // _pairsR = allocator.CAllocSpan<uint16>( entriesPerBucket );

    }

    // Generate fx for a whole table
    //-----------------------------------------------------------
    void Run()
    {
        // Prepare input files
        _ioQueue.SeekBucket( _yId[0], 0, SeekOrigin::Begin );

        if constexpr( rTable > TableId::Table2 )
            _ioQueue.SeekBucket( _idxId[0], 0, SeekOrigin::Begin );    

        _ioQueue.SeekBucket( _metaId[0], 0, SeekOrigin::Begin );
        _ioQueue.CommitCommands();
        
        // Allocate buffers
        StackAllocator allocator( _context.heapBuffer, _context.heapSize );
        AllocateBuffers( allocator, _context.tmp1BlockSize, _context.tmp2BlockSize,  _context.fpThreadCount, false );

        // Init buffers
        const uint32 threadCount = _context.fpThreadCount;

        {
            const uint32 matchesPerThread =_entriesPerBucket / threadCount;
            auto pairBuffer = _pairBuffer;

            for( uint32 i = 0; i < threadCount; i++ )
            {
                _pairs[i]  = pairBuffer.Slice( 0, matchesPerThread );
                pairBuffer = pairBuffer.Slice( matchesPerThread );
            }

            for( uint32 i = 0; i < _numBuckets; i++ )
            {
                const uint32 bufIdx = i & 1; // % 2
                _y    [i] = _yBuffers    [bufIdx];
                _index[i] = _indexBuffers[bufIdx];
                _meta [i] = _metaBuffers [bufIdx].As<TMetaIn>();
            }
        }

        // Set some initial fence status
        _mapWriteFence.Signal( 0 );
        
        // Run mmulti-threaded
        Job::Run( *_context.threadPool, threadCount, [=]( Job* self ) {
            RunMT( self );
        });

        _context.entryCounts[(int)rTable] = _tableEntryCount;

        Log::Line( " Sorting      : Completed in %.2lf seconds.", TicksToSeconds( _sortTime  ) );
        Log::Line( " Distribution : Completed in %.2lf seconds.", TicksToSeconds( _distributeTime ) );
        Log::Line( " Matching     : Completed in %.2lf seconds.", TicksToSeconds( _matchTime ) );
        Log::Line( " Fx           : Completed in %.2lf seconds.", TicksToSeconds( _fxTime    ) );

        // Ensure all I/O has completed
        _fxWriteFence.Wait( _numBuckets, _tableIOWait );
        _context.fencePool->RestoreAllFences();

        #if _DEBUG
            ValidateIndices();
        #endif
    }

private:
    //-----------------------------------------------------------
    void RunMT( Job* self )
    {
        const uint32 threadCount = self->JobCount();
        const uint32 id          = self->JobId();

        ReadNextBucket( self, 0 );

        for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
        {
            // if( bucket == 63 && self->JobId() == 0 ) BBDebugBreak();

            ReadNextBucket( self, bucket + 1 ); // Read next bucket in background
            WaitForFence( self, _yReadFence, bucket );

            Span<uint32> yInput     = _y[bucket];
            const uint32 entryCount = yInput.Length();
            
            Span<uint32> sortKey = _sortKey.SliceSize( entryCount );
            SortY( self, entryCount, yInput.Ptr(), _yTmp.As<uint32>().Ptr(), sortKey.Ptr(), _metaTmp[0].As<uint32>().Ptr() );


            ///
            /// Write reverse map, given the previous table's y origin indices
            ///
            if constexpr ( rTable > TableId::Table2 )
            {
                WaitForFence( self, _indexReadFence, bucket );
                ASSERT( _index[bucket].Length() == entryCount );

                Span<uint32> indices = _metaTmp[0].As<uint32>().SliceSize( entryCount );

                SortOnYKey( self, sortKey, _index[bucket], indices );
                WriteMap( self, indices, _map.SliceSize( entryCount ), _mapOffset );
            }
            else
            {
                // #TOOD Write T1 x
            }


            ///
            /// Match
            ///
            Span<Pair> matches = Match( self, bucket, yInput );

            // Count the total match count
            uint32 totalMatches = 0;
            uint32 matchOffset  = 0;

            for( uint32 i = 0; i < threadCount; i++ )
                totalMatches += (uint32)_pairs[i].Length();

            for( uint32 i = 0; i < id; i++ )
                matchOffset += (uint32)_pairs[i].Length();

            ASSERT( totalMatches <= _entriesPerBucket );

            // Prevent overflow entries
            const uint64 tableEntryCount = _tableEntryCount;
            if( bucket == _numBuckets-1 && (tableEntryCount + totalMatches) > _maxTableEntries )
            {
                // Prevent calculating fx for overflow matches
                if( self->IsLastThread() )
                {
                    const uint32 overflowEntries = (uint32)( (tableEntryCount + totalMatches) - _maxTableEntries );
                    ASSERT( overflowEntries < matches.Length() );

                    matches = matches.SliceSize( matches.Length() - overflowEntries );
                }

                totalMatches = _maxTableEntries;
            }

            WritePairs( self, bucket, totalMatches, matches, matchOffset );

            // Sort meta on Y
            Span<TMetaIn> metaTmp = _meta[bucket].SliceSize( yInput.Length() );
            Span<TMetaIn> metaIn  = _metaTmp[0].As<TMetaIn>().SliceSize( yInput.Length() );

            WaitForFence( self, _metaReadFence, bucket );
            SortOnYKey( self, sortKey, metaTmp, metaIn );

            // Gen fx & write
            {
                Span<uint64>   yOut    = _yTmp.Slice( matchOffset, matches.Length() );
                Span<TMetaOut> metaOut = _metaTmp[1].As<TMetaOut>().Slice( matchOffset, matches.Length() );  //( (TMetaOut*)metaTmp.Ptr(), matches.Length() );

                TimePoint timer;
                if( self->IsControlThread() )
                    timer = TimerBegin();

                GenFx( self, bucket, matches, yInput, metaIn, yOut, metaOut );
                self->SyncThreads();

                if( self->IsControlThread() )
                    _fxTime += TimerEndTicks( timer );

                WriteEntries( self, bucket,  (uint32)_tableEntryCount + matchOffset, yOut, metaOut, _yWriteBuffer, _metaWriteBuffer, _indexWriteBuffer );
            }

            if( self->IsControlThread() )
            {
                _tableEntryCount += totalMatches;
                _mapOffset       += entryCount;
            }
        }
    }

    //-----------------------------------------------------------
    void SortY( Job* self, const uint64 entryCount, uint32* ySrc, uint32* yTmp, uint32* sortKeySrc, uint32* sortKeyTmp )
    {
        constexpr uint32 Radix      = 256;
        constexpr uint32 iterations = 4;
        const     uint32 shiftBase  = 8;

        TimePoint timer;
        if( self->IsControlThread() )
            timer = TimerBegin();

        uint32 shift = 0;

        uint32 counts     [Radix];
        uint32 prefixSum  [Radix];
        uint32 totalCounts[Radix];

        uint64 length, offset, end;
        GetThreadOffsets( self, entryCount, length, offset, end );

        uint32* input    = ySrc;
        uint32* tmp      = yTmp;
        uint32* keyInput = sortKeySrc;
        uint32* keyTmp   = sortKeyTmp;

        // Gen sort key first
        for( uint64 i = offset; i < end; i++ )
            sortKeySrc[i] = (uint32)i;

        for( uint32 iter = 0; iter < iterations ; iter++, shift += shiftBase )
        {
            // Zero-out the counts
            memset( counts, 0, sizeof( counts ) );

            // Grab our scan region from the input
            const uint32* src    = input    + offset;
            const uint32* keySrc = keyInput + offset;

            // Store the occurrences of the current 'digit'
            for( uint64 i = 0; i < length; i++ )
                counts[(src[i] >> shift) & 0xFF]++;

            self->CalculatePrefixSum( Radix, counts, prefixSum, totalCounts );

            for( uint64 i = length; i > 0; )
            {
                const uint32 value  = src[--i];
                const uint32 bucket = (value >> shift) & 0xFF;

                const uint64 dstIdx = --prefixSum[bucket];

                tmp   [dstIdx] = value;
                keyTmp[dstIdx] = keySrc[i];
            }

            // Swap arrays
            std::swap( input, tmp );
            std::swap( keyInput, keyTmp );
            self->SyncThreads();
        }

        if( self->IsControlThread() )
            _sortTime += TimerEndTicks( timer );
    }

    //-----------------------------------------------------------
    template<typename T>
    void SortOnYKey( Job* self, const Span<uint32> key, const Span<T> input, Span<T> output )
    {
        TimePoint timer;
        if( self->IsControlThread() )
            timer = TimerBegin();

        uint32 count, offset, end;
        GetThreadOffsets( self, (uint32)key.Length(), count, offset, end );

        for( uint32 i = offset; i < end; i++ )
            output[i] = input[key[i]];


        self->SyncThreads();
        
        if( self->IsControlThread() )
            _sortTime += TimerEndTicks( timer );
    }

    //-----------------------------------------------------------
    Span<Pair> Match( Job* self, const uint32 bucket, Span<uint32> yEntries )
    {
        const uint32 id          = self->JobId();
        const uint32 threadCount = self->JobCount();

        // use metaTmp as a buffer for group boundaries
        Span<uint32> groupBuffer  = _metaTmp[0].As<uint32>();
        const uint32 maxGroups    = (uint32)( groupBuffer.Length() / threadCount );
        Span<uint32> groupIndices = groupBuffer.Slice( maxGroups * id, maxGroups );

        TimePoint timer;
        if( self->IsControlThread() )
            timer = TimerBegin();
    
        auto matches = _matcher.Match( self, bucket, yEntries, groupIndices, _pairs[id] );
        _pairs[id] = matches;

        self->SyncThreads();
        if( self->IsControlThread() )
            _matchTime += TimerEndTicks( timer );

        return matches;
    }

    //-----------------------------------------------------------
    void WriteMap( Job* self, const Span<uint32> bucketIndices, Span<uint64> mapOut, const uint32 tableOffset )
    {
        const uint32 bucketShift = _k - _bucketBits;

        uint32 counts     [_numBuckets] = { 0 };
        uint32 pfxSum     [_numBuckets];
        uint32 totalCounts[_numBuckets];

        uint32 count, offset, _;
        GetThreadOffsets( self, (uint32)bucketIndices.Length(), count, offset, _ );

        auto indices = bucketIndices.Slice( offset, count );

        // Count buckets
        for( size_t i = 0; i < indices.Length(); i++ )
        {
            const uint32 b = indices[i] >> bucketShift; ASSERT( b < _numBuckets );
            counts[b]++;
        }

        self->CalculatePrefixSum( _numBuckets, counts, pfxSum, totalCounts );

        // Wait for write fence
        if( self->BeginLockBlock() )
            _mapWriteFence.Wait();
        self->EndLockBlock();

        // Distribute as 64-bit entries
        const uint64 outOffset = tableOffset + offset;

        for( size_t i = 0; i < indices.Length(); i++ )
        {
            const uint64 origin = indices[i];
            const uint32 b      = origin >> bucketShift;    ASSERT( b < _numBuckets );

            const uint32 dstIdx = --pfxSum[b];

            mapOut[dstIdx] = (origin << _k) | (outOffset + i);  // (origin, dst index)
        }

        // Write map to disk
        if( self->BeginLockBlock() )
        {
            memcpy( _mapCounts, totalCounts , sizeof( totalCounts ) );

            _ioQueue.WriteBucketElementsT<uint64>( FileId::MAP2 + (FileId)rTable-1, mapOut.Ptr(), _mapCounts );
            _ioQueue.SignalFence( _mapWriteFence );
            _ioQueue.CommitCommands();
        }
        self->EndLockBlock();
    }

    //-----------------------------------------------------------
    void WritePairs( Job* self, const uint32 bucket, const uint32 totalMatchCount, const Span<Pair> matches, const uint64 dstOffset )
    {
        ASSERT( dstOffset + matches.length <= totalMatchCount );

        // Need to wait for our write buffer to be ready to use again
        if( self->BeginLockBlock() )
        {
            if( bucket > 0 )
                _pairWriteFence.Wait( bucket );

            uint64 bitBucketSizes = totalMatchCount * _pairBitSize;
            _pairBitWriter.BeginWriteBuckets( &bitBucketSizes, _pairsWriteBuffer );
        }
        self->EndLockBlock();

        // #TOOD: Write pairs raw? Or bucket-relative? Maybe raw for now

        // #NOTE: For now we write compressed as in standard FP.        
        BitWriter writer = _pairBitWriter.GetWriter( 0, dstOffset * _pairBitSize );

        ASSERT( matches.Length() > 2 );
        PackPairs( self, matches.Slice( 0, 2 ), writer );
        self->SyncThreads();
        PackPairs( self, matches.Slice( 2 ), writer );
        
        // Write to disk
        if( self->BeginLockBlock() )
        {
            _pairBitWriter.Submit();
            _ioQueue.SignalFence( _pairWriteFence, bucket + 1 );
            _ioQueue.CommitCommands();
        }
        self->EndLockBlock();
    }

    //-----------------------------------------------------------
    void PackPairs( Job* self, const Span<Pair> pairs, BitWriter& writer )
    {
        self;

        const uint32 shift = _pairsRightBits;
        const uint64 mask  = ( 1ull << shift ) - 1;

        const Pair* pair = pairs.Ptr();
        const Pair* end  = pair + pairs.Length();

        while( pair < end )
        {
            ASSERT( pair->right - pair->left < _pairsMaxDelta );

            writer.Write( ( (uint64)(pair->right - pair->left) << shift ) | ( pair->left & mask ), _pairBitSize );
            pair++;
        }
    }

    //-----------------------------------------------------------
    void WriteEntries( Job* self,
                       const uint32         bucket,
                       const uint32         idxOffset,
                       const Span<uint64>   yIn,
                       const Span<TMetaOut> metaIn,
                             Span<uint32>   yOut,
                             Span<TMetaOut> metaOut,
                             Span<uint32>   idxOut )
    {
        TimePoint timer;
        if( self->IsControlThread() )
            timer = TimerBegin();

        const uint32 blockSize = (uint32)_ioQueue.BlockSize( _yId[1] );
        const uint32 id        = (uint32)self->JobId();

        // Distribute to buckets
        const uint32 logBuckets = bblog2( _numBuckets );
        static_assert( kExtraBits <= logBuckets );

        const uint32 bucketShift = _k + kExtraBits - logBuckets;
        const uint64 yMask       = ( 1ull << bucketShift ) - 1;

        const int64 entryCount = (int64)yIn.Length();

        uint32 counts    [_numBuckets] = {};
        uint32 pfxSum    [_numBuckets];
        uint32 pfxSumMeta[_numBuckets];

        const uint32 sliceIdx = bucket & 1; // % 2

        Span<uint32> ySliceCounts          = _sliceCountY[sliceIdx];
        Span<uint32> metaSliceCounts       = _sliceCountMeta[sliceIdx];
        Span<uint32> yAlignedSliceCount    = _alignedSliceCountY[sliceIdx];
        Span<uint32> metaAlignedsliceCount = _alignedsliceCountMeta[sliceIdx];

        for( int64 i = 0; i < entryCount; i++ )
            counts[yIn[i] >> bucketShift]++;

        self->CalculateBlockAlignedPrefixSum<uint32>( _numBuckets, blockSize, counts, pfxSum, ySliceCounts.Ptr(), _offsetsY[id], yAlignedSliceCount.Ptr() );
        self->CalculateBlockAlignedPrefixSum<TMetaOut>( _numBuckets, blockSize, counts, pfxSumMeta, metaAlignedsliceCount.Ptr(), _offsetsMeta[id], metaAlignedsliceCount.Ptr() );

        // Distribute to buckets
        for( int64 i = 0; i < entryCount; i++ )
        {
            const uint64 y       = yIn[i];
            const uint32 yBucket = (uint32)(y >> bucketShift);
            const uint32 yDst    = --pfxSum    [yBucket];
            const uint32 metaDst = --pfxSumMeta[yBucket];

            yOut   [yDst]    = (uint32)(y & yMask);
            idxOut [yDst]    = idxOffset + (uint32)i;   ASSERT( (uint64)idxOffset + (uint64)i < (1ull << _k) );
            metaOut[metaDst] = metaIn[i];
        }

        // Write to disk
        if( self->BeginLockBlock() )
        {
            // #NOTE: Should we wait per file?
            if( bucket > 0 )
                _fxWriteFence.Wait( bucket, _tableIOWait ); 

            const FileId yId    = _yId   [1];
            const FileId metaId = _metaId[1];
            const FileId idxId  = _idxId [1];

            _ioQueue.WriteBucketElementsT<uint32>  ( yId   , yOut   .Ptr(),  yAlignedSliceCount.Ptr()   , ySliceCounts.Ptr() );
            _ioQueue.WriteBucketElementsT<uint32>  ( idxId , idxOut .Ptr(),  yAlignedSliceCount.Ptr()   , ySliceCounts.Ptr() );
            _ioQueue.WriteBucketElementsT<TMetaOut>( metaId, metaOut.Ptr(),  metaAlignedsliceCount.Ptr(), metaSliceCounts.Ptr() );
            _ioQueue.SignalFence( _fxWriteFence, bucket+1 );
            _ioQueue.CommitCommands();
        }
        self->EndLockBlock();


        if( self->IsControlThread() )
            _distributeTime += TimerEndTicks( timer );
    }

    //-----------------------------------------------------------
    template<typename TYOut>
    void GenFx( Job* self,
                const uint32        bucket,
                const Span<Pair>    pairs,
                const Span<uint32>  yIn,
                const Span<TMetaIn> metaIn,
                Span<TYOut>         yOut,
                Span<TMetaOut>      metaOut )
    {
        constexpr size_t MetaInMulti  = TableMetaIn <rTable>::Multiplier;
        constexpr size_t MetaOutMulti = TableMetaOut<rTable>::Multiplier;
        static_assert( MetaInMulti != 0, "Invalid metaKMultiplier" );


        const uint32 k           = 32;
        const uint32 shiftBits   = MetaOutMulti == 0 ? 0 : kExtraBits;  // Table 7 (identified by 0 metadata output) we don't have k + kExtraBits sized y's.
                                                                        // so we need to shift by 32 bits, instead of 26.
        const uint32 ySize       = k + kExtraBits;         // = 38
        const uint32 yShift      = 64 - (k + shiftBits);   // = 26 or 32
        const size_t metaSize    = k * MetaInMulti;
        const size_t metaSizeLR  = metaSize * 2;

        const size_t bufferSize  = CDiv( ySize + metaSizeLR, 8 );

        // const uint32 id         = self->JobId();
        const uint32 matchCount = pairs.Length();
        const uint64 yMask      = (uint64)bucket << 32;

        // Hashing
        uint64 input [5]; // y + L + R
        uint64 output[4]; // blake3 hashed output

        blake3_hasher hasher;

        static_assert( bufferSize <= sizeof( input ), "Invalid fx input buffer size." );

        #if _DEBUG
            uint64 prevY    = yIn[pairs[0].left];
            uint64 prevLeft = 0;
        #endif

        for( int64 i = 0; i < matchCount; i++ )
        {
            const auto& pair = pairs[i];
            const uint32 left  = pair.left;
            const uint32 right = pair.right;
            ASSERT( left < right );

            const uint64 y = yMask | (uint64)yIn[left];

            #if _DEBUG
                ASSERT( y >= prevY );
                ASSERT( left >= prevLeft );
                prevY    = y;
                prevLeft = left;
            #endif

            // Extract metadata
            auto& mOut = metaOut[i];

            if constexpr( MetaInMulti == 1 )
            {
                const uint64 l = metaIn[left ];
                const uint64 r = metaIn[right];

                input[0] = Swap64( y << 26 | l >> 6  );
                input[1] = Swap64( l << 58 | r << 26 );

                // Metadata is just L + R of 8 bytes
                if constexpr( MetaOutMulti == 2 )
                    mOut = l << 32 | r;
            }
            else if constexpr( MetaInMulti == 2 )
            {
                const uint64 l = metaIn[left ];
                const uint64 r = metaIn[right];

                input[0] = Swap64( y << 26 | l >> 38 );
                input[1] = Swap64( l << 26 | r >> 38 );
                input[2] = Swap64( r << 26 );

                // Metadata is just L + R again of 16 bytes
                if constexpr( MetaOutMulti == 4 )
                {
                    mOut.m0 = l;
                    mOut.m1 = r;
                }
            }
            else if constexpr( MetaInMulti == 3 )
            {
                const uint64 l0 = (uint64)metaIn[left ].m0 | ( (uint64)metaIn[left ].m1 << 32 );
                const uint64 l1 = metaIn[left ].m2;
                const uint64 r0 = (uint64)metaIn[right].m0 | ( (uint64)metaIn[right].m1 << 32 );
                const uint64 r1 = metaIn[right].m2;

                input[0] = Swap64( y  << 26 | l0 >> 38 );
                input[1] = Swap64( l0 << 26 | l1 >> 6  );
                input[2] = Swap64( l1 << 58 | r0 >> 6  );
                input[3] = Swap64( r0 << 58 | r1 << 26 );
            }
            else if constexpr( MetaInMulti == 4 )
            {
                // const uint64 l0 = metaInA[left ];
                // const uint64 l1 = metaInB[left ];
                // const uint64 r0 = metaInA[right];
                // const uint64 r1 = metaInB[right];
                const K32Meta4 l = metaIn[left];
                const K32Meta4 r = metaIn[right];

                input[0] = Swap64( y    << 26 | l.m0 >> 38 );
                input[1] = Swap64( l.m0 << 26 | l.m1 >> 38 );
                input[2] = Swap64( l.m1 << 26 | r.m0 >> 38 );
                input[3] = Swap64( r.m0 << 26 | r.m1 >> 38 );
                input[4] = Swap64( r.m1 << 26 );
            }

            // Hash the input
            blake3_hasher_init( &hasher );
            blake3_hasher_update( &hasher, input, bufferSize );
            blake3_hasher_finalize( &hasher, (uint8_t*)output, sizeof( output ) );

            const uint64 f = Swap64( *output ) >> yShift;
            yOut[i] = (TYOut)f;

            if constexpr ( MetaOutMulti == 2 && MetaInMulti == 3 )
            {
                const uint64 h0 = Swap64( output[0] );
                const uint64 h1 = Swap64( output[1] );

                mOut = h0 << ySize | h1 >> 26;
            }
            else if constexpr ( MetaOutMulti == 3 )
            {
                const uint64 h0 = Swap64( output[0] );
                const uint64 h1 = Swap64( output[1] );
                const uint64 h2 = Swap64( output[2] );

                uint64 m0 = h0 << ySize | h1 >> 26;
                mOut.m0 = (uint32)m0;
                mOut.m0 = (uint32)(m0 >> 32);
                mOut.m2 = (uint32)( ((h1 << 6) & 0xFFFFFFC0) | h2 >> 58 );
            }
            else if constexpr ( MetaOutMulti == 4 && MetaInMulti != 2 ) // In = 2 is calculated above with L + R
            {
                const uint64 h0 = Swap64( output[0] );
                const uint64 h1 = Swap64( output[1] );
                const uint64 h2 = Swap64( output[2] );

                mOut.m0 = h0 << ySize | h1 >> 26;
                mOut.m1 = h1 << 38    | h2 >> 26;
            }
        }
    }

    //-----------------------------------------------------------
    void ReadNextBucket( Job* self, const uint32 bucket )
    {
        if( bucket >= _numBuckets )
            return;

        if( !self->IsControlThread() )
            return;

        _ioQueue.ReadBucketElementsT( _yId[0], _y[bucket] );
        _ioQueue.SignalFence( _yReadFence, bucket + 1 );

        if constexpr ( rTable > TableId::Table2 )
        {
            _ioQueue.ReadBucketElementsT( _idxId[0], _index[bucket] );
            _ioQueue.SignalFence( _indexReadFence, bucket + 1 );
        }

        _ioQueue.ReadBucketElementsT<TMetaIn>( _metaId[0], _meta[bucket] );
        _ioQueue.SignalFence( _metaReadFence, bucket + 1 );

        _ioQueue.CommitCommands();
    }

    //-----------------------------------------------------------
    void WaitForFence( Job* self, Fence& fence, const uint32 bucket )
    {
        if( self->BeginLockBlock() )
            fence.Wait( bucket+1, _tableIOWait );

        self->EndLockBlock();
    }

private:
    // DEBUG
    #if _DEBUG
    //-----------------------------------------------------------
    void ValidateIndices()
    {
        Log::Line( "[DEBUG: Validating table %u indices]", rTable+1 );

        // Load indices
        const uint64 entryCount = _context.entryCounts[(int)rTable];

        const FileId fileId = _idxId[1];
        _ioQueue.SeekBucket( fileId, 0, SeekOrigin::Begin );
        _ioQueue.CommitCommands();

        Span<uint32> indices    = bbcvirtallocboundednuma_span<uint32>( entryCount );
        Span<uint32> tmpIndices = bbcvirtallocboundednuma_span<uint32>( entryCount );
        
        {
            Log::Line( " Loading indices" );
            Fence fence;
            Span<uint32> writer = indices;

            for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
            {
                Span<uint32> reader = tmpIndices;

                _ioQueue.ReadBucketElementsT<uint32>( fileId, reader );
                _ioQueue.SignalFence( fence );
                _ioQueue.CommitCommands();
                fence.Wait();

                reader.CopyTo( writer );
                writer = writer.Slice( reader.Length() );
            }
        }

        // Now sort our entries
        Log::Line( " Sorting indices" );
        RadixSort256::Sort<BB_DP_MAX_JOBS>( *_context.threadPool, indices.Ptr(), tmpIndices.Ptr(), indices.Length() );

        // Now validate them
        Log::Line( " Validating indices" );
        for( size_t i = 0; i < indices.Length(); i++ )
        {
            ASSERT( indices[i] == i );
        }

        Log::Line( " Indices are valid" );
        bbvirtfreebounded( indices.Ptr() );
        bbvirtfreebounded( tmpIndices.Ptr() );
    }
    #endif

private:
    DiskPlotContext&    _context;
    DiskBufferQueue&    _ioQueue;

    uint32              _entriesPerBucket = 0;
    std::atomic<uint32> _mapOffset        = 0;  // For writing maps
    std::atomic<uint64> _tableEntryCount  = 0;  // For writing indices

    // I/O
    FileId              _yId   [2];
    FileId              _idxId [2];
    FileId              _metaId[2];
    BitBucketWriter<1>  _pairBitWriter;

    // Read buffers
    Span<uint32>     _yBuffers    [2];
    Span<uint32>     _indexBuffers[2];
    Span<K32Meta4>   _metaBuffers [2];

    // Read views
    Span<uint32>     _y    [_numBuckets];
    Span<uint32>     _index[_numBuckets];
    Span<TMetaIn>    _meta [_numBuckets];

    // Write buffers
    Span<uint32>     _yWriteBuffer;
    Span<uint32>     _indexWriteBuffer;
    Span<TMetaOut>   _metaWriteBuffer;
    Span<uint64>     _map;
    byte*            _pairsWriteBuffer;
    
    // Working buffers
    Span<uint64>     _yTmp;
    Span<uint32>     _sortKey;
    Span<K32Meta4>   _metaTmp[2];
    Span<Pair>       _pairBuffer;

    // Working views
    Span<Pair>       _pairs[BB_DP_MAX_JOBS];    // Pairs buffer divided per thread

    // Matching
    FxMatcherBounded _matcher;

    // Map
    // MapWriter<_numBuckets, false> _mapWriter;
    uint32 _mapCounts[_numBuckets];

    // Distributing to buckets
    Span<uint32*>    _offsetsY;
    Span<uint32*>    _offsetsMeta;
    Span<uint32>     _sliceCountY   [2];
    Span<uint32>     _sliceCountMeta[2];
    Span<uint32>     _alignedSliceCountY   [2];
    Span<uint32>     _alignedsliceCountMeta[2];

    // I/O Synchronization fences
    Fence& _yReadFence;
    Fence& _metaReadFence;
    Fence& _indexReadFence;
    Fence& _fxWriteFence;
    Fence& _pairWriteFence;
    Fence& _mapWriteFence;


public:
    // Timings
    Duration _tableIOWait    = Duration::zero();
    Duration _sortTime       = Duration::zero();
    Duration _distributeTime = Duration::zero();
    Duration _matchTime      = Duration::zero();
    Duration _fxTime         = Duration::zero();
};

