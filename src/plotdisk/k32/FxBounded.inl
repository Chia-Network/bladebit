#pragma once
#include "plotdisk/DiskPlotContext.h"
#include "plotdisk/DiskPlotConfig.h"
#include "plotdisk/DiskBufferQueue.h"
#include "util/StackAllocator.h"
#include "FpMatchBounded.inl"
#include "b3/blake3.h"

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
    static constexpr uint64 _maxTableEntries = 1ull << _k;

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
    static size_t GetRequiredHeapSize( const size_t t1BlockSize, const size_t t2BlockSize )
    {
        DiskPlotContext cx = {};

        DiskPlotFxBounded<rTable, _numBuckets> instance( cx );

        DummyAllocator allocator;
        instance.AllocateBuffers( allocator, t1BlockSize, t2BlockSize );
        const size_t requiredSize = allocator.Size();

        return requiredSize;
    }

    //-----------------------------------------------------------
    void AllocateBuffers( IAllocator& allocator, const size_t t1BlockSize, const size_t t2BlockSize )
    {
        const uint64 kEntryCount      = 1ull << _k;
        const uint64 entriesPerBucket = (uint64)( kEntryCount / _numBuckets * BB_DP_XTRA_ENTRIES_PER_BUCKET );
        _entriesPerBucket = entriesPerBucket;

        _y[0] = allocator.CAllocSpan<uint32>( entriesPerBucket, t2BlockSize );
        _y[1] = allocator.CAllocSpan<uint32>( entriesPerBucket, t2BlockSize );
        _yTmp = allocator.CAllocSpan<uint32>( entriesPerBucket, t2BlockSize );

        _sortKey = allocator.CAllocSpan<uint32>( entriesPerBucket );

        _meta[0]  = Span<TMetaIn>( (TMetaIn*)allocator.CAlloc<Meta4>( entriesPerBucket, t2BlockSize ), entriesPerBucket );
        _meta[1]  = Span<TMetaIn>( (TMetaIn*)allocator.CAlloc<Meta4>( entriesPerBucket, t2BlockSize ), entriesPerBucket );
        _metaTmp  = allocator.CAllocSpan<Meta4>( entriesPerBucket, t2BlockSize );

        _index[0] = allocator.CAllocSpan<uint32>( entriesPerBucket, t2BlockSize );
        _index[1] = allocator.CAllocSpan<uint32>( entriesPerBucket, t2BlockSize );

        _map        = allocator.CAllocSpan<uint64>( entriesPerBucket );
        _pairBuffer = allocator.CAllocSpan<Pair>  ( entriesPerBucket );

        _pairsL = allocator.CAllocSpan<uint32>( entriesPerBucket );
        _pairsR = allocator.CAllocSpan<uint16>( entriesPerBucket );
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
        AllocateBuffers( allocator, _context.tmp1BlockSize, _context.tmp2BlockSize );

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
        }
        
        // Run mmulti-threaded
        Job::Run( *_context.threadPool, threadCount, [=]( Job* self ) {
            RunMT( self );
        });

        // Ensure all I/O has completed
        Fence& fence = _context.fencePool->RequireFence();
        fence.Reset( 0 );
        _ioQueue.SignalFence( fence, 1 );
        _ioQueue.CommitCommands();
        fence.Wait();

        _context.fencePool->RestoreAllFences();
    }

private:
    //-----------------------------------------------------------
    void RunMT( Job* self )
    {
        const TableId lTable      = rTable-1;
        const uint32  threadCount = self->JobCount();

        ReadNextBucket( self, 0 );

        for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
        {
            ReadNextBucket( self, bucket + 1 ); // Read next bucket in background
            WaitForFence( self, _yReadFence, bucket );

            // const uint32 entryCount = _context.bucketCounts[(int)lTable][bucket];

            Span<uint32> yInput     = _y[bucket];
            const uint32 entryCount = yInput.Length();
            
            Span<uint32> sortKey = _sortKey.SliceSize( entryCount );
            SortY( self, entryCount, yInput.Ptr(), _yTmp.Ptr(), sortKey.Ptr(), _metaTmp.As<uint32>().Ptr() );

            const Span<Pair> matches = Match( self, bucket, yInput );
//            WritePairs( self );

            // Count the total match count
            uint64 totalMatches = 0;
            self->SyncThreads();
            for( uint32 i = 0; i < threadCount; i++ )
                totalMatches += _pairs[i].Length();
            
            // #TODO: This is only for the last table..
            // if( totalMatches > _maxTableEntries )
            // {
            //     // Prevent calculating fx for overflow matches
            //     if( self->IsLastThread() )
            //     {
            //         const uint32 overflowEntries = (uint32)( totalMatches - _maxTableEntries );
            //         ASSERT( overflowEntries < matches.Length() );

            //         matches = matches.Slice( 0, matches.Length() - overflowEntries );
            //     }

            //     totalMatches = _maxTableEntries;
            // }

            // Generate and write map
            if constexpr ( rTable > TableId::Table2 )
            {
                WaitForFence( self, _indexReadFence, bucket );
            // Span<uint32> indexInput = _index  [bucket];
//            WriteMap( self, indexInput, sortKey ); // Write reverse map, given the previous table's y origin indices
            }

            // Sort meta on Y
            WaitForFence( self, _metaReadFence, bucket );

            Span<TMetaIn> metaTmp = _meta[0].As<TMetaIn>().SliceSize( yInput.Length() );
            Span<TMetaIn> metaIn  = _metaTmp.As<TMetaIn>().SliceSize( yInput.Length() );
            SortOnYKey( self, sortKey, metaTmp, metaIn );

            // Gen fx
            if( self->IsLastThread() )
            Span<uint32>   yOut    = _yTmp;
            Span<TMetaOut> metaOut = metaTmp.As<TMetaOut>().SliceSize( matches.Length() );

            // GenFx( self, yInput, metaIn, matches, yOut, metaOut, bucket );


            // #TODO: Write bucekt
//            WriteEntries( self )
        }
    }

    //-----------------------------------------------------------
    void SortY( Job* self, const uint64 entryCount, uint32* ySrc, uint32* yTmp, uint32* sortKeySrc, uint32* sortKeyTmp )
    {
        constexpr uint32 Radix      = 256;
        constexpr uint32 iterations = 4;
        const     uint32 shiftBase  = 8;

        uint32 shift = 0;

        uint32 counts     [Radix];
        uint32 prefixSum  [Radix];
        uint32 totalCounts[Radix];

        uint64 length, offset, _;
        GetThreadOffsets( self, entryCount, length, offset, _ );

        uint32* input    = ySrc;
        uint32* tmp      = yTmp;
        uint32* keyInput = sortKeySrc;
        uint32* keyTmp   = sortKeyTmp;

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
    }

    //-----------------------------------------------------------
    template<typename T>
    void SortOnYKey( Job* self, const Span<uint32> key, const Span<T> input, Span<T> output )
    {
        uint32 count, offset, end;
        GetThreadOffsets( self, (uint32)key.Length(), count, offset, end );

        for( uint32 i = offset; i < end; i++ )
            output[i] = input[key[i]];

        self->SyncThreads();
    }

    //-----------------------------------------------------------
    Span<Pair> Match( Job* self, const uint32 bucket, Span<uint32> yEntries )
    {
        const uint32 id          = self->JobId();
        const uint32 threadCount = self->JobCount();

        // use metaTmp as a buffer for group boundaries
        Span<uint32> groupBuffer  = _metaTmp.As<uint32>();
        const uint32 maxGroups    = (uint32)( groupBuffer.Length() / threadCount );
        Span<uint32> groupIndices = groupBuffer.Slice( maxGroups * id, maxGroups );

        auto matches = _matcher.Match( self, bucket, yEntries, groupIndices, _pairs[id] );
        _pairs[id] = matches;

        return matches;
    }

    //-----------------------------------------------------------
    void WriteMap( Job* self, const Span<uint32> indexInput, const Span<uint32> sortKey )
    {

    }

    //-----------------------------------------------------------
    void WritePairs( Job* self, const Span<Pair> pairs, const int64 matchCount, const uint32 bucket )
    {
        // #TOOD: Write pairs raw? Or bucket-relative? Maybe raw for now
    }

    //-----------------------------------------------------------
    void WriteEntries( Job* self,
                       const Span<uint64>   yIn,
                       const Span<TMetaIn>  metaIn,
                             Span<uint32>   yOut,
                             Span<TMetaOut> metaOut,
                             Span<uint32>   indices,
                       const uint32         bucket )
    {
        // Distribute to buckets
        const uint32 logBuckets = bblog2( _numBuckets );
        static_assert( kExtraBits <= logBuckets );

        const uint32 bucketShift = _k + kExtraBits - logBuckets;
        const uint64 yMask       = ( 1ull << bucketShift ) - 1;

        const int64 entryCount = (int64)yIn.Length();

        uint32 counts     [_numBuckets] = {};
        uint32 pfxSum     [_numBuckets];
//        uint32 totalCounts[_numBuckets];
        uint32* bucketSlices = _context.bucketSlices[(int)rTable & 1][bucket];

        for( int64 i = 0; i < entryCount; i++ )
            counts[yIn[i] >> bucketShift]++;

        self->CalculatePrefixSum( _numBuckets, counts, pfxSum, bucketSlices );

        // Distribute to buckets
        int64 offset, _;
        GetThreadOffsets( self, (int64)yIn.Length(), _, offset, _ );

        for( int64 i = 0; i < entryCount; i++ )
        {
            const uint64 y      = yIn[i];
            const uint32 dstIdx = --pfxSum[y >> bucketShift];

            yOut   [dstIdx] = (uint32)(y & yMask);
            metaOut[dstIdx] = metaIn[i];
            indices[dstIdx] = (uint32)(offset + i);
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

            _ioQueue.WriteBucketElementsT<uint32>( yId   , yOut.Ptr()   , bucketSlices );
            _ioQueue.WriteBucketElementsT<TMetaOut>( metaId, (TMetaOut*)metaOut.Ptr(), bucketSlices );
            _ioQueue.WriteBucketElementsT<uint32>( idxId , indices.Ptr(), bucketSlices );
            _ioQueue.SignalFence( _fxWriteFence, bucket+1 );
            _ioQueue.CommitCommands();
        }
        self->EndLockBlock();
    }

    //-----------------------------------------------------------
    void GenFx( Job* self,
                const Span<uint32> yIn,
                const Span<Meta4>  metaIn,
                const Span<Pair>   pairs )
    {
        using TYOut    = typename K32TYOut<rTable>::Type;
        using TMetaIn  = typename K32MetaType<rTable>::In;
        using TMetaOut = typename K32MetaType<rTable>::Out;

//        TMetaOut

        const Span<TMetaIn> tableMetaIn ( (TMetaIn*)metaIn.Ptr(), metaIn.Length() );

        // Output buffers
//        Span<TMetaOut> tableMetaOut( (TMetaOut*)metaOut.Ptr(), metaOut.Length() );

//        GenFx<TYOut, TMetaIn, TMetaOut>( self, yIn, tableMetaIn, pairs, )
    }

    //-----------------------------------------------------------
    template<typename TYOut, typename TMetaIn, typename TMetaOut>
    void GenFx( Job* self,
                const Span<uint32>  yIn,
                const Span<TMetaIn> metaIn,
                const Span<Pair>    pairs,
                Span<TYOut>         yOut,
                Span<TMetaOut>      metaOut,
                const uint32        bucket )
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

        const uint32 id         = self->JobId();
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
                const Meta4 l = metaIn[left];
                const Meta4 r = metaIn[right];

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
        if( !self->IsControlThread() )
            return;

        _ioQueue.ReadBucketElementsT( _yId[0], _y[bucket] );
        _ioQueue.SignalFence( _yReadFence, bucket + 1 );

        _ioQueue.ReadBucketElementsT( _idxId[0], _index[bucket] );
        _ioQueue.SignalFence( _indexReadFence, bucket + 1 );

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
    DiskPlotContext& _context;
    DiskBufferQueue& _ioQueue;

    uint32 _entriesPerBucket = 0;

    // I/O
    FileId _yId   [2];
    FileId _idxId [2];
    FileId _metaId[2];

    Span<uint32>     _y    [_numBuckets];
    Span<uint32>     _index[_numBuckets];
    Span<TMetaIn>    _meta [_numBuckets];
    Span<uint64>     _map;
    Span<uint32>     _pairsL;
    Span<uint16>     _pairsR;
    
    // Temp working buffers
    Span<uint32>     _yTmp;
    Span<uint32>     _sortKey;
    Span<Meta4>      _metaTmp;
    Span<Pair>       _pairBuffer;
    Span<Pair>       _pairs[BB_DP_MAX_JOBS];    // Pairs buffer divided per thread

    // Matching
    FxMatcherBounded _matcher;

    // I/O Synchronization fences
    Fence& _yReadFence;
    Fence& _metaReadFence;
    Fence& _indexReadFence;
    Fence& _fxWriteFence;
    Fence& _pairWriteFence;
    Fence& _mapWriteFence;


    // I/O wait accumulator
    Duration _tableIOWait = Duration::zero();
};
