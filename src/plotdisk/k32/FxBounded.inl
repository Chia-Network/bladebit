#pragma once
#include "plotdisk/DiskPlotContext.h"
#include "plotdisk/DiskPlotConfig.h"
#include "plotdisk/DiskBufferQueue.h"
#include "plotdisk/BitBucketWriter.h"
#include "plotdisk/MapWriter.h"
#include "plotdisk/BlockWriter.h"
#include "util/StackAllocator.h"
#include "FpMatchBounded.inl"
#include "b3/blake3.h"

#if _DEBUG
    #include "algorithm/RadixSort.h"
    #include "plotting/PlotValidation.h"
    #include "plotdisk/DiskPlotDebug.h"

    // #define _VALIDATE_Y 1
    #if _VALIDATE_Y
        // uint32       _refBucketCounts[BB_DP_MAX_BUCKET_COUNT];
        // Span<uint64> _yRef;
        // Span<uint64> _yRefWriter;
        
        template<uint32 _numBuckets>
        void DbgValidateY( const TableId table, const FileId fileId, DiskPlotContext& context
            #if BB_DP_FP_MATCH_X_BUCKET
            ,const Span<K32CrossBucketEntries> crossBucketEntriesIn
            #endif
         );
    #endif

    // #define DBG_VALIDATE_INDICES 1
    #if DBG_VALIDATE_INDICES
        static Span<uint32> _dbgIndices;
        static uint64       _dbgIdxCount = 0;
    #endif

#endif


template<TableId rTable, uint32 _numBuckets>
class DiskPlotFxBounded
{
    using TMetaIn  = typename K32MetaType<rTable>::In;
    using TMetaOut = typename K32MetaType<rTable>::Out;
    using TYOut    = typename K32TYOut<rTable>::Type;
    using Job     = AnonPrefixSumJob<uint32>;

    static constexpr uint32 _k               = 32;
    static constexpr uint64 _maxTableEntries = (1ull << _k) - 1;
    static constexpr uint64 _maxSliceEntries = (uint64)((_maxTableEntries / _numBuckets / _numBuckets ) * BB_DP_ENTRY_SLICE_MULTIPLIER);


    static constexpr uint32 _bucketBits      = bblog2( _numBuckets );
    static constexpr uint32 _pairsMaxDelta   = 512;
    static constexpr uint32 _pairsLeftBits   = _k - _bucketBits + 1;    // Buckets may overflow, so need an extra bit
    static constexpr uint32 _pairsRightBits  = bblog2( _pairsMaxDelta );
    static constexpr uint32 _pairBitSize     = _pairsLeftBits + _pairsRightBits;

public:
    //-----------------------------------------------------------
    DiskPlotFxBounded( DiskPlotContext& context )
        : _context       ( context )
        , _ioQueue       ( *context.ioQueue )
        , _matcher       ( _numBuckets )
        , _yReadFence    ( context.fencePool ? context.fencePool->RequireFence() : *(Fence*)nullptr )
        , _metaReadFence ( context.fencePool ? context.fencePool->RequireFence() : *(Fence*)nullptr )
        , _indexReadFence( context.fencePool ? context.fencePool->RequireFence() : *(Fence*)nullptr )
        , _fxWriteFence  ( context.fencePool ? context.fencePool->RequireFence() : *(Fence*)nullptr )
        , _pairWriteFence( context.fencePool ? context.fencePool->RequireFence() : *(Fence*)nullptr )
        , _mapWriteFence ( context.fencePool ? context.fencePool->RequireFence() : *(Fence*)nullptr )
    #if BB_DP_FP_MATCH_X_BUCKET

    #endif
    {
        const TableId lTable = rTable - 1;

        _yId   [0] = FileId::FX0    + (FileId)((int)lTable & 1);
        _idxId [0] = FileId::INDEX0 + (FileId)((int)lTable & 1);
        _metaId[0] = FileId::META0  + (FileId)((int)lTable & 1);

        _yId   [1] = FileId::FX0    + (FileId)((int)rTable & 1);
        _idxId [1] = FileId::INDEX0 + (FileId)((int)rTable & 1);
        _metaId[1] = FileId::META0  + (FileId)((int)rTable & 1);

        if( context.cfg && context.cfg->alternateBuckets )
        {
            _interleaved = ((uint)rTable & 1) == 0; // Only even tables have interleaved writes when alternating mode is enabled
    
            _yId   [0] = FileId::FX0;
            _idxId [0] = FileId::INDEX0;
            _metaId[0] = FileId::META0;
            _yId   [1] = FileId::FX0;
            _idxId [1] = FileId::INDEX0;
            _metaId[1] = FileId::META0;
        }
    }

    //-----------------------------------------------------------
    ~DiskPlotFxBounded()
    {
    }

    //-----------------------------------------------------------
    static void GetRequiredHeapSize( IAllocator& allocator, const size_t t1BlockSize, const size_t t2BlockSize, const uint32 threadCount )
    {
        DiskPlotContext cx = {};

        DiskPlotFxBounded<TableId::Table4, _numBuckets> instance( cx );
        instance.AllocateBuffers( allocator, t1BlockSize, t2BlockSize, threadCount, true );
    }

    //-----------------------------------------------------------
    void AllocateBuffers( IAllocator& allocator, const size_t t1BlockSize, const size_t t2BlockSize, const uint32 threadCount, const bool dryRun )
    {
        const uint64 kEntryCount            = 1ull << _k;
        const uint64 yEntriesPerBlock       = t2BlockSize / sizeof( uint32 );
        const uint64 entriesPerBucket       = (uint64)( kEntryCount / _numBuckets * BB_DP_XTRA_ENTRIES_PER_BUCKET );
        const uint64 entriesPerSlice        = entriesPerBucket / _numBuckets;

        // #TODO: Do these for meta as well, since this is for y
        const uint64 entriesPerSliceAligned = RoundUpToNextBoundaryT( entriesPerSlice, yEntriesPerBlock ) + yEntriesPerBlock;
        const uint64 writeEntriesPerBucket  = entriesPerSliceAligned * _numBuckets;


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
        _metaTmp[0] = allocator.CAllocSpan<K32Meta4>( entriesPerBucket, t1BlockSize ); // Align this to T1 as we use it to write x to disk.
        _metaTmp[1] = allocator.CAllocSpan<K32Meta4>( entriesPerBucket, t1BlockSize );
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
        _yWriteBuffer     = allocator.CAllocSpan<uint32  >( writeEntriesPerBucket, t2BlockSize );
        _indexWriteBuffer = allocator.CAllocSpan<uint32  >( writeEntriesPerBucket, t2BlockSize );
        _metaWriteBuffer  = allocator.CAllocSpan<TMetaOut>( writeEntriesPerBucket, t2BlockSize );

        const size_t pairsWriteSize = CDiv( entriesPerBucket * _pairBitSize, 8 );
        _pairsWriteBuffer = allocator.AllocT<byte>( pairsWriteSize, t1BlockSize );
        byte* pairBlocks  = (byte*)allocator.CAlloc( 1, t1BlockSize, t1BlockSize );

        if( !dryRun )
            _pairBitWriter = BitBucketWriter<1>( _ioQueue, FileId::T1 + (FileId)rTable, pairBlocks );
        
        if constexpr ( rTable == TableId::Table2 )
        {
            _xWriter = BlockWriter<uint32>( allocator, FileId::T1, _mapWriteFence, t1BlockSize, entriesPerBucket );
        }
        else
        {
            _mapWriteBuffer = allocator.CAllocSpan<uint64>  ( entriesPerBucket, t1BlockSize );
            ASSERT( (uintptr_t)_mapWriteBuffer.Ptr() / t1BlockSize * t1BlockSize == (uintptr_t)_mapWriteBuffer.Ptr() );

            if( !dryRun )
            {
                _mapWriter = MapWriter<_numBuckets, false>( _ioQueue, FileId::MAP2 + (FileId)(rTable-2), allocator, 
                                                            entriesPerBucket, t1BlockSize, _mapWriteFence, _tableIOWait );
            }
            else
            {
                _mapWriter = MapWriter<_numBuckets, false>( entriesPerBucket, allocator, t1BlockSize );
            }
        }
        
        // _pairsL = allocator.CAllocSpan<uint32>( entriesPerBucket );
        // _pairsR = allocator.CAllocSpan<uint16>( entriesPerBucket );

        #if BB_DP_FP_MATCH_X_BUCKET
        {
            const size_t crossBucketPairAllocSize = (size_t)_pairBitSize * BB_DP_CROSS_BUCKET_MAX_ENTRIES;
            _crossBucketWriteBuffer = allocator.AllocT<byte>( crossBucketPairAllocSize, t1BlockSize );
        }
        #endif
    }

    // Generate fx for a whole table
    //-----------------------------------------------------------
    void Run( IAllocator& allocator
        #if BB_DP_FP_MATCH_X_BUCKET
            , Span<K32CrossBucketEntries> crossBucketEntriesIn
            , Span<K32CrossBucketEntries> crossBucketEntriesOut
        #endif
    )
    {
        #if BB_DP_FP_MATCH_X_BUCKET
            _crossBucketEntriesIn  = crossBucketEntriesIn;
            _crossBucketEntriesOut = crossBucketEntriesOut;
        #endif

        #if DBG_VALIDATE_TABLES
            _dbgPlot.AllocTable( rTable );
        #endif

        #if (_DEBUG && DBG_VALIDATE_INDICES)
            if( _dbgIndices.Ptr() == nullptr )
                _dbgIndices = bbcvirtallocboundednuma_span<uint32>( _maxTableEntries );
            _dbgIdxCount = 0;
        #endif

        // Prepare input files
        _ioQueue.SeekBucket( _yId[0], 0, SeekOrigin::Begin );
        _ioQueue.SeekBucket( _yId[1], 0, SeekOrigin::Begin );

        if( rTable > TableId::Table2 )
        {
            _ioQueue.SeekBucket( _idxId[0], 0, SeekOrigin::Begin );    
            _ioQueue.SeekBucket( _idxId[1], 0, SeekOrigin::Begin );    
        }

        _ioQueue.SeekBucket( _metaId[0], 0, SeekOrigin::Begin );
        if( rTable < TableId::Table7 )
            _ioQueue.SeekBucket( _metaId[1], 0, SeekOrigin::Begin );

        _ioQueue.CommitCommands();
        
        // Allocate buffers
        AllocateBuffers( allocator, _context.tmp1BlockSize, _context.tmp2BlockSize,  _context.fpThreadCount, false );

        // Init buffers
        const uint32 threadCount = _context.fpThreadCount;

        for( uint32 i = 0; i < _numBuckets; i++ )
        {
            const uint32 bufIdx = i & 1; // % 2
            _y    [i] = _yBuffers    [bufIdx];
            _index[i] = _indexBuffers[bufIdx];
            _meta [i] = _metaBuffers [bufIdx].template As<TMetaIn>();
        }
        

        // Set some initial fence status
        _mapWriteFence.Signal( 0 );

        // Run mmulti-threaded
        Job::Run( *_context.threadPool, threadCount, [=]( Job* self ) {
            RunMT( self );
        });

        _context.entryCounts[(int)rTable] = _tableEntryCount;
        
        #if _DEBUG
        {
            uint64 tableEntryCount = 0;
            for( uint32 i = 0; i < _numBuckets; i++ )
                tableEntryCount += _context.bucketCounts[(int)rTable][i];

            ASSERT( tableEntryCount == _tableEntryCount );

            #if DBG_VALIDATE_INDICES
            if constexpr ( rTable > TableId::Table2 )
            {
                Log::Line( "[DEBUG: Validating indices]" );
                ASSERT( _dbgIdxCount );

                uint32* tmpIndices = bbcvirtallocbounded<uint32>( _maxTableEntries );
                RadixSort256::Sort<BB_DP_MAX_JOBS>( *_context.threadPool, _dbgIndices.Ptr(), tmpIndices, _dbgIdxCount );
                bbvirtfreebounded( tmpIndices );

                for( uint64 i = 0; i < _dbgIdxCount; i++ )
                {
                    ASSERT( _dbgIndices[i] == i);
                }

                _dbgIdxCount = 0;
                Log::Line( " OK" );
            }
            #endif
        }
        #endif

        Log::Line( " Sorting      : Completed in %.2lf seconds.", TicksToSeconds( _sortTime  ) );
        Log::Line( " Distribution : Completed in %.2lf seconds.", TicksToSeconds( _distributeTime ) );
        Log::Line( " Matching     : Completed in %.2lf seconds.", TicksToSeconds( _matchTime ) );
        Log::Line( " Fx           : Completed in %.2lf seconds.", TicksToSeconds( _fxTime    ) );

        // Ensure all I/O has completed
        _fxWriteFence.Wait( _numBuckets, _tableIOWait );
        _context.fencePool->RestoreAllFences();

        #if DBG_VALIDATE_TABLES
            _dbgPlot.FinishTable( rTable );
        #endif

        #if _DEBUG
            // ValidateIndices();
        
            #if _VALIDATE_Y
                DbgValidateY<_numBuckets>( rTable, _yId[1], _context
                    #if BB_DP_FP_MATCH_X_BUCKET
                    , _crossBucketEntriesOut
                    #endif
                );
            #endif

            #if BB_DP_DBG_VALIDATE_BOUNDED_PAIRS
                if( rTable > TableId::Table2 )
                    BB_DBG_ValidateBoundedPairs( _numBuckets, rTable-1, _context );
                if( rTable == TableId::Table7 )
            BB_DBG_ValidateBoundedPairs( _numBuckets, rTable, _context );
            #endif
        #endif // _DEBUG
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
            ReadNextBucket( self, bucket + 1 ); // Read next bucket in background
            WaitForFence( self, _yReadFence, bucket );

            Span<uint32> yInput     = _y[bucket];
            const uint32 entryCount = (uint32)yInput.Length();

            Span<uint32> sortKey = _sortKey.SliceSize( entryCount );
            SortY( self, entryCount, yInput.Ptr(), _yTmp.template As<uint32>().Ptr(), sortKey.Ptr(), _metaTmp[1].template As<uint32>().Ptr() );

            ///
            /// Write reverse map, given the previous table's y origin indices
            ///
            if constexpr ( rTable > TableId::Table2 )
            {
                WaitForFence( self, _indexReadFence, bucket );

                Span<uint32> unsortedIndices = _index[bucket];       ASSERT( unsortedIndices.Length() == entryCount );
                Span<uint32> indices         = _metaTmp[1].template As<uint32>().SliceSize( entryCount );

                #if (_DEBUG && DBG_VALIDATE_INDICES)
                    if( self->BeginLockBlock() )
                    {
                        // indices.CopyTo( _dbgIndices.Slice( _dbgIdxCount, entryCount ) );
                        unsortedIndices.CopyTo( _dbgIndices.Slice( _dbgIdxCount, entryCount ) );
                        
                        _dbgIdxCount += entryCount;
                    }
                    self->EndLockBlock();
                #endif

                SortOnYKey( self, sortKey, unsortedIndices, indices );
                WriteMap( self, bucket, indices, _mapWriteBuffer, _mapOffset );
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
            ASSERT( matchOffset < totalMatches );

            // #TEST
            #if _DEBUG
            //     if( self->IsControlThread() )
            //         Log::Line( " [%3u] : %u", bucket, totalMatches );
            #endif

            // Prevent overflow entries
            const uint64 tableEntryCount = _tableEntryCount;
            if( bucket == _numBuckets-1 && (tableEntryCount + totalMatches) > _maxTableEntries )
            {
                const uint32 overflowEntries = (uint32)( (tableEntryCount + totalMatches) - _maxTableEntries );

                // Prevent calculating fx for overflow matches
                if( self->IsLastThread() )
                {
                    ASSERT( overflowEntries < matches.Length() );

                    matches = matches.SliceSize( matches.Length() - overflowEntries );
                }

                totalMatches -= overflowEntries;
                ASSERT( tableEntryCount + totalMatches == _maxTableEntries );
            }

            WritePairs( self, bucket, totalMatches, matches, matchOffset );

            ///
            /// Sort meta on Y
            ///
            WaitForFence( self, _metaReadFence, bucket );
    
            Span<TMetaIn> metaUnsorted = _meta[bucket].SliceSize( entryCount );                 ASSERT( metaUnsorted.Length() == entryCount );
            Span<TMetaIn> metaIn       = _metaTmp[0].template As<TMetaIn>().SliceSize( entryCount );

            if constexpr ( rTable == TableId::Table2 )
            {
                // #TODO: Simplify this, allowing BlockWriter to let the user specify a buffer, like in BitWriter
                // Get and set shared x buffer for other threads
                if( self->BeginLockBlock() )
                    _xWriteBuffer = Span<TMetaIn>( _xWriter.GetNextBuffer( _tableIOWait ), entryCount );
                self->EndLockBlock();

                // Grap shared buffer
                metaIn = _xWriteBuffer;
            }

            SortOnYKey( self, sortKey, metaUnsorted, metaIn );

            #if BB_DP_FP_MATCH_X_BUCKET
                SaveCrossBucketMetadata( self, bucket, metaIn );
            #endif
            
            // On Table 2, metadata is our x values, which have to be saved as table 1
            if constexpr ( rTable == TableId::Table2 )
            {
                // Write (sorted-on-y) x back to disk
                if( self->BeginLockBlock() )
                {
                    #if DBG_VALIDATE_TABLES
                        _dbgPlot.WriteYX( bucket, yInput, metaIn );
                    #endif

                    _xWriter.SubmitBuffer( _ioQueue, entryCount );
                    if( bucket == _numBuckets - 1 )
                        _xWriter.SubmitFinalBlock( _ioQueue );
                }
                self->EndLockBlock();
            }

            /// Gen fx & write
            {
                Span<TYOut>    yOut    = _yTmp.template As<TYOut>().Slice( matchOffset, matches.Length() );
                Span<TMetaOut> metaOut = _metaTmp[1].template As<TMetaOut>().Slice( matchOffset, matches.Length() );  //( (TMetaOut*)metaTmp.Ptr(), matches.Length() );

                TimePoint timer;
                if( self->IsControlThread() )
                    timer = TimerBegin();

                // Generate fx for cross-bucket matches, and save the matches to an in-memory buffer
                #if BB_DP_FP_MATCH_X_BUCKET
                    if( bucket > 0 )
                        GenCrossBucketFx( self, bucket-1 );
                #endif

                GenFx( self, bucket, matches, yInput, metaIn, yOut, metaOut );
                self->SyncThreads();

                if( self->IsControlThread() )
                    _fxTime += TimerEndTicks( timer );

                #if _VALIDATE_Y
                // if( self->IsControlThread() )
                // {
                //     if( _yRef.Ptr() == nullptr )
                //     {
                //         _yRef = Span<uint64>( bbcvirtallocboundednuma<uint64>( 1ull << 32 ), 1ull << 32 );
                //         _yRefWriter = _yRef;
                //     }

                //     _yTmp.SliceSize( totalMatches ).CopyTo( _yRefWriter );
                //     _yRefWriter = _yRefWriter.Slice( totalMatches );
                // }
                #endif

                WriteEntries( self, bucket, (uint32)_tableEntryCount + matchOffset, yOut, metaOut, _yWriteBuffer, _metaWriteBuffer, _indexWriteBuffer );
            }

            if( self->IsControlThread() )
            {
                _tableEntryCount += totalMatches;
                _mapOffset       += entryCount;

                // Save bucket length before y-sort since the pairs remain unsorted
                _context.ptrTableBucketCounts[(int)rTable][bucket] = (uint32)totalMatches;

                #if BB_DP_FP_MATCH_X_BUCKET
                if( bucket > 0 )
                    _context.ptrTableBucketCounts[(int)rTable][bucket-1] += _matcher.GetCrossBucketInfo( bucket - 1 ).matchCount;
                #endif
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
        ASSERT( key.Length() == input.Length() && input.Length() == output.Length() );

        TimePoint timer;
        if( self->IsControlThread() )
            timer = TimerBegin();

        uint32 count, offset, end;
        GetThreadOffsets( self, (uint32)key.Length(), count, offset, end );

        for( uint32 i = offset; i < end; i++ )
            output[i] = input[key[i]];

        if( self->BeginLockBlock() )
            _sortTime += TimerEndTicks( timer );
        self->EndLockBlock();
    }

    //-----------------------------------------------------------
    Span<Pair> Match( Job* self, const uint32 bucket, Span<uint32> yEntries )
    {
        const uint32 id          = self->JobId();
        const uint32 threadCount = self->JobCount();

        // use metaTmp as a buffer for group boundaries
        Span<uint32> groupBuffer  = _metaTmp[1].template As<uint32>();
        const uint32 maxGroups    = (uint32)( groupBuffer.Length() / threadCount );
        Span<uint32> groupIndices = groupBuffer.Slice( maxGroups * id, maxGroups );

        TimePoint timer;
        if( self->IsControlThread() )
            timer = TimerBegin();

        // Scan for groups & perform cross-bucket matches for the previous bucket
        // _matcher.ScanGroupsAndMatchPreviousBucket( self, bucket, yEntries, groupIndices );

        // // Match this bucket's matches
        // _matcher.Match( self, bucket, yEntries, _pairs[id] );

        const uint32 maxMatchesPerThread = _entriesPerBucket / threadCount;
        _pairs[id] = _pairBuffer.Slice( maxMatchesPerThread * id, maxMatchesPerThread );
   
        auto matches = _matcher.Match( self, bucket, yEntries, groupIndices, _pairs[id] );
        _pairs[id] = matches;

        if( self->BeginLockBlock() )
        {
            _matchTime += TimerEndTicks( timer );
        }
        self->EndLockBlock();

        return matches;
    }
    
#if BB_DP_FP_MATCH_X_BUCKET
    //-----------------------------------------------------------
    void SaveCrossBucketMetadata( Job* self, const uint32 bucket, const Span<TMetaIn> metaIn )
    {
        if( self->IsControlThread() )
            _matcher.SaveCrossBucketMeta( bucket, metaIn );
    }

    //-----------------------------------------------------------
    void WriteCrossBucketPairs( Job* self, const uint32 bucket )
    {
        ASSERT( bucket < _numBuckets );
        ASSERT( self->JobId() == 0 );       // Only to be called by the control thread

        auto& info = _matcher.GetCrossBucketInfo( bucket );
        if( info.matchCount < 1 )
            return;
        
        const Span<Pair> pairs( info.pair, info.matchCount );
        const uint32     offset = info.matchOffset;
        
        // Apply pair offset
        for( uint32 i = 0; i < info.matchCount; i++ )
            pairs[i].AddOffset( offset );

        // Bit-serialize
        uint64 bitBucketSizes = pairs.Length() * _pairBitSize;
        _pairBitWriter.BeginWriteBuckets( &bitBucketSizes, _crossBucketWriteBuffer );

        BitWriter writer = _pairBitWriter.GetWriter( 0, 0 );
        PackPairs( self, pairs, writer );

        // #TODO: Keep a different buffer? But for now, we have to remove the offsets, as we
        //        still have to perform cross-bucket fx with them
        for( uint32 i = 0; i < info.matchCount; i++ )
            pairs[i].SubstractOffset( offset );

        // Submit to disk
        _pairBitWriter.Submit();
    }

    //-----------------------------------------------------------
    void GenCrossBucketFx( Job* self, const uint32 bucket )
    {
        ASSERT( bucket < _numBuckets );

        auto& info = _matcher.GetCrossBucketInfo( bucket );

        if( info.matchCount < 1 )
            return;
        
        if( self->BeginLockBlock() )
        {
            const uint32 matchCount       = info.matchCount;
            const uint32 prevBucketLength = info.PrevBucketEntryCount();

            const Span<Pair>    pairs( info.pair               , matchCount );
            const Span<uint32>  y    ( info.savedY             , prevBucketLength );
            const Span<TMetaIn> meta ( (TMetaIn*)info.savedMeta, prevBucketLength + info.curBucketMetaLength );

            Span<TYOut>    yOut    = _yTmp      .template As<TYOut>()   .SliceSize( matchCount );
            Span<TMetaOut> metaOut = _metaTmp[1].template As<TMetaOut>().SliceSize( matchCount );

            GenFx( self, bucket, pairs, y, meta, yOut, metaOut );

            // Distribute new values to in-memory cross-bucket cache
            {
                const uint32 bucketBits = bblog2( _numBuckets );
                static_assert( kExtraBits <= bucketBits );

                const uint32 yBits       = ( std::is_same<TYOut, uint32>::value ? _k : _k + kExtraBits );
                const uint32 bucketShift = yBits - bucketBits;
                const TYOut  yMask       = std::is_same<TYOut, uint32>::value ? 0xFFFFFFFF : ( 1ull << bucketShift ) - 1; // No masking-out for Table 7
              
                const uint32 indxeBase = _tableEntryCount.load( std::memory_order_acquire );
                _tableEntryCount.store( indxeBase + matchCount, std::memory_order_release );

                auto& crossBucketEntries = _crossBucketEntriesOut;

                for( uint32 i = 0; i < matchCount; i++ )
                {
                    const TYOut  y       = yOut[i];
                    const uint32 yBucket = (uint32)(y >> bucketShift);

                    auto& bucketFile          = crossBucketEntries[yBucket];
                    const uint32 bucketLength = bucketFile.length;

                    Span<TMetaOut> metaDst( (TMetaOut*)bucketFile.meta, BB_DP_CROSS_BUCKET_MAX_ENTRIES );

                    bucketFile.y    [bucketLength] = (uint32)(y & yMask);
                    bucketFile.index[bucketLength] = indxeBase + i;
                    metaDst         [bucketLength] = metaOut[i];

                    bucketFile.length = bucketLength + 1;

                    _context.bucketCounts[(int)rTable][yBucket]++;
                }
            }
        }
        self->EndLockBlock();
    }

#endif // BB_DP_FP_MATCH_X_BUCKET

    //-----------------------------------------------------------
    void WritePairs( Job* self, const uint32 bucket, const uint32 totalMatchCount, 
                     const Span<Pair> matches, const uint64 dstOffset )
    {
        ASSERT( dstOffset + matches.length <= totalMatchCount );
        
        if( self->BeginLockBlock() )
        {
            // Wait for our write buffer to be ready to use again
            if( bucket > 0 )
            {
                _pairWriteFence.Wait( bucket );

                #if BB_DP_FP_MATCH_X_BUCKET
                    WriteCrossBucketPairs( self, bucket - 1 );
                #endif
            }

            // Ready the bit-writer for serialization
            uint64 bitBucketSizes = (uint64)totalMatchCount * _pairBitSize;
            _pairBitWriter.BeginWriteBuckets( &bitBucketSizes, _pairsWriteBuffer );
        }
        self->EndLockBlock();

        // #NOTE: For now we write compressed as in standard FP.        
        BitWriter writer = _pairBitWriter.GetWriter( 0, dstOffset * _pairBitSize );

        ASSERT( matches.Length() > 2 );
        PackPairs( self, matches.SliceSize( 2 ), writer );
        self->SyncThreads();
        PackPairs( self, matches.Slice( 2 ), writer );

        // Write to disk
        if( self->BeginLockBlock() )
        {
            _pairBitWriter.Submit();

            if( bucket == _numBuckets-1 )
                _pairBitWriter.SubmitLeftOvers();

            _ioQueue.SignalFence( _pairWriteFence, bucket + 1 );
            _ioQueue.CommitCommands();
        }
        self->EndLockBlock();
    }

    //-----------------------------------------------------------
    void PackPairs( Job* self, const Span<Pair> pairs, BitWriter& writer )
    {
        self;

        const uint32 shift = _pairsLeftBits;
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
    void WriteMap( Job* self, const uint32 bucket, const Span<uint32> bucketIndices, Span<uint64> mapOut, const uint32 tableOffset )
    {
        uint64 outMapBucketCounts[_numBuckets];     // Pass actual bucket counts and implement them
        uint32 totalCounts       [_numBuckets];
        // uint64 totalBitCounts    [_numBuckets];

        _mapWriter.WriteJob( self, bucket, tableOffset, bucketIndices, mapOut,
            outMapBucketCounts, totalCounts, _mapBitCounts );

        if( bucket == _numBuckets-1 && self->IsControlThread() )
            _mapWriter.SubmitFinalBits();
        ///
        /// #TODO: Fix and re-enabled Un-compressed method?:
        ///
        // const uint32 bucketShift = _k - _bucketBits;
        // const uint32 blockSize   = (uint32)_ioQueue.BlockSize( FileId::MAP2 );

        // uint32 counts [_numBuckets] = { 0 };
        // uint32 pfxSum [_numBuckets];
        // uint32 offsets[_numBuckets];

        // uint32 count, offset, _;
        // GetThreadOffsets( self, (uint32)bucketIndices.Length(), count, offset, _ );

        // auto indices = bucketIndices.Slice( offset, count );

        // // Count buckets
        // for( size_t i = 0; i < indices.Length(); i++ )
        // {
        //     const uint32 b = indices[i] >> bucketShift; ASSERT( b < _numBuckets );
        //     counts[b]++;
        // }

        // uint32* pSliceCounts   = nullptr;
        // uint32* pAlignedCounts = nullptr;
        // uint32* pOffsets       = offsets;

        // if( self->IsControlThread() )
        // {
        //     pSliceCounts   = _mapSliceCounts;
        //     pAlignedCounts = _mapAlignedCounts;
        //     pOffsets       = _mapOffsets;
        // }
        // else
        //     memcpy( offsets, _mapOffsets, sizeof( offsets ) );

        // // self->CalculateBlockAlignedPrefixSum<uint64>( _numBuckets, blockSize, counts, pfxSum, pSliceCounts, pOffsets, pAlignedCounts );
        // self->CalculatePrefixSum( _numBuckets, counts, pfxSum, totalCounts );

        // // Wait for write fence
        // if( self->BeginLockBlock() )
        //     _mapWriteFence.Wait();
        // self->EndLockBlock();

        // // Distribute as 64-bit entries
        // const uint64 outOffset = tableOffset + offset;

        // for( size_t i = 0; i < indices.Length(); i++ )
        // {
        //     const uint64 origin = indices[i];
        //     const uint32 b      = origin >> bucketShift;    ASSERT( b < _numBuckets );

        //     const uint32 dstIdx = --pfxSum[b]; ASSERT( dstIdx < mapOut.Length() );

        //     mapOut[dstIdx] = (origin << _k) | (outOffset + i);  // (origin, dst index)
        // }

        // // Write map to disk
        // if( self->BeginLockBlock() )
        // {
        //     _ioQueue.WriteBucketElementsT<uint64>( FileId::MAP2 + (FileId)rTable-2, mapOut.Ptr(), _mapAlignedCounts, _mapSliceCounts );
        //     _ioQueue.SignalFence( _mapWriteFence );
        //     _ioQueue.CommitCommands();
        // }
        // self->EndLockBlock();
    }

    //-----------------------------------------------------------
    void WriteEntries( Job* self,
                       const uint32         bucket,
                       const uint32         idxOffset,
                       const Span<TYOut>    yIn,
                       const Span<TMetaOut> metaIn,
                             Span<uint32>   yOut,
                             Span<TMetaOut> metaOut,
                             Span<uint32>   idxOut )
    {
        TimePoint timer;
        if( self->IsControlThread() )
            timer = TimerBegin();

        const uint32 blockSize   = (uint32)_ioQueue.BlockSize( _yId[1] );
        const uint32 id          = (uint32)self->JobId();

        // Distribute to buckets
        const uint32 bucketBits = bblog2( _numBuckets );
        static_assert( kExtraBits <= bucketBits );

        const uint32 yBits       = ( std::is_same<TYOut, uint32>::value ? _k : _k + kExtraBits );
        const uint32 bucketShift = yBits - bucketBits;
        const TYOut  yMask       = std::is_same<TYOut, uint32>::value ? 0xFFFFFFFF : ( 1ull << bucketShift ) - 1; // No masking-out for Table 7

        const int64 entryCount = (int64)yIn.Length();

        uint32 counts    [_numBuckets] = {};
        uint32 pfxSum    [_numBuckets];
        uint32 pfxSumMeta[_numBuckets];

        const uint32 sliceIdx = bucket & 1; // % 2

        Span<uint32> ySliceCounts          = _sliceCountY[sliceIdx];
        Span<uint32> metaSliceCounts       = _sliceCountMeta[sliceIdx];
        Span<uint32> yAlignedSliceCount    = _alignedSliceCountY[sliceIdx];
        Span<uint32> metaAlignedSliceCount = _alignedsliceCountMeta[sliceIdx];


        // Count
        for( int64 i = 0; i < entryCount; i++ )
            counts[yIn[i] >> bucketShift]++;
    
        self->CalculateBlockAlignedPrefixSum<uint32>( _numBuckets, blockSize, counts, pfxSum, ySliceCounts.Ptr(), _offsetsY[id], yAlignedSliceCount.Ptr() );

        if ( rTable < TableId::Table7 )
            self->CalculateBlockAlignedPrefixSum<TMetaOut>( _numBuckets, blockSize, counts, pfxSumMeta, metaSliceCounts.Ptr(), _offsetsMeta[id], metaAlignedSliceCount.Ptr() );

        if( bucket > 0 )
        {
            if( self->BeginLockBlock() )
                _fxWriteFence.Wait( bucket, _tableIOWait );     // #TODO: Double-buffer to avoid waiting here // #TODO: Either use a spin wait or have all threads suspend here
            self->EndLockBlock();
        }

        // Distribute to buckets
        for( int64 i = 0; i < entryCount; i++ )
        {
            const TYOut  y       = yIn[i];
            const uint32 yBucket = (uint32)(y >> bucketShift);
            const uint32 yDst    = --pfxSum    [yBucket];

            yOut  [yDst] = (uint32)(y & yMask);
            idxOut[yDst] = idxOffset + (uint32)i;   ASSERT( (uint64)idxOffset + (uint64)i < (1ull << _k) );

            if constexpr ( rTable < TableId::Table7 )
            {
                const uint32 metaDst = --pfxSumMeta[yBucket];
                metaOut[metaDst] = metaIn[i];
            }
        }

        // Write to disk
        if( self->BeginLockBlock() )
        {
            const FileId yId    = _yId   [1];
            const FileId metaId = _metaId[1];
            const FileId idxId  = _idxId [1];

            ASSERT( ySliceCounts.Length() == metaSliceCounts.Length() );

            #if _DEBUG
            {
                const uint64 maxSliceEntries = _maxSliceEntries;
                for( uint32 i = 0; i < _numBuckets; i++ )
                {
                    ASSERT( yAlignedSliceCount[i] <= maxSliceEntries );
                    ASSERT( metaAlignedSliceCount[i] <= maxSliceEntries );
                } 
            }
            #endif

            _ioQueue.WriteBucketElementsT<uint32>( yId  , _interleaved, yOut   .Ptr(),  yAlignedSliceCount.Ptr(), ySliceCounts.Ptr() );
            _ioQueue.WriteBucketElementsT<uint32>( idxId, _interleaved, idxOut .Ptr(),  yAlignedSliceCount.Ptr(), ySliceCounts.Ptr() );
                
            if( rTable < TableId::Table7 )
                _ioQueue.WriteBucketElementsT<TMetaOut>( metaId, _interleaved, metaOut.Ptr(),  metaAlignedSliceCount.Ptr(), ySliceCounts.Ptr() );

            _ioQueue.SignalFence( _fxWriteFence, bucket+1 );
            _ioQueue.CommitCommands();

            // Save bucket counts
            for( uint32 i = 0; i < _numBuckets; i++ )
                _context.bucketCounts[(int)rTable][i] += ySliceCounts[i];
        }
        self->EndLockBlock();

        if( self->IsControlThread() )
            _distributeTime += TimerEndTicks( timer );
    }

    //-----------------------------------------------------------
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
        
        const uint32 bucketBits = bblog2( _numBuckets );
        const uint32 yBits      = k + kExtraBits - bucketBits;
        const uint64 yMask      = ((uint64)bucket) << yBits;

        const size_t metaSize    = k * MetaInMulti;
        const size_t metaSizeLR  = metaSize * 2;

        const size_t bufferSize  = CDiv( ySize + metaSizeLR, 8 );

        // const uint32 id         = self->JobId();
        const uint32 matchCount = (uint32)pairs.Length();

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
                const uint64 l0 = metaIn[left ].m0;
                const uint64 l1 = metaIn[left ].m1 & 0xFFFFFFFF;
                const uint64 r0 = metaIn[right].m0;
                const uint64 r1 = metaIn[right].m1 & 0xFFFFFFFF;
            
                input[0] = Swap64( y  << 26 | l0 >> 38 );
                input[1] = Swap64( l0 << 26 | l1 >> 6  );
                input[2] = Swap64( l1 << 58 | r0 >> 6  );
                input[3] = Swap64( r0 << 58 | r1 << 26 );
            }
            else if constexpr( MetaInMulti == 4 )
            {
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

                mOut.m0 = h0 << ySize | h1 >> 26;
                mOut.m1 = ((h1 << 6) & 0xFFFFFFC0) | h2 >> 58;
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

        const bool interleaved = !_interleaved; // If the rTable is interleaved, then the L table is not interleaved and vice-versa.

        _ioQueue.ReadBucketElementsT( _yId[0], interleaved, _y[bucket] );
        _ioQueue.SignalFence( _yReadFence, bucket + 1 );

        if constexpr ( rTable > TableId::Table2 )
        {
            _ioQueue.ReadBucketElementsT( _idxId[0], interleaved, _index[bucket] );
            _ioQueue.SignalFence( _indexReadFence, bucket + 1 );
        }

        _ioQueue.ReadBucketElementsT<TMetaIn>( _metaId[0], interleaved, _meta[bucket] );
        _ioQueue.SignalFence( _metaReadFence, bucket + 1 );

        _ioQueue.CommitCommands();
    }

    //-----------------------------------------------------------
    void WaitForFence( Job* self, Fence& fence, const uint32 bucket )
    {
        if( self->BeginLockBlock() )
        {
            fence.Wait( bucket+1, _tableIOWait );

            #if BB_DP_FP_MATCH_X_BUCKET
                // Add cross-bucket entries saved in the memory cache
                if( &fence == &_yReadFence )
                    _y[bucket] = _crossBucketEntriesIn[bucket].CopyYAndExtend( _y[bucket] );
                else if( &fence == &_indexReadFence )
                    _index[bucket] = _crossBucketEntriesIn[bucket].CopyIndexAndExtend( _index[bucket] );
                else if( &fence == &_metaReadFence )
                    _meta[bucket] = _crossBucketEntriesIn[bucket].CopyMetaAndExtend( _meta[bucket] );
            #endif
        }

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

        const bool interleaved = ((uint32)rTable & 1) == 0;

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

                _ioQueue.ReadBucketElementsT<uint32>( fileId, interleaved, reader );
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
    Span<uint32>        _yBuffers    [2];
    Span<uint32>        _indexBuffers[2];
    Span<K32Meta4>      _metaBuffers [2];

    // Read views
    Span<uint32>        _y    [_numBuckets];
    Span<uint32>        _index[_numBuckets];
    Span<TMetaIn>       _meta [_numBuckets];

    // Write buffers
    Span<uint32>        _yWriteBuffer;
    Span<uint32>        _indexWriteBuffer;
    Span<TMetaOut>      _metaWriteBuffer;
    Span<uint64>        _mapWriteBuffer;
    byte*               _pairsWriteBuffer;
    BlockWriter<uint32> _xWriter;               // Used for Table2  (there's no map, but just x.)
    Span<uint32>        _xWriteBuffer;          // Set by the control thread for other threads to use

    // Writers for when using alternating mode, for non-interleaved writes
    bool                _interleaved = true;
    
    // Working buffers
    Span<uint64>        _yTmp;
    Span<uint32>        _sortKey;
    Span<K32Meta4>      _metaTmp[2];
    Span<Pair>          _pairBuffer;

    // Working views
    Span<Pair>          _pairs[BB_DP_MAX_JOBS];    // Pairs buffer divided per thread

    // Matching
    FxMatcherBounded _matcher;

    // Map
    MapWriter<_numBuckets, false> _mapWriter;
    uint64                        _mapBitCounts[_numBuckets];    // Used by the map writer. Single instace shared accross jobs

    // Distributing to buckets
    Span<uint32*> _offsetsY;
    Span<uint32*> _offsetsMeta;
    Span<uint32>  _sliceCountY   [2];
    Span<uint32>  _sliceCountMeta[2];
    Span<uint32>  _alignedSliceCountY   [2];
    Span<uint32>  _alignedsliceCountMeta[2];

    // I/O Synchronization fences
    Fence& _yReadFence;
    Fence& _metaReadFence;
    Fence& _indexReadFence;
    Fence& _fxWriteFence;
    Fence& _pairWriteFence;
    Fence& _mapWriteFence;

    #if BB_DP_FP_MATCH_X_BUCKET
        Span<K32CrossBucketEntries> _crossBucketEntriesIn;
        Span<K32CrossBucketEntries> _crossBucketEntriesOut;

        byte* _crossBucketWriteBuffer = nullptr;
    #endif

public:
    // Timings
    Duration _tableIOWait    = Duration::zero();
    Duration _sortTime       = Duration::zero();
    Duration _distributeTime = Duration::zero();
    Duration _matchTime      = Duration::zero();
    Duration _fxTime         = Duration::zero();
};


#if _VALIDATE_Y

//-----------------------------------------------------------
template<uint32 _numBuckets>
void DbgValidateY( const TableId table, const FileId fileId, DiskPlotContext& context
    #if BB_DP_FP_MATCH_X_BUCKET
        ,const Span<K32CrossBucketEntries> crossBucketEntriesIn
    #endif
 )
{
    if( table >= TableId::Table2 && table < TableId::Table7 )
    {
        Log::Line( "[DEBUG: Validating table y %u]", table+1 );

        DiskBufferQueue& ioQueue = *context.ioQueue;
        Fence fence;

        ioQueue.SeekBucket( fileId, 0, SeekOrigin::Begin );
        ioQueue.CommitCommands();

        Span<uint64> yRef;
        // Load ref
        {
            Log::Line( " Loading ref" );
            // Debug::LoadRefTableByName( table, "t%d.y-dp-unbounded.tmp", yRef );
            Debug::LoadDPUnboundedY( table, yRef );
        }

        Span<uint64> yTmp = bbcvirtallocboundednuma_span<uint64>( yRef.Length() );

        // Read our entries

        // Read our entries
        {
            const uint32 k          = 32;
            const uint32 bucketBits = bblog2( _numBuckets );
            const uint32 yBits      = k + kExtraBits - bucketBits;

            Span<uint64> reader = yTmp.SliceSize( yRef.Length() / 2 );
            Span<uint64> tmp    = yTmp.Slice( yRef.Length() / 2 );

            for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
            {
                Log::Line( " Validating %u", bucket );

                Span<uint32> bucketReader = reader.template As<uint32>();

                ioQueue.ReadBucketElementsT( fileId, bucketReader );
                ioQueue.SignalFence( fence );
                ioQueue.CommitCommands();
                fence.Wait();

                #if BB_DP_FP_MATCH_X_BUCKET
                    bucketReader = crossBucketEntriesIn[bucket].CopyYAndExtend( bucketReader );
                #endif

                // Sort
                RadixSort256::Sort<BB_MAX_JOBS>( *context.threadPool, bucketReader.Ptr(), tmp.template As<uint32>().Ptr(), bucketReader.Length() );

                // Expand
                const uint64 yMask   = ((uint64)bucket) << yBits;
                Span<uint64> yValues = tmp;

                for( uint32 i = 0; i < bucketReader.Length(); i++ )
                    yValues[i] = yMask | (uint64)bucketReader[i];

                // Validate
                for( uint32 i = 0; i < bucketReader.Length(); i++ )
                {
                    const uint64 r = yRef   [i];
                    const uint64 y = yValues[i];

                    ASSERT( y == r );
                }

                yRef = yRef.Slice( bucketReader.Length() );
            }
        }

        // Cleanup
        bbvirtfreebounded( yTmp.Ptr() );

        ioQueue.SeekBucket( fileId, 0, SeekOrigin::Begin );
        ioQueue.CommitCommands();
    }

    // _yRefWriter = _yRef;
}

#endif

