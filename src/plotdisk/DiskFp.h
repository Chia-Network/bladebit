#pragma once
#include "util/Util.h"
#include "plotdisk/DiskPlotConfig.h"
#include "plotdisk/DiskBufferQueue.h"
#include "plotdisk/DiskPlotContext.h"
#include "plotdisk/FpGroupMatcher.h"
#include "plotdisk/FpFxGen.h"
#include "plotdisk/BlockWriter.h"
#include "plotdisk/DiskPlotInfo.h"
#include "plotdisk/BitBucketWriter.h"
#include "util/StackAllocator.h"
#include "plotting/TableWriter.h"

#if _DEBUG
    #include "DiskPlotDebug.h"
#endif


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

template<TableId table>
struct FpMapType { using Type = uint64; };

template<>
struct FpMapType<TableId::Table2> { using Type = uint32; };

template<TableId table, uint32 _numBuckets, bool IsT7Out = false>
class DiskFp
{
public:
    using InInfo = DiskPlotInfo<table-1, _numBuckets>;
    using Info   = DiskPlotInfo<table, _numBuckets>;

    static constexpr uint32 _k               = Info::_k;
    static constexpr uint64 MaxBucketEntries = Info::MaxBucketEntries;
    static constexpr uint32 MapBucketCount   = table == TableId::Table2 ? 1 : _numBuckets + 1;

    using Entry    = FpEntry<table-1>;
    using EntryOut = FpEntry<table>;
    using TMetaIn  = typename TableMetaType<table>::MetaIn;
    using TMetaOut = typename TableMetaType<table>::MetaOut;
    using TYOut    = typename FpFxGen<table>::TYOut;
    using TAddress = uint64;

public:

    //-----------------------------------------------------------
    inline DiskFp( DiskPlotContext& context, const FileId inFxId, const FileId outFxId )
        : _context( context )
        , _pool   ( *context.threadPool )
        , _ioQueue( *context.ioQueue )
        , _inFxId ( inFxId  )
        , _outFxId( outFxId )
        , _threadCount( context.fpThreadCount )
    {
        _crossBucketInfo.maxBuckets = _numBuckets;
    }

    //-----------------------------------------------------------
    inline void Run()
    {
        _ioQueue.SeekBucket( _inFxId , 0, SeekOrigin::Begin );
        _ioQueue.SeekBucket( _outFxId, 0, SeekOrigin::Begin );
        _ioQueue.CommitCommands();

        StackAllocator alloc( _context.heapBuffer, _context.heapSize );
        AllocateBuffers( alloc, _context.tmp2BlockSize, _context.tmp1BlockSize );

        FpTable();

        _context.entryCounts[(int)table] = (uint64)_tableEntryCount;

        // #TODO: Pass in the fences and then wait before we start the next table...
        // Wait for all writes to finish
        _ioQueue.SignalFence( _writeFence, _numBuckets+1 );
        _ioQueue.CommitCommands();
        _writeFence.Wait( _numBuckets+1 );

#if _DEBUG
        if( 0 && table == TableId::Table2 )
        {
            Log::Line( "Validating Xs" );
            uint32* xRef     = nullptr;
            uint64  refCount = 0;

            FatalIf( !Debug::LoadRefTable( "/mnt/p5510a/reference/t1.x.tmp", xRef, refCount ),
                "Failed to load ref table" );
            ASSERT( refCount == (1ull << _K) );

            int err;
            uint32* xs = (uint32*)IOJob::ReadAllBytesDirect( "/mnt/p5510a/disk_tmp/t1_0.tmp", err );
            FatalIf( !xs, "Failed to rad Xs with error: %d", err );

            for( uint64 i = 0; i < refCount; i++ )
            {
                if( xs[i] != xRef[i] )
                {
                    if( xs[i+1] == xRef[i] || xs[i] == xRef[i+1] )
                    {
                        i++;
                        continue;
                    }
                    else
                    {
                        uint32 xA[3] = { xs[i], xs[i+1], xs[i+2] };
                        uint32 xB[3] = { xRef[i], xRef[i+1], xRef[i+2] };

                        std::sort( xA, &xA[3] );
                        std::sort( xB, &xB[3] );

                        if( memcmp( xA, xB, sizeof( xA ) ) == 0 )
                        {
                            i+=2;
                            continue;
                        }
                    }
                    ASSERT( 0 );
                }
            }

            bbvirtfree( xRef );
            bbvirtfree( xs );
            Log::Line( "All good!" );
        }
#endif
    }

    //-----------------------------------------------------------
    inline void RunF7()
    {
        _threadCount = _context.cThreadCount;

        _ioQueue.SeekBucket( _inFxId, 0, SeekOrigin::Begin );
        _ioQueue.CommitCommands();

        StackAllocator alloc( _context.heapBuffer, _context.heapSize );
        AllocateBuffers( alloc, _context.tmp2BlockSize, _context.tmp1BlockSize );

        _mapBitWriter.SetFileId( FileId::MAP7 );
        WriteCTables();

        // Wait for all writes to finish
        _ioQueue.SignalFence( _writeFence, _numBuckets+1 );
        _ioQueue.CommitCommands();
        _writeFence.Wait( _numBuckets+1 );
    }

    //-----------------------------------------------------------
    inline void AllocateBuffers( IAllocator& alloc, const size_t fxBlockSize, const size_t pairBlockSize, bool dryRun = false )
    {
        using info = DiskPlotInfo<TableId::Table4, _numBuckets>;
        
        const size_t maxEntries      = info::MaxBucketEntries;
        const size_t maxSliceEntries = info::MaxBucketSliceEntries;
        const size_t maxEntriesX     = maxEntries + BB_DP_CROSS_BUCKET_MAX_ENTRIES;
       
        // Working buffers
        _entries[0] = alloc.CAlloc<Entry>( maxEntries );
        _entries[1] = alloc.CAlloc<Entry>( maxEntries );

        _y[0]    = alloc.CAlloc<uint64>( maxEntriesX );
        _y[1]    = alloc.CAlloc<uint64>( maxEntriesX );
        
        _map[0]  = alloc.CAlloc<uint64>( maxEntriesX );

        _meta[0] = (TMetaIn*)alloc.CAlloc<Meta4>( maxEntriesX );
        _meta[1] = (TMetaIn*)alloc.CAlloc<Meta4>( maxEntriesX );
        
        _pair[0] = alloc.CAlloc<Pair>( maxEntriesX );
        
        // IO buffers
        const uint32 bucketBits      = bblog2( _numBuckets );
        const size_t entrySizeBits   = info::EntrySizePackedBits;       // Entries stored in fx file (tmp2)
        const size_t pairBits        = _k + 1 - bucketBits + 9;         // Entries stored in table file (tmp1)
        const size_t mapBits         = _k + 1 - bucketBits + _k + 1;    // Entries stored in map file (tmp1)

        // Because bitwriter needs each slice buffer to be rounded up
        // to block-size, we need to account for that here when allocating.
        const size_t entrySliceAlloc = (size_t)RoundUpToNextBoundaryT( CDiv( (uint128)maxSliceEntries * entrySizeBits, 8 ), (uint128)fxBlockSize   );
        const size_t mapSliceAslloc  = (size_t)RoundUpToNextBoundaryT( CDiv( (uint128)maxSliceEntries * mapBits      , 8 ), (uint128)pairBlockSize );
        
        const size_t genEntriesPerBucket  = (size_t)( CDivT( maxEntries, (size_t)_numBuckets ) * BB_DP_XTRA_ENTRIES_PER_BUCKET );
        const size_t perBucketEntriesSize = RoundUpToNextBoundaryT( CDiv( entrySizeBits * genEntriesPerBucket, 8 ), fxBlockSize ) + 
                                            RoundUpToNextBoundaryT( CDiv( entrySizeBits * BB_DP_CROSS_BUCKET_MAX_ENTRIES, 8 ), fxBlockSize );

        const size_t fxWriteSize   = entrySliceAlloc * _numBuckets; //(size_t)_numBuckets * perBucketEntriesSize;
        const size_t pairWriteSize = CDiv( ( maxEntriesX ) * pairBits, 8 ); 
        const size_t mapWriteSize  = mapSliceAslloc * _numBuckets;   //CDiv( ( maxEntriesX ) * mapBits , 8 ); // #TODO: Might need the cross bucket entries added here too... But maybe the multiplier covers it

        // #TODO: These need to have buffers rounded-up to block size per bucket I think...
        //         So we need to have this divided into bucket slices, then each sice rounded-up
        //         to the block size, finally, the sum of that is the allocation size.
        _fxRead [0]   = alloc.Alloc( fxWriteSize, fxBlockSize );
        _fxRead [1]   = alloc.Alloc( fxWriteSize, fxBlockSize );
        _fxWrite[0]   = alloc.Alloc( fxWriteSize, fxBlockSize );
        _fxWrite[1]   = alloc.Alloc( fxWriteSize, fxBlockSize );

        _pairWrite[0] = alloc.Alloc( pairWriteSize, pairBlockSize );
        _pairWrite[1] = alloc.Alloc( pairWriteSize, pairBlockSize );

        _mapWrite[0]  = alloc.Alloc( mapWriteSize, pairBlockSize );
        _mapWrite[1]  = alloc.Alloc( mapWriteSize, pairBlockSize );

        // Block bit buffers
        byte* fxBlocks   = (byte*)alloc.CAlloc( _numBuckets, fxBlockSize  , fxBlockSize   );  // Fx write
        byte* pairBlocks = (byte*)alloc.CAlloc( 1          , pairBlockSize, pairBlockSize );  // Pair write
        byte* mapBlocks  = nullptr;

        // #TODO: Just re-use another buffer for x in T2. mapWriteSize will be already biffer than we need I think.
        //        Or do NOT allocate map write buffers in T2. Just allocate xWriter.
        if( table == TableId::Table2 )
            _xWriter = BlockWriter<uint32>( alloc, FileId::T1, _mapWriteFence, pairBlockSize, maxEntries );
        else
            mapBlocks = (byte*)alloc.CAlloc( MapBucketCount, pairBlockSize, pairBlockSize );  // Map write

        if( !dryRun )
        {
            const FileId mapWriterId = IsT7Out ? FileId::MAP7 :
                                       table == TableId::Table2 ? FileId::T1 : FileId::MAP2 + (FileId)table-2; // Writes previous buffer's key as a map

            if( !IsT7Out )
            {
                _fxBitWriter   = BitBucketWriter<_numBuckets>( _ioQueue, _outFxId, (byte*)fxBlocks );
                _pairBitWriter = BitBucketWriter<1>          ( _ioQueue, FileId::T1 + (FileId)table, (byte*)pairBlocks );
            }

            if( table > TableId::Table2 )
                _mapBitWriter = BitBucketWriter<MapBucketCount>( _ioQueue, mapWriterId, (byte*)mapBlocks );
        }
    }

    //-----------------------------------------------------------
    inline static size_t GetRequiredHeapSize( const size_t fxBlockSize, const size_t pairBlockSize )
    {
        DiskPlotContext cx = { 0 };
        DiskFp<TableId::Table2, _numBuckets, false> fp2( cx, FileId::None, FileId::None );
        DiskFp<TableId::Table4, _numBuckets, false> fp4( cx, FileId::None, FileId::None );

        //  #TODO: Revert checking both, use whichever is bigger after testing explicitly
        DummyAllocator alloc2;
        DummyAllocator alloc4;
        fp2.AllocateBuffers( alloc2, fxBlockSize, pairBlockSize, true );
        fp4.AllocateBuffers( alloc4, fxBlockSize, pairBlockSize, true );
        
        const size_t reqSize = std::max( alloc2.Size(), alloc4.Size() );
        return reqSize;
    }

    //-----------------------------------------------------------
    inline Duration IOWaitTime() const { return _ioWaitTime; }
    
// private:
    //-----------------------------------------------------------
    inline void FpTable()
    {
        // Load initial bucket
        LoadBucket( 0 );

        for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
        {
            // Load next bucket in the background
            if( bucket + 1 < _numBuckets )
                LoadBucket( bucket + 1 );

            _readFence.Wait( bucket + 1, _ioWaitTime );

            FpBucket( bucket );
        }

        _fxBitWriter  .SubmitLeftOvers();
        _pairBitWriter.SubmitLeftOvers();

        if( table > TableId::Table2 )
            _mapBitWriter .SubmitLeftOvers();
    }

    //-----------------------------------------------------------
    inline void FpBucket( const uint32 bucket )
    {
        // Log::Verbose( " Bucket %u", bucket );
        const int64 inEntryCount = (int64)_context.bucketCounts[(int)table-1][bucket];
        // const bool  isLastBucket = bucket + 1 == _numBuckets;

        const byte* packedEntries = GetReadBufferForBucket( bucket );

        // The first part of the buffer is reserved for storing cross bucket entries. 
        uint64*  y     = _y   [0] + BB_DP_CROSS_BUCKET_MAX_ENTRIES;
        TMetaIn* meta  = _meta[0] + BB_DP_CROSS_BUCKET_MAX_ENTRIES; 
        Pair*    pairs = _pair[0] + BB_DP_CROSS_BUCKET_MAX_ENTRIES;
        uint64*  map   = _map [0] + BB_DP_CROSS_BUCKET_MAX_ENTRIES; // Needs prefix offset because it is used as meta input in cross-bucket matches

        // Expand entries to be 64-bit aligned, sort them, then unpack them to individual components
        ExpandEntries( packedEntries, 0, _entries[0], inEntryCount );
        SortEntries<Entry, InInfo::YBitSize>( _entries[0], _entries[1], inEntryCount );
        UnpackEntries( bucket, _entries[0], inEntryCount, y, map, meta );

        WriteMap( bucket, inEntryCount, map, (uint64*)_meta[1] );

        BB_DBG_DP_DumpUnboundedY( table-1, bucket, _context, Span<uint64>( y, inEntryCount ) );

        const TMetaIn* inMeta = ( table == TableId::Table2 ) ? (TMetaIn*)map : meta;

        TYOut*    outY    = (TYOut*)_y[1];
        TMetaOut* outMeta = (TMetaOut*)_meta[1];

        // Match
        const int64 outEntryCount = MatchEntries( bucket, inEntryCount, y, inMeta, pairs );

        #if !BB_DP_DBG_UNBOUNDED_DISABLE_CROSS_BUCKET
            const int64 crossMatchCount  = _crossBucketInfo.matchCount;
        #else
            const int64 crossMatchCount  = 0;
        #endif
        const int64 writeEntrtyCount = outEntryCount + crossMatchCount;

        #if !BB_DP_DBG_UNBOUNDED_DISABLE_CROSS_BUCKET
            if( bucket > 0 )
            {
                ProcessCrossBucketEntries( bucket - 1, pairs - crossMatchCount, outY, outMeta );
                ASSERT( crossMatchCount <= std::numeric_limits<uint32>::max() );

                _context.ptrTableBucketCounts[(int)table][bucket-1] += (uint32)crossMatchCount;
            }
        #endif
        
        WritePairsToDisk( bucket, writeEntrtyCount, pairs - crossMatchCount );

        FxGen( bucket, outEntryCount, pairs, y, inMeta, outY + crossMatchCount, outMeta + crossMatchCount );

        WriteEntriesToDisk( bucket, writeEntrtyCount, outY, outMeta, (EntryOut*)_entries[0] );

        // Save bucket length before y-sort since the pairs remain unsorted
        _context.ptrTableBucketCounts[(int)table][bucket] += (uint32)outEntryCount;

        _tableEntryCount += writeEntrtyCount;
        _mapOffset       += (uint64)inEntryCount;
        _crossBucketInfo.matchCount  = 0;
    }

    //-----------------------------------------------------------
    void WriteMap( const uint32 bucket, const int64 entryCount, const uint64* map, uint64* outMap )
    {
        ASSERT( entryCount < (1ll << _k) );
        
        if constexpr ( table == TableId::Table2 )
        {
            // #NOTE: We don't unpack xs directly into the write buffer because
            //        map is used as meta during matching, which needs a prefix zone
            //        to perfrom cross-bucket matches.
            uint32* xBuffer = _xWriter.GetNextBuffer( _ioWaitTime );

            AnonMTJob::Run( _pool, _threadCount, [=]( AnonMTJob* self ) {
                
                int64 count, offset, end;
                GetThreadOffsets( self, entryCount, count, offset, end );

                if( count )
                    memcpy( xBuffer + offset, ((uint32*)map) + offset, (size_t)count * sizeof( uint32 ) );
            });

            _xWriter.SubmitBuffer( _ioQueue, (size_t)entryCount );

            if( bucket == _numBuckets - 1 )
                _xWriter.SubmitFinalBlock( _ioQueue );

            return;
        }

        // Write the key as a map to the final entry location.
        // We do this by writing into buckets to final sorted (current) location
        // into its original bucket, with the offset in that bucket.
        using Job = AnonPrefixSumJob<uint32>;

        uint32 totalCounts   [_numBuckets+1];
        uint64 totalBitCounts[_numBuckets+1];

        Job::Run( _pool, _threadCount, [&]( Job* self ) {

            const uint32 bucketBits   = Info::BucketBits;
            const uint32 bucketShift  = _k - bucketBits;
            const uint32 bitSize      = Info::MapBitSize + _k - bucketBits;
            const uint32 encodeShift  = Info::MapBitSize;
            const uint32 numBuckets   = _numBuckets + 1;


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
            const uint64 tableOffset = _mapOffset + (uint64)offset;

            const uint64* reverseMap    = map + offset;
                  uint64* outMapBuckets = outMap;

            for( int64 i = 0; i < count; i++ )
            {
                const uint64 origin = reverseMap[i];
                const uint64 b      = origin >> bucketShift;
                const uint32 dstIdx = --pfxSum[b];

                const uint64 finalIdx = tableOffset + (uint64)i;
                outMapBuckets[dstIdx] = (origin << encodeShift) | finalIdx;

                ASSERT( (int64)dstIdx < entryCount );
                ASSERT( finalIdx < ( 1ull << encodeShift ) );
            }

            auto&   bitWriter = _mapBitWriter;
            uint64* bitCounts = totalBitCounts;

            if( self->IsControlThread() )
            {
                self->LockThreads();

                // Convert counts to bit sizes
                for( uint32 i = 0; i < numBuckets; i++ )
                    bitCounts[i] = (uint64)totalCounts[i] * bitSize;

                byte* writeBuffer = GetMapWriteBuffer( bucket );

                // Wait for the buffer to be available first
                if( bucket > 1 )
                    _mapWriteFence.Wait( bucket - 2, _ioWaitTime );

                bitWriter.BeginWriteBuckets( bitCounts, writeBuffer );

                self->ReleaseThreads();
            }
            else
                self->WaitForRelease();


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
            self->SyncThreads();
            if( self->IsControlThread() )
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

                bitWriter.Submit();
                _ioQueue.SignalFence( _mapWriteFence, bucket );
                _ioQueue.CommitCommands();
            }
        });
    }

    //-----------------------------------------------------------
    inline int64 MatchEntries( const uint32 bucket, const int64 inEntryCount, const uint64* y, const TMetaIn* meta, Pair* pairs )
    {
        _crossBucketInfo.bucket = bucket;
        FpGroupMatcher matcher( _context, MaxBucketEntries, _y[1], (Pair*)_meta[1], pairs );
        const int64 entryCount = (int64)matcher.Match( inEntryCount, y, meta, &_crossBucketInfo );

        return entryCount;
    }

    //-----------------------------------------------------------
    void ProcessCrossBucketEntries( const uint32 bucket, Pair* dstPairs, TYOut* outY, TMetaOut* outMeta )
    {
        FpCrossBucketInfo& info = _crossBucketInfo;
        // ASSERT( info.matchCount );

        if( !info.matchCount )
            return;

        FpFxGen<table>::ComputeFx( (int64)info.matchCount, info.pair, info.y, (TMetaIn*)info.meta, outY, outMeta, 0 );
        
        const uint32 matchOffset = (uint32)info.matchOffset[0]; // Grab the previous bucket's offset and apply it
        const Pair * srcPairs    = info.pair;

        for( uint64 i = 0; i < info.matchCount; i++ )
        {
            auto& src = srcPairs[i];
            auto& dst = dstPairs[i];
            dst.left  = src.left  + matchOffset;
            dst.right = src.right + matchOffset;
        }
    }

    //-----------------------------------------------------------
    void FxGen( const uint32 bucket, const int64 entryCount, 
                const Pair* pairs, const uint64* inY, const TMetaIn* inMeta, 
                TYOut* outY, TMetaOut* outMeta )
    {
        FpFxGen<table> fx( _pool, _threadCount );
        fx.ComputeFxMT( entryCount, pairs, inY, inMeta, outY, outMeta );
    }

    //-----------------------------------------------------------
    inline void WritePairsToDisk( const uint32 bucket, const int64 entryCount, const Pair* pairs )
    {
        uint64 bitBucketSizes = (uint64)entryCount * Info::PairBitSize;
        byte*  writeBuffer    = GetPairWriteBuffer( bucket );

        _pairWriteFence.Wait( bucket, _ioWaitTime );
        _pairBitWriter.BeginWriteBuckets( &bitBucketSizes, writeBuffer );

        AnonMTJob::Run( _pool, _threadCount, [=]( AnonMTJob* self ) {

            int64 count, offset, end;
            GetThreadOffsets( self, entryCount, count, offset, end );

            const Pair* pair = pairs + offset;

            BitWriter writer = _pairBitWriter.GetWriter( 0, (uint64)offset * Info::PairBitSize );

            ASSERT( count >= 2 );
            PackPairs( 2, pair, writer );
            self->SyncThreads();
            PackPairs( count-2, pair+2, writer );
        });

        _pairBitWriter.Submit();
        _ioQueue.SignalFence( _pairWriteFence, bucket + 1 );   // #TODO: Don't signal here, signal on cross-bucket?
        _ioQueue.CommitCommands();
    }

    //-----------------------------------------------------------
    inline static void PackPairs( const int64 entryCount, const Pair* pair, BitWriter& writer )
    {
        const uint32 entrySizeBits = Info::PairBitSize;
        const uint32 shift         = Info::PairBitSizeL;
        const uint64 mask          = ( 1ull << shift ) - 1;

        const Pair* end = pair + entryCount;

        while( pair < end )
        {
            ASSERT( pair->right - pair->left < 512 );

            writer.Write( ( (uint64)(pair->right - pair->left) << shift ) | ( pair->left & mask ), entrySizeBits );
            pair++;
        }
    }

    //-----------------------------------------------------------
    inline void WriteEntriesToDisk( const uint32 bucket, const int64 entryCount, const TYOut* outY, const TMetaOut* outMeta, EntryOut* dstEntriesBuckets )
    {
        using Job = AnonPrefixSumJob<uint32>;

        Job::Run( _pool, _threadCount, [=]( Job* self ) {
            
            const uint32 yBits         = Info::YBitSize;
            const uint32 yShift        = Info::BucketShift;
            const uint64 yMask         = ( 1ull << yBits ) - 1;
            const size_t entrySizeBits = Info::EntrySizePackedBits;

            int64 count, offset, end;
            GetThreadOffsets( self, entryCount, count, offset, end );

            // Count Y buckets
            uint32 counts        [_numBuckets] = { 0 };
            uint32 pfxSum        [_numBuckets];
            uint32 totalCounts   [_numBuckets];
            uint64 totalBitCounts[_numBuckets];

            const TYOut* ySrc    = outY + offset;
            const TYOut* ySrcEnd = outY + end;

            do {
                counts[(*ySrc) >> yShift]++;
            } while( ++ySrc < ySrcEnd );

            self->CalculatePrefixSum( _numBuckets, counts, pfxSum, totalCounts );

            // Distribute to the appropriate buckets as an expanded entry
            const int64     mapIdx     = _tableEntryCount + offset;
            const TMetaOut* metaSrc    = outMeta + offset;
                  EntryOut* dstEntries = dstEntriesBuckets;

            ySrc = outY + offset;

            for( int64 i = 0; i < count; i++ )
            {
                const uint64 y      = ySrc[i];
                const uint32 dstIdx = --pfxSum[y >> yShift];
                ASSERT( (int64)dstIdx < entryCount );

                EntryOut& e = dstEntries[dstIdx];
                e.ykey =  ( (uint64)(mapIdx+i) << yBits ) | ( y & yMask );  // Write y and key/map as packed already

                if constexpr ( table < TableId::Table7 )
                    e.meta = metaSrc[i];
            }

            // Prepare to pack entries
            auto& bitWriter = _fxBitWriter;

            if( self->IsControlThread() )
            {
                self->LockThreads();

                for( uint32 i = 0; i < _numBuckets; i++ )
                    _context.bucketCounts[(int)table][i] += totalCounts[i];

                // Convert counts to bit sizes
                for( uint32 i = 0; i < _numBuckets; i++ )
                    totalBitCounts[i] = totalCounts[i] * entrySizeBits;
                
                byte* writeBuffer = GetWriteBuffer( bucket );
                
                // Wait for the buffer to be available first
                if( bucket > 1 )
                    _writeFence.Wait( bucket - 2, _ioWaitTime );

                bitWriter.BeginWriteBuckets( totalBitCounts, writeBuffer );

                _sharedTotalBitCounts = totalBitCounts;

                self->ReleaseThreads();
            }
            else
                self->WaitForRelease();


            // Bit-compress/pack each bucket entries
            uint64* totalBucketBitCounts = _sharedTotalBitCounts;
            uint64  bitsWritten          = 0;

            for( uint32 i = 0; i < _numBuckets; i++ )
            {
                const uint64 writeOffset = pfxSum[i];
                const uint64 bitOffset   = writeOffset * entrySizeBits - bitsWritten;
                bitsWritten += totalBucketBitCounts[i];

                ASSERT( bitOffset + counts[i] * entrySizeBits <= totalBucketBitCounts[i] );

                BitWriter writer = bitWriter.GetWriter( i, bitOffset );

                const EntryOut* entry = dstEntries + writeOffset;
                ASSERT( counts[i] >= 2 );

                // Compress a couple of entries first, so that we don't get any simultaneaous writes to the same fields
                PackEntries( entry, 2, writer );
                self->SyncThreads();
                PackEntries( entry + 2, (int64)counts[i]-2, writer );
            }

            // Write buckets to disk
            if( self->IsControlThread() )
            {
                self->LockThreads();
                bitWriter.Submit();
                _ioQueue.SignalFence( _writeFence, bucket );
                _ioQueue.CommitCommands();
                self->ReleaseThreads();
            }
            else
                self->WaitForRelease();
        });
    }

    //-----------------------------------------------------------
    void WriteCTables()
    {
        _readFence .Reset( 0 );
        _writeFence.Reset( 0 );

        uint32 c1NextCheckpoint = 0;  // How many C1 entries to skip until the next checkpoint. If there's any entries here, it means the last bucket wrote a
        uint32 c2NextCheckpoint = 0;  // checkpoint entry and still had entries which did not reach the next checkpoint.
                                      // Ex. Last bucket had 10005 entries, so it wrote a checkpoint at 0 and 10000, then it counted 5 more entries, so
                                      // the next checkpoint would be after 9995 entries.

        // These buffers are small enough on k32 (around 1.6MiB for C1, C2 is negligible), we keep the whole thing in memory,
        // while we write C3 to the actual file
        const uint32 c1Interval     = kCheckpoint1Interval;
        const uint32 c2Interval     = kCheckpoint1Interval * kCheckpoint2Interval;

        const uint64 tableLength    = _context.entryCounts[(int)TableId::Table7];
        const uint32 c1TotalEntries = (uint32)CDiv( tableLength, (int)c1Interval ) + 1; // +1 because chiapos adds an extra '0' entry at the end
        const uint32 c2TotalEntries = (uint32)CDiv( tableLength, (int)c2Interval ) + 1; // +1 because we add a short-circuit entry to prevent C2 lookup overflows
                                                                                        // #TODO: Remove the extra c2 entry?

        const size_t c1TableSizeBytes = c1TotalEntries * sizeof( uint32 );
        const size_t c2TableSizeBytes = c2TotalEntries * sizeof( uint32 );

        // Use meta1 for here, it's big enough to hold both at any bucket size
        // meta1 is 64MiB on  which is enough to c1 and c2
        uint32* c1Buffer = (uint32*)_meta[1];
        uint32* c2Buffer = c1Buffer + c1TotalEntries;

        uint32 c3ParkOverflowCount = 0;                 // Overflow entries from a bucket that did not make it into a C3 park this bucket. Saved for the next bucket.
        uint32 c3ParkOverflow[kCheckpoint1Interval];    // They are then copied to a "prefix region" in the f7 buffer of the next park.

        size_t c3TableSizeBytes = 0; // Total size of the C3 table

        // Seek to the start of the C3 table instead of writing garbage data.
        _ioQueue.SeekFile( FileId::PLOT, 0, (int64)(c1TableSizeBytes + c2TableSizeBytes), SeekOrigin::Current );
        _ioQueue.CommitCommands();

        // Load initial bucket
        LoadBucket( 0 );

        uint32* c1Writer = c1Buffer;
        uint32* c2Writer = c2Buffer;

        for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
        {
            // Load next bucket in the background
            if( bucket + 1 < _numBuckets )
                LoadBucket( bucket + 1 );

            _readFence.Wait( bucket + 1, _ioWaitTime );

            const uint32 bucketLength  = (int64)_context.bucketCounts[(int)TableId::Table7][bucket];
            const byte*  packedEntries = GetReadBufferForBucket( bucket );

            uint32* f7 = ((uint32*)_y[0]) + kCheckpoint1Interval;

            using T7Entry = FpEntry<TableId::Table7>;
            static_assert( sizeof( T7Entry ) == sizeof( uint64 ) );

            ExpandEntries<T7Entry>( packedEntries, 0, (T7Entry*)_entries[0], (int64)bucketLength );
            // #if _DEBUG
            // {
            //     UnpackEntries<T7Entry, uint32, true>( bucket, (T7Entry*)_entries[0], (int64)bucketLength, f7, _map[0], nullptr );
            //     RadixSort256::Sort<BB_DP_MAX_JOBS>( *_context.threadPool, f7, (uint32*)_map[0], bucketLength );
            // }
            // #endif
            SortEntries<T7Entry, Info::YBitSize>( (T7Entry*)_entries[0], (T7Entry*)_entries[1], (int64)bucketLength );

            // Any bucket count above 128 will only require 3 iterations w/ k=32, so 
            // the sorted entries will be at the _entries[1] buffer in that case
            auto* sortedEntries = (T7Entry*)( _numBuckets > 128 ? _entries[1] : _entries[0] );

            UnpackEntries<T7Entry, uint32>( bucket, sortedEntries, (int64)bucketLength, f7, _map[0], nullptr );
            WriteMap( bucket, (int64)bucketLength, _map[0], (uint64*)_meta[0] );

            _mapOffset += (uint64)bucketLength;

            /// Now handle f7 and write them into C tables
            /// We will set the addersses to these tables accordingly.

            // Write C1
            {
                ASSERT( bucketLength > c1NextCheckpoint );

                // #TODO: Do C1 multi-threaded? For now jsut single-thread it...
                for( uint32 i = c1NextCheckpoint; i < bucketLength; i += c1Interval )
                    *c1Writer++ = Swap32( f7[i] );
                
                // Track how many entries we covered in the last checkpoint region
                const uint32 c1Length          = bucketLength - c1NextCheckpoint;
                const uint32 c1CheckPointCount = CDiv( c1Length, (int)c1Interval );

                c1NextCheckpoint = c1CheckPointCount * c1Interval - c1Length;
            }

            // Write C2
            {
                // C2 has so few entries on k=32 that there's no sense in doing it multi-threaded
                static_assert( _K == 32 );

                if( c2NextCheckpoint >= bucketLength )
                    c2NextCheckpoint -= bucketLength;   // No entries to write in this bucket
                else
                {
                    for( uint32 i = c2NextCheckpoint; i < bucketLength; i += c2Interval )
                        *c2Writer++ = Swap32( f7[i] );
                
                    // Track how many entries we covered in the last checkpoint region
                    const uint32 c2Length          = bucketLength - c2NextCheckpoint;
                    const uint32 c2CheckPointCount = CDiv( c2Length, (int)c2Interval );

                    c2NextCheckpoint = c2CheckPointCount * c2Interval - c2Length;
                }
            }

            // Write C3
            {
                const bool isLastBucket = bucket == _numBuckets-1;

                uint32* c3F7           = f7;
                uint32  c3BucketLength = bucketLength;

                if( c3ParkOverflowCount )
                {
                    // Copy our overflow to the prefix region of our f7 buffer
                    c3F7 -= c3ParkOverflowCount;
                    c3BucketLength += c3ParkOverflowCount;

                    memcpy( c3F7, c3ParkOverflow, sizeof( uint32 ) * c3ParkOverflowCount );

                    c3ParkOverflowCount = 0;
                }
                
                
            #if _DEBUG
                // #TODO: TEST: Remove this
                // Dump f7's that have the value of 0xFFFFFFFF for now,
                // this is just for compatibility with RAM bladebit
                // for testing plots against it.
                // if( isLastBucket )
                // {
                //     while( c3F7[c3BucketLength-1] == 0xFFFFFFFF )
                //         c3BucketLength--;
                // }
            #endif

                // See TableWriter::GetC3ParkCount for details
                uint32 parkCount       = c3BucketLength / kCheckpoint1Interval;
                uint32 overflowEntries = c3BucketLength - ( parkCount * kCheckpoint1Interval );

                // Greater than 1 because the first entry is excluded as it is written in C1 instead.
                if( isLastBucket && overflowEntries > 1 )
                {
                    overflowEntries = 0;
                    parkCount++;
                }
                else if( overflowEntries && !isLastBucket )
                {
                    // Save any entries that don't fill-up a full park for the next bucket
                    memcpy( c3ParkOverflow, c3F7 + c3BucketLength - overflowEntries, overflowEntries * sizeof( uint32 ) );
                    
                    c3ParkOverflowCount = overflowEntries;
                    c3BucketLength -= overflowEntries;
                }

                const size_t c3BufferSize = CalculateC3Size() * parkCount;
                      byte*  c3Buffer     = GetWriteBuffer( bucket );

                if( bucket > 1 )
                    _writeFence.Wait( bucket - 2, _ioWaitTime );

                // #NOTE: This function uses re-writes our f7 buffer, so ensure it is done after
                //        that buffer is no longer needed.
                const size_t sizeWritten = TableWriter::WriteC3Parallel<BB_DP_MAX_JOBS>( *_context.threadPool, 
                                                _threadCount, c3BucketLength, c3F7, c3Buffer );
                ASSERT( sizeWritten == c3BufferSize );

                c3TableSizeBytes += sizeWritten;

                // Write the C3 table to the plot file directly
                _ioQueue.WriteFile( FileId::PLOT, 0, c3Buffer, c3BufferSize );
                _ioQueue.SignalFence( _writeFence, bucket );
                _ioQueue.CommitCommands();
            }
        }

        // Submit any left-over map bits
        _mapBitWriter.SubmitLeftOvers();

        // Seek back to the begining of the C1 table and
        // write C1 and C2 buffers to file, then seek back to the end of the C3 table

        c1Buffer[c1TotalEntries-1] = 0;          // Chiapos adds a trailing 0
        c2Buffer[c2TotalEntries-1] = 0xFFFFFFFF; // C2 overflow protection      // #TODO: Remove?

        _readFence.Reset( 0 );

        _ioQueue.SeekBucket( FileId::PLOT, -(int64)( c1TableSizeBytes + c2TableSizeBytes + c3TableSizeBytes ), SeekOrigin::Current );
        _ioQueue.WriteFile( FileId::PLOT, 0, c1Buffer, c1TableSizeBytes );
        _ioQueue.WriteFile( FileId::PLOT, 0, c2Buffer, c2TableSizeBytes );
        _ioQueue.SeekBucket( FileId::PLOT, (int64)c3TableSizeBytes, SeekOrigin::Current );

        _ioQueue.SignalFence( _readFence, 1 );
        _ioQueue.CommitCommands();

        // Save C table addresses into the plot context.
        // And set the starting address for the table 1 to be written
        const size_t headerSize = _ioQueue.PlotHeaderSize();

        _context.plotTablePointers[7] = headerSize;                                       // C1
        _context.plotTablePointers[8] = _context.plotTablePointers[7] + c1TableSizeBytes; // C2
        _context.plotTablePointers[9] = _context.plotTablePointers[8] + c2TableSizeBytes; // C3
        _context.plotTablePointers[0] = _context.plotTablePointers[9] + c3TableSizeBytes; // T1

        // Save sizes
        _context.plotTableSizes[7] = c1TableSizeBytes;
        _context.plotTableSizes[8] = c2TableSizeBytes;
        _context.plotTableSizes[9] = c3TableSizeBytes;

        // Wait for all commands to finish
        _readFence.Wait( 1 );
    }

    //-----------------------------------------------------------
    inline static void PackEntries( const EntryOut* entries, const int64 entryCount, BitWriter& writer )
    {
        const uint32 yBits         = Info::YBitSize;
        const uint32 mapBits       = Info::MapBitSize;
        const uint32 ykeyBits      = yBits + mapBits;
        const size_t metaOutMulti  = Info::MetaMultiplier;
        const size_t entrySizeBits = Info::EntrySizePackedBits;

        static_assert( ykeyBits + metaOutMulti * _k == entrySizeBits );
        static_assert( metaOutMulti != 1 );

        const EntryOut* entry = entries;
        const EntryOut* end   = entry + entryCount;

        while( entry < end )
        {
            writer.Write( entry->ykey, ykeyBits );

            if constexpr ( metaOutMulti == 2 )
            {
                writer.Write( entry->meta, 64 );
            }
            else if constexpr ( metaOutMulti == 3 )
            {
                writer.Write( entry->meta.m0, 64 );
                writer.Write( entry->meta.m1, 32 );
            }
            else if constexpr ( metaOutMulti == 4 )
            {
                writer.Write( entry->meta.m0, 64 );
                writer.Write( entry->meta.m1, 64 );
            }

            entry++;
        }
    }

    //-----------------------------------------------------------
    template<typename TEntry, typename TY>
    inline void UnpackEntries( const uint32 bucket, const TEntry* packedEntries, const int64 entryCount, TY* outY, uint64* outMap, TMetaIn* outMeta )
    {
        using TMap = typename FpMapType<table>::Type;

        AnonMTJob::Run( _pool, _threadCount, [=]( AnonMTJob* self ) {

            constexpr uint32 metaMultipler = IsT7Out ? 0 : InInfo::MetaMultiplier;

            const uint32 yBits = IsT7Out ? Info::YBitSize : InInfo::YBitSize;
            const uint64 yMask = 0xFFFFFFFFFFFFFFFFull >> (64-yBits);

            const uint64 bucketBits = ((uint64)bucket) << yBits;

            int64 count, offset, end;
            GetThreadOffsets( self, entryCount, count, offset, end );

            TY*      y    = (TY*)outY;
            TMap*    map  = (TMap*)outMap;
            TMetaIn* meta = outMeta;
            
            const TEntry* entries = packedEntries;

            for( int64 i = offset; i < end; i++ )
            {
                auto& e = entries[i];

                const uint64 ykey = e.ykey;
                y  [i] = (TY)( bucketBits | ( ykey & yMask ) );
                map[i] = (TMap)( ykey >> yBits );
                ASSERT( map[i] <= ( 1ull << Info::MapBitSize ) - 1 );

                // Can't be 1 (table 1 only), because in that case the metadata is x, 
                // which is stored as the map
                if constexpr ( metaMultipler > 1 )
                    meta[i] = e.meta;
            }
        });
    }

    //-----------------------------------------------------------
    template<typename TEntry, uint32 yBitSize>
    inline void SortEntries( TEntry* entries, TEntry* tmpEntries, const int64 entryCount )
    {
        AnonPrefixSumJob<uint32>::Run( _pool, _threadCount, [=]( AnonPrefixSumJob<uint32>* self ) {
            
            int64 count, offset, end;
            GetThreadOffsets( self, entryCount, count, offset, end );

            const uint32 remainderBits = _k - yBitSize;
            EntrySort<remainderBits>( self, count, offset, entries, tmpEntries );
        });
    }

    //-----------------------------------------------------------
    template<uint32 remainderBits, typename T, typename TJob, typename BucketT>
    inline static void EntrySort( PrefixSumJob<TJob, BucketT>* self, const int64 entryCount, const int64 offset, 
                                  T* entries, T* tmpEntries )
    {
        ASSERT( self );
        ASSERT( entries );
        ASSERT( tmpEntries );
        ASSERT( entryCount > 0 );

        constexpr uint Radix = 256;

        constexpr uint32 MaxIter    = 4;
        constexpr int32  iterations = MaxIter - remainderBits / 8;
        constexpr uint32 shiftBase  = 8;

        BucketT counts     [Radix];
        BucketT pfxSum     [Radix];
        BucketT totalCounts[Radix];

        uint32 shift = 0;
        T* input  = entries;
        T* output = tmpEntries;

        const uint32 lastByteRemainderBits = remainderBits & 7;    // & 7 == % 8
        const uint32 lastByteMask          = 0xFF >> lastByteRemainderBits;
              uint32 masks[MaxIter]        = { 0xFF, 0xFF, 0xFF, 0xFF };

        masks[iterations-1] = lastByteMask;

        for( int32 iter = 0; iter < iterations ; iter++, shift += shiftBase )
        {
            const uint32 mask = masks[iter];

            // Zero-out the counts
            memset( counts, 0, sizeof( BucketT ) * Radix );

            T*       src   = input + offset;
            const T* start = src;
            const T* end   = start + entryCount;

            do {
                counts[(src->ykey >> shift) & mask]++;
            } while( ++src < end );

            self->CalculatePrefixSum( Radix, counts, pfxSum, totalCounts );

            while( --src >= start )
            {
                const T       value  = *src;
                const uint64  bucket = (value.ykey >> shift) & mask;

                const BucketT dstIdx = --pfxSum[bucket];
                
                output[dstIdx] = value;
            }

            std::swap( input, output );
            self->SyncThreads();
        }
    }

    //-----------------------------------------------------------
    // Convert entries from packed bits into 64-bit aligned entries
    //-----------------------------------------------------------
    template<typename TEntry>
    inline void ExpandEntries( const void* packedEntries, const uint64 inputBitOffset, TEntry* expendedEntries, const int64 entryCount )
    {
        AnonMTJob::Run( _pool, _threadCount, [=]( AnonMTJob* self ) {
            
            constexpr uint32 packedEntrySize = IsT7Out ? Info::YBitSize + Info::MapBitSize : InInfo::EntrySizePackedBits;

            int64 count, offset, end;
            GetThreadOffsets( self, entryCount, count, offset, end );
            
            const uint64 inputBitOffset = packedEntrySize * (uint64)offset;
            const size_t bitCapacity    = CDiv( packedEntrySize * (uint64)entryCount, 64 ) * 64;

            DiskFp<table,_numBuckets,IsT7Out>::ExpandEntries<TEntry>( packedEntries, inputBitOffset, bitCapacity, expendedEntries + offset, count );
        });
    }

    //-----------------------------------------------------------
    template<typename TEntry>
    inline static void ExpandEntries( const void* packedEntries, const uint64 inputBitOffset, const size_t bitCapacity,
                                      TEntry* expendedEntries, const int64 entryCount )
    {
        constexpr uint32 yBits           = IsT7Out? Info::YBitSize : InInfo::YBitSize;
        constexpr uint32 mapBits         = table == TableId::Table2 ? _k : InInfo::MapBitSize;
        constexpr uint32 metaMultipler   = IsT7Out ? 0 : InInfo::MetaMultiplier;
        constexpr uint32 packedEntrySize = IsT7Out ? yBits + mapBits : InInfo::EntrySizePackedBits;
        
        BitReader reader( (uint64*)packedEntries, bitCapacity, inputBitOffset );

              TEntry* out = expendedEntries;
        const TEntry* end = out + entryCount;
        
        for( ; out < end; out++ )
        {
            if constexpr ( table == TableId::Table2 || IsT7Out )
            {
                out->ykey = reader.ReadBits64( packedEntrySize );
            }
            else
            {
                // #TODO: Can still get ykey in a single step like above
                const uint64 y   = reader.ReadBits64( yBits   );
                const uint64 map = reader.ReadBits64( mapBits );

                out->ykey = y | ( map << yBits );

                if constexpr ( metaMultipler == 2 )
                {
                    out->meta = reader.ReadBits64( 64 );
                }
                else if constexpr ( metaMultipler == 3 )
                {
                    // #TODO: Try aligning entries to 32-bits instead.
                    out->meta.m0 = reader.ReadBits64( 64 );
                    out->meta.m1 = reader.ReadBits64( 32 );
                }
                else if constexpr ( metaMultipler == 4 )
                {
                    out->meta.m0 = reader.ReadBits64( 64 );
                    out->meta.m1 = reader.ReadBits64( 64 );
                }
            }
        }
    }
    
    //-----------------------------------------------------------
    inline void LoadBucket( const uint32 bucket )
    {
        const size_t entrySizeBits = IsT7Out ? Info::EntrySizePackedBits : InInfo::EntrySizePackedBits;
        const int32  loadTable     = IsT7Out ? (int)table : (int)table-1;

        const uint64 inBucketLength  = _context.bucketCounts[loadTable][bucket];
        const size_t bucketSizeBytes = CDiv( entrySizeBits * (uint64)inBucketLength, 64 ) * 64 / 8;

        byte* readBuffer = GetReadBufferForBucket( bucket );
        ASSERT( (size_t)readBuffer / _ioQueue.BlockSize( _inFxId ) * _ioQueue.BlockSize( _inFxId ) == (size_t)readBuffer );
        
        _ioQueue.SeekFile( _inFxId, bucket, 0, SeekOrigin::Begin );
        _ioQueue.ReadFile( _inFxId, bucket, readBuffer, bucketSizeBytes );
        _ioQueue.SignalFence( _readFence, bucket+1 );
        _ioQueue.CommitCommands();
    }

    //-----------------------------------------------------------
    inline byte* GetReadBufferForBucket( const uint32 bucket )
    {
        return (byte*)_fxRead[bucket % 2];
    }

    //-----------------------------------------------------------
    inline byte* GetWriteBuffer( const uint32 bucket )
    {
        return (byte*)_fxWrite[bucket % 2];
    }

    //-----------------------------------------------------------
    inline byte* GetPairWriteBuffer( const uint32 bucket )
    {
        return (byte*)_pairWrite[bucket % 2];
    }

    //-----------------------------------------------------------
    inline byte* GetMapWriteBuffer( const uint32 bucket )
    {
        return (byte*)_mapWrite[bucket % 2];
    }

private:
    DiskPlotContext& _context;
    ThreadPool&      _pool;
    DiskBufferQueue& _ioQueue;
    FileId           _inFxId;
    FileId           _outFxId;
    Fence            _readFence;            // #TODO: Pass these in, have them pre-created so that we don't re-create them per-table?
    Fence            _writeFence;
    Fence            _mapWriteFence;
    Fence            _pairWriteFence;
    Fence            _crossBucketFence;

    int64            _tableEntryCount = 0;  // Current table entry count
    uint64           _mapOffset       = 0;  // Offset for writing the map entries

    Duration         _ioWaitTime      = Duration::zero();

    // Working buffers, all of them have enough to hold  entries for a single bucket + cross bucket entries
    Entry*   _entries[2]  = { 0 };   // Unpacked entries   // #TODO: Create read buffers of unpacked size and then just use that as temp entries
    uint64*  _y   [2]     = { 0 };
    uint64*  _map [1]     = { 0 };
    TMetaIn* _meta[2]     = { 0 };
    Pair*    _pair[1]     = { 0 };

    void*   _fxRead [2]   = { 0 };
    void*   _fxWrite[2]   = { 0 };

    void*   _pairWrite[2] = { 0 };
    void*   _mapWrite [2] = { 0 };

    uint32  _threadCount;
    uint64* _sharedTotalBitCounts = nullptr;  // Total bucket bit sizes when writing Fx across all threads

    BitBucketWriter<_numBuckets>    _fxBitWriter;
    BitBucketWriter<1>              _pairBitWriter;

    BlockWriter<uint32>             _xWriter;       // Used for Table2  (there's no map, but just x.)
    BitBucketWriter<MapBucketCount> _mapBitWriter;  // Used after Table2

    FpCrossBucketInfo _crossBucketInfo;
};


#pragma GCC diagnostic pop

