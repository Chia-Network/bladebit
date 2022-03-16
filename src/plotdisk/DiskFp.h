#pragma once
#include "util/Util.h"
#include "plotdisk/DiskPlotConfig.h"
#include "plotdisk/DiskBufferQueue.h"
#include "plotdisk/DiskPlotContext.h"
#include "plotdisk/FpGroupMatcher.h"
#include "plotdisk/FpFxGen.h"
#include "DiskPlotInfo.h"
#include "BitBucketWriter.h"
#include "util/StackAllocator.h"


template<TableId table>
struct FpMapType { using Type = uint64; };

template<>
struct FpMapType<TableId::Table2> { using Type = uint32; };

template<TableId table, uint32 _numBuckets>
class DiskFp
{
public:
    using InInfo = DiskPlotInfo<table-1, _numBuckets>;
    using Info   = DiskPlotInfo<table, _numBuckets>;

    static constexpr uint32 _k               = Info::_k;
    static constexpr uint64 MaxBucketEntries = Info::MaxBucketEntries;

    using Entry    = FpEntry<table-1>;
    using EntryOut = FpEntry<table>;
    using TMetaIn  = typename TableMetaType<table>::MetaIn;
    using TMetaOut = typename TableMetaType<table>::MetaOut;
    using TAddress = uint64;
    using TYOut    = typename FpFxGen<table>::TYOut;

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
    }

    //-----------------------------------------------------------
    inline void AllocateBuffers( IAllocator& alloc, const size_t fxBlockSize, const size_t pairBlockSize, bool dryRun = false )
    {
        using info = DiskPlotInfo<TableId::Table4, _numBuckets>;
        
        const uint32 bucketBits    = bblog2( _numBuckets );
        const size_t maxEntries    = info::MaxBucketEntries;
        const size_t maxEntriesX   = maxEntries + BB_DP_CROSS_BUCKET_MAX_ENTRIES;
        const size_t entrySizeBits = info::EntrySizePackedBits;
        
        const size_t genEntriesPerBucket  = (size_t)( CDivT( maxEntries, (size_t)_numBuckets ) * BB_DP_XTRA_ENTRIES_PER_BUCKET );
        const size_t perBucketEntriesSize = RoundUpToNextBoundaryT( CDiv( entrySizeBits * genEntriesPerBucket, 8 ), fxBlockSize ) + 
                                            RoundUpToNextBoundaryT( CDiv( entrySizeBits * BB_DP_CROSS_BUCKET_MAX_ENTRIES, 8 ), fxBlockSize );

        // Working buffers
        _entries[0] = alloc.CAlloc<Entry>( maxEntries );
        _entries[1] = alloc.CAlloc<Entry>( maxEntries );

        _y[0]    = alloc.CAlloc<uint64>( maxEntriesX );
        _y[1]    = alloc.CAlloc<uint64>( maxEntriesX );
        
        _map[0]  = alloc.CAlloc<uint64>( maxEntriesX );

        _meta[0] = (TMetaIn*)alloc.CAlloc<Meta4>( maxEntriesX );
        _meta[1] = (TMetaIn*)alloc.CAlloc<Meta4>( maxEntriesX );
        
        _pair[0] = alloc.CAlloc<Pair> ( maxEntriesX );
        
        // IO buffers
        const size_t pairBits    = _k + 1 - bucketBits + 9;
        const size_t mapBits     = _k + 1 - bucketBits + _k + 1;

        const size_t fxWriteSize   = (size_t)_numBuckets * perBucketEntriesSize;
        const size_t pairWriteSize = CDiv( ( maxEntriesX ) * pairBits, 8 );
        const size_t mapWriteSize  = CDiv( ( maxEntriesX ) * mapBits , 8 );

        // #TODO: Set actual buffers
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
        byte* xBlocks    = (byte*)alloc.CAlloc( 1          , pairBlockSize, pairBlockSize );  // x write
        byte* mapBlocks  = (byte*)alloc.CAlloc( _numBuckets, pairBlockSize, pairBlockSize );  // Map write

        if( !dryRun )
        {
            const FileId mapWriterId = FileId::MAP2 + (FileId)table-2; // Writes previous buffer's key as a map

            _fxBitWriter   = BitBucketWriter<_numBuckets>( _ioQueue, _outFxId, (byte*)fxBlocks );
            _pairBitWriter = BitBucketWriter<1>          ( _ioQueue, FileId::T1 + (FileId)table, (byte*)pairBlocks );
            _xBitWriter    = BitBucketWriter<1>          ( _ioQueue, FileId::T1,  (byte*)xBlocks );
            _mapBitWriter  = BitBucketWriter<_numBuckets>( _ioQueue, mapWriterId, (byte*)mapBlocks );
        }
    }

    //-----------------------------------------------------------
    inline static size_t GetRequiredHeapSize( const size_t fxBlockSize, const size_t pairBlockSize )
    {
        DiskPlotContext cx = { 0 };
        DiskFp<TableId::Table4, _numBuckets> fp( cx, FileId::None, FileId::None );

        DummyAllocator alloc;
        fp.AllocateBuffers( alloc, fxBlockSize, pairBlockSize, true );
        
        return alloc.Size();
    }

    //-----------------------------------------------------------
    inline Duration ReadWaitTime() const { return _readWaitTime; }
    
    //-----------------------------------------------------------
    inline Duration WriteWaitTime() const { return _writeWaitTime; }

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

            _readFence.Wait( bucket + 1, _readWaitTime );
            FpBucket( bucket );
        }

        _fxBitWriter  .SubmitLeftOvers();
        _pairBitWriter.SubmitLeftOvers();
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

        // Expand entries to be 64-bit aligned, sort them, then unpack them to individual components
        ExpandEntries( packedEntries, 0, _entries[0], inEntryCount );
        SortEntries( _entries[0], _entries[1], inEntryCount );
        UnpackEntries( bucket, _entries[0], inEntryCount, y, _map[0], meta );

        WriteMap( bucket, inEntryCount, _map[0], (uint64*)_meta[1] );

        const TMetaIn* inMeta = ( table == TableId::Table2 ) ? (TMetaIn*)_map[0] : meta;

        TYOut*    outY    = (TYOut*)_y[1];
        TMetaOut* outMeta = (TMetaOut*)_meta[1];

        // Match
        const int64 outEntryCount = MatchEntries( bucket, inEntryCount, y, inMeta, pairs );

        const int64 crossMatchCount  = _crossBucketInfo.matchCount;
        const int64 writeEntrtyCount = outEntryCount + crossMatchCount;

        if( bucket > 0 )
            ProcessCrossBucketEntries( bucket - 1, pairs - crossMatchCount, outY, outMeta );
        
        WritePairsToDisk( bucket, writeEntrtyCount, pairs - crossMatchCount );

        FxGen( bucket, outEntryCount, pairs, y, inMeta, outY + crossMatchCount, outMeta + crossMatchCount );

        WriteEntriesToDisk( bucket, writeEntrtyCount, outY, outMeta, (EntryOut*)_entries[0] );

        // Save bucket length before y-sort since the pairs remain unsorted
        _context.ptrTableBucketCounts[(int)table][bucket] += (uint64)writeEntrtyCount;

        _tableEntryCount += writeEntrtyCount;
        _crossBucketInfo.matchOffset = (uint64)outEntryCount;   // Set the offset for the next cross-bucket entries
        _crossBucketInfo.matchCount  = 0;
    }

    //-----------------------------------------------------------
    void WriteMap( const uint32 bucket, const int64 entryCount, const uint64* map, uint64* outMap )
    {
        using TMap = typename FpMapType<table>::Type;
        ASSERT( entryCount < (1ll << _k) );
        
        if constexpr ( table == TableId::Table2 )
        {
            byte* writeBuffer = GetMapWriteBuffer( bucket );
            
            if( bucket > 1 )
                _mapWriteFence.Wait( bucket - 2, _writeWaitTime );

             AnonMTJob::Run( _pool, _threadCount, [=]( AnonMTJob* self ) {

                int64 count, offset, end;
                GetThreadOffsets( self, entryCount, count, offset, end );

                const TMap* inMap  = ((TMap*)map) + offset;
                      TMap* outMap = ((TMap*)writeBuffer) + offset;
                
                // Just copy to the write buffer
                memcpy( outMap, inMap, sizeof( TMap ) * count );

                // Write sorted x back to disk
                self->SyncThreads();
                if( self->IsControlThread() )
                {
                    const uint64 totalBits = entryCount * sizeof( TMap ) * 8;

                    _xBitWriter.BeginWriteBuckets( &totalBits, writeBuffer );
                    _xBitWriter.Submit();
                    _ioQueue.SignalFence( _mapWriteFence, bucket );
                    _ioQueue.CommitCommands();
                }
            });

            return;
        }

        // Write the key as a map to the final entry location.
        // We do this by writing into buckets to final sorted (current) location
        // into its original bucket, with the offset in that bucket.
        using Job = AnonPrefixSumJob<uint32>;

        uint32 totalCounts   [_numBuckets];
        uint64 totalBitCounts[_numBuckets];
        
        Job::Run( _pool, _threadCount, [&]( Job* self ) {

            const uint32 shift       = Info::MapBitSize - Info::BucketBits;
            const uint32 bitSize     = Info::MapBitSize + Info::MapBitSize - Info::BucketBits;
            const uint32 encodeShift = Info::MapBitSize;

            int64 count, offset, end;
            GetThreadOffsets( self, entryCount, count, offset, end );

            uint32 counts        [_numBuckets] = { 0 };
            uint32 pfxSum        [_numBuckets];
            // uint64 totalBitCounts[_numBuckets];
            
            const uint64* inIdx    = map + offset;
            const uint64* inIdxEnd = map + end;

            // Count buckets
            do {
                const uint64 b = *inIdx >> shift;
                ASSERT( b < _numBuckets );
                counts[b]++;
            } while( ++inIdx < inIdxEnd );

            self->CalculatePrefixSum( _numBuckets, counts, pfxSum, totalCounts );

            // Convert map entries from source inded to reverse map
            const uint64 tableOffset = (uint64)( _tableEntryCount + offset );
            
            const uint64* reverseMap    = map + offset;
                  uint64* outMapBuckets = outMap; 

            for( int64 i = 0; i < count; i++ )
            {
                const uint64 m = reverseMap[i];
                const uint64 b = m >> shift;

                const uint32 dstIdx = --pfxSum[b];
                ASSERT( (int64)dstIdx < entryCount );

                const uint64 finalIdx = (uint64)(tableOffset + i);
                outMapBuckets[dstIdx] = (m << encodeShift) | finalIdx;
            }

            auto& bitWriter = _mapBitWriter;
            uint64* bitCounts = totalBitCounts;

            if( self->IsControlThread() )
            {
                self->LockThreads();

                // Convert counts to bit sizes
                for( uint32 i = 0; i < _numBuckets; i++ )
                    bitCounts[i] = (uint64)totalCounts[i] * bitSize;
                
                byte* writeBuffer = GetMapWriteBuffer( bucket );
                
                // Wait for the buffer to be available first
                if( bucket > 1 )
                    _mapWriteFence.Wait( bucket - 2, _writeWaitTime );

                bitWriter.BeginWriteBuckets( bitCounts, writeBuffer );

                self->ReleaseThreads();
            }
            else
                self->WaitForRelease();

            // Bit-compress/pack each bucket entries
            uint64 bitsWritten = 0;

            for( uint32 i = 0; i < _numBuckets; i++ )
            {
                if( counts[i] < 1 )
                    continue;

                const uint64 writeOffset = pfxSum[i];
                const uint64 bitOffset   = writeOffset * bitSize - bitsWritten;
                bitsWritten += bitCounts[i];
                
                ASSERT( bitOffset + counts[i] * bitSize <= bitCounts[i] );

                BitWriter writer = bitWriter.GetWriter( i, bitOffset );
                
                const uint64* mapToWrite    = outMapBuckets + writeOffset; 
                const uint64* mapToWriteEnd = mapToWrite + counts[i]; 
                
                // Compress a couple of entries first, so that we don't get any simultaneaous writes to the same fields
                const uint64* mapToWriteEndPass1 = mapToWrite + std::min( counts[i], 2u ); 

                while( mapToWrite < mapToWriteEndPass1 )
                {
                    writer.Write( *mapToWrite, bitSize );
                    mapToWrite++;
                }

                self->SyncThreads();

                while( mapToWrite < mapToWriteEnd )
                {
                    writer.Write( *mapToWrite, bitSize );
                    mapToWrite++;
                }
            }

            // Write to disk
            self->SyncThreads();
            if( self->IsControlThread() )
            {
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

        // FxGen( bucket, info.matchCount, info.pair, info.y, (TMetaIn*)info.meta, outY, outMeta );
        FpFxGen<table>::ComputeFx( (int64)info.matchCount, info.pair, info.y, (TMetaIn*)info.meta, outY, outMeta, 0 );
        
        const uint32 matchOffset = (uint32)info.matchOffset;
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

        _pairWriteFence.Wait( bucket, _writeWaitTime );
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
                    _writeFence.Wait( bucket - 2, _writeWaitTime );

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
    inline void UnpackEntries( const uint32 bucket, const Entry* packedEntries, const int64 entryCount, uint64* outY, uint64* outMap, TMetaIn* outMeta )
    {
        using TMap = typename FpMapType<table>::Type;

        AnonMTJob::Run( _pool, _threadCount, [=]( AnonMTJob* self ) {

            constexpr uint32 metaMultipler = InInfo::MetaMultiplier;

            const uint32 yBits = InInfo::YBitSize;
            const uint64 yMask = 0xFFFFFFFFFFFFFFFFull >> (64-yBits);

            const uint64 bucketBits = ((uint64)bucket) << yBits;

            int64 count, offset, end;
            GetThreadOffsets( self, entryCount, count, offset, end );

            uint64*  y    = outY;
            TMap*    map  = (TMap*)outMap;
            TMetaIn* meta = outMeta;
            
            const Entry* entries = packedEntries;

            for( int64 i = offset; i < end; i++ )
            {
                auto& e = entries[i];

                const uint64 ykey = e.ykey;
                y  [i] = bucketBits | ( ykey & yMask );
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
    inline void SortEntries( Entry* entries, Entry* tmpEntries, const int64 entryCount )
    {
        AnonPrefixSumJob<uint32>::Run( _pool, _threadCount, [=]( AnonPrefixSumJob<uint32>* self ) {
            
            int64 count, offset, end;
            GetThreadOffsets( self, entryCount, count, offset, end );

            const uint32 remainderBits = _k - InInfo::YBitSize;
            EntrySort( self, count, offset, entries, tmpEntries, remainderBits );
        });
    }

    //-----------------------------------------------------------
    template<typename T, typename TJob, typename BucketT>
    inline static void EntrySort( PrefixSumJob<TJob, BucketT>* self, const int64 entryCount, const int64 offset, 
                                  T* entries, T* tmpEntries, const uint32 remainderBits )
    {
        ASSERT( self );
        ASSERT( entries );
        ASSERT( tmpEntries );
        ASSERT( entryCount > 0 );

        constexpr uint Radix = 256;

        constexpr int32  iterations = 4;
        constexpr uint32 shiftBase  = 8;

        BucketT counts     [Radix];
        BucketT pfxSum     [Radix];
        BucketT totalCounts[Radix];

        uint32 shift = 0;
        T* input  = entries;
        T* output = tmpEntries;

        const uint32 lastByteMask = 0xFF >> remainderBits;
        uint32 masks[iterations]  = { 0xFF, 0xFF, 0xFF, lastByteMask };

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
    inline void ExpandEntries( const void* packedEntries, const uint64 inputBitOffset, Entry* expendedEntries, const int64 entryCount )
    {
        AnonMTJob::Run( _pool, _threadCount, [=]( AnonMTJob* self ) {
            
            constexpr uint32 packedEntrySize = InInfo::EntrySizePackedBits;

            int64 count, offset, end;
            GetThreadOffsets( self, entryCount, count, offset, end );
            
            const uint64 inputBitOffset = packedEntrySize * (uint64)offset;
            const size_t bitCapacity    = CDiv( packedEntrySize * (uint64)entryCount, 64 ) * 64;

            DiskFp<table,_numBuckets>::ExpandEntries( packedEntries, inputBitOffset, bitCapacity, expendedEntries + offset, count );
        });
    }

    //-----------------------------------------------------------
    inline static void ExpandEntries( const void* packedEntries, const uint64 inputBitOffset, const size_t bitCapacity,
                                      Entry* expendedEntries, const int64 entryCount )
    {
        constexpr uint32 yBits           = InInfo::YBitSize;
        // const     uint64 yMask           = ( 1ull << yBits ) - 1;
        constexpr uint32 mapBits         = table == TableId::Table2 ? _k : InInfo::MapBitSize;
        // const     uint64 mapMask         = ( ( 1ull << mapBits) - 1 ) << yBits;
        constexpr uint32 metaMultipler   = InInfo::MetaMultiplier;
        constexpr uint32 packedEntrySize = InInfo::EntrySizePackedBits;
        
        BitReader reader( (uint64*)packedEntries, bitCapacity, inputBitOffset ); //CDiv( (uint64)entryCount * packedEntrySize, 64 ) * 64, inputBitOffset );

              Entry* out = expendedEntries;
        const Entry* end = out + entryCount;
        
        for( ; out < end; out++ )
        {
            if constexpr ( table == TableId::Table2 )
            {
                // const uint64 y = reader.ReadBits64( yBits + mapBits );
                // out->ykey = ( y & yMask ) | ( ( y & mapMask ) << yBits );
                out->ykey = reader.ReadBits64( packedEntrySize );
            }
            else
            {
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
        const uint64 inBucketLength  = _context.bucketCounts[(int)table-1][bucket];
        const size_t bucketSizeBytes = CDiv( InInfo::EntrySizePackedBits * (uint64)inBucketLength, 64 ) * 64 / 8;

        byte* readBuffer = GetReadBufferForBucket( bucket );
        _ioQueue.SeekFile( _inFxId, bucket, 0, SeekOrigin::Begin );
        _ioQueue.ReadFile( _inFxId, bucket, readBuffer, bucketSizeBytes );
        
        // We need to read a little bit from the next bucket as well, so we can do proper group matching across buckets
        if( bucket+1 < _numBuckets )
        {
            const size_t nextBucketLoadSize = CDiv( InInfo::EntrySizePackedBits * (uint64)BB_DP_CROSS_BUCKET_MAX_ENTRIES, 64 ) * 64 / 8;
            
            _ioQueue.SeekFile( _inFxId, bucket+1, 0, SeekOrigin::Begin );
            _ioQueue.ReadFile( _inFxId, bucket+1, readBuffer + bucketSizeBytes, nextBucketLoadSize );
        }
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
    DiskPlotContext& _context    ;
    ThreadPool&      _pool       ;
    DiskBufferQueue& _ioQueue    ;
    FileId           _inFxId     ;
    FileId           _outFxId    ;
    Fence            _readFence  ;      // #TODO: Pass these in, have them pre-created so that we don't re-create them per-table
    Fence            _writeFence ;
    Fence            _mapWriteFence;
    Fence            _pairWriteFence;
    Fence            _crossBucketFence;

    int64            _tableEntryCount = 0;  // Current table entry count

    Duration         _readWaitTime  = Duration::zero();
    Duration         _writeWaitTime = Duration::zero();

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

    BitBucketWriter<_numBuckets> _fxBitWriter;
    BitBucketWriter<1>           _pairBitWriter;
    BitBucketWriter<_numBuckets> _mapBitWriter;
    BitBucketWriter<1>           _xBitWriter;

    FpCrossBucketInfo _crossBucketInfo;
};