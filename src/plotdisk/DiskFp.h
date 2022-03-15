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

public:

    //-----------------------------------------------------------
    inline DiskFp( DiskPlotContext& context )
        : _context( context )
        , _pool   ( *context.threadPool )
        , _ioQueue( *context.ioQueue )
        , _inFxId ( FileId::FX0 )
        , _outFxId( FileId::FX1 )
        , _threadCount( context.fpThreadCount )
        // , _bucket ( 0 )
    {
        _crossBucketInfo.maxBuckets = _numBuckets;
    }

    //-----------------------------------------------------------
    inline void Run()
    {
        StackAllocator alloc( _context.heapBuffer, _context.heapSize );
        AllocateBuffers( alloc, _context.tmp2BlockSize, _context.tmp1BlockSize );

        FpTable();
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
        void* fxBlocks = alloc.CAlloc( fxBlockSize  , fxBlockSize  , _numBuckets );  // Fx write
        alloc.CAlloc( pairBlockSize, pairBlockSize, _numBuckets );  // Pair write
        alloc.CAlloc( pairBlockSize, pairBlockSize, _numBuckets );  // Map write

        if( !dryRun )
        {
            _fxBitWriter = BitBucketWriter<_numBuckets>( *_context.ioQueue, _outFxId, (byte*)fxBlocks );
        }
    }

    //-----------------------------------------------------------
    inline static size_t GetRequiredHeapSize( const size_t fxBlockSize, const size_t pairBlockSize )
    {
        DiskPlotContext cx = { 0 };
        DiskFp<TableId::Table4, _numBuckets> fp( cx );

        DummyAllocator alloc;
        fp.AllocateBuffers( alloc, fxBlockSize, pairBlockSize, true );
        
        return alloc.Size();
    }

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
    }

    //-----------------------------------------------------------
    inline void FpBucket( const uint32 bucket )
    {
        Log::Line( " Bucket %u", bucket );
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

        // Write reverse-map
        if( table == TableId::Table2 )
        {
            // Write sorted x back to disk
        }
        else
        {
            // WriteReverseMap( _map[0], inEntryCount );
        }

        const int64 outEntryCount = MatchEntries( bucket, inEntryCount, y, meta, pairs );
        
        // Write pairs
        // EncodeAndWritePairs( _pairs[0], outEntryCount );

        TMetaIn* inMeta = ( table == TableId::Table2 ) ? (TMetaIn*)_map[0] : _meta[0];
        FxGen( bucket, outEntryCount, pairs, y, inMeta );

        WriteEntriesToDisk( bucket, outEntryCount, _y[1], (TMetaOut*)_meta[1], (EntryOut*)_entries[0] );

        // Save bucket length before y-sort since the pairs remain unsorted
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
    void FxGen( const uint32 bucket, const int64 entryCount, const Pair* pairs, const uint64* inY, const TMetaIn* inMeta )
    {
        using TYOut = typename FpFxGen<table>::TYOut;

        FpFxGen<table> fx( _pool, _threadCount );
        fx.ComputeFxMT( entryCount, pairs, inY, inMeta, (TYOut*)_y[1], (TMetaOut*)_meta[1] );
    }

    //-----------------------------------------------------------
    inline void WriteEntriesToDisk( const uint32 bucket, const int64 entryCount, const uint64* outY, const TMetaOut* outMeta, EntryOut* dstEntriesBuckets )
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

            const uint64* ySrc    = outY + offset;
            const uint64* ySrcEnd = outY + end;

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
                
                // #TODO: Wait for the buffer to be available first
                byte* writeBuffer = GetWriteBuffer( bucket );

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
                self->ReleaseThreads();

                // #TODO: Signal fence
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
        AnonMTJob::Run( _pool, _threadCount, [=]( AnonMTJob* self ) {

            constexpr uint32 metaMultipler = InInfo::MetaMultiplier;

            const uint32 yBits = InInfo::YBitSize;
            const uint64 yMask = 0xFFFFFFFFFFFFFFFFull >> (64-yBits);

            const uint64 bucketBits = ((uint64)bucket) << yBits;

            int64 count, offset, end;
            GetThreadOffsets( self, entryCount, count, offset, end );

            uint64*  y    = outY;
            uint64*  map  = outMap;
            TMetaIn* meta = outMeta;
            
            const Entry* entries = packedEntries;

            for( int64 i = offset; i < end; i++ )
            {
                auto& e = entries[i];

                const uint64 ykey = e.ykey;
                y  [i] = bucketBits | ( ykey & yMask );
                map[i] = ykey >> yBits;

                // Can't be 1 (table 1 only), because in that case the metadata is x, 
                // which is stored as the map
                if constexpr ( metaMultipler > 1 )
                {
                    meta[i] = e.meta;
                }
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

private:
    DiskPlotContext& _context    ;
    ThreadPool&      _pool       ;
    DiskBufferQueue& _ioQueue    ;
    FileId           _inFxId     ;
    FileId           _outFxId    ;
    Fence            _readFence  ;
    Fence            _writeFence ;
    Fence            _mapWriteFence;
    Fence            _lpWriteFence ;

    int64            _tableEntryCount;  // Current table entry count

    Duration         _readWaitTime = Duration::zero();

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

    BitBucketWriter<_numBuckets>  _fxBitWriter;

    FpCrossBucketInfo _crossBucketInfo;
    // For simplicity when doing mult-threaded bucket processing
    // int64            _bucketEntryCount;
    // int64            _lengths[BB_DP_MAX_JOBS];
    // int64            _offsets[BB_DP_MAX_JOBS];
    // int64            _ends   [BB_DP_MAX_JOBS];
    // uint32           _bucket    ;
};