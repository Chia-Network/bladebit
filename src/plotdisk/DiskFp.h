#pragma once
#include "util/Util.h"
#include "plotdisk/DiskPlotConfig.h"
#include "plotdisk/DiskBufferQueue.h"
#include "plotdisk/DiskPlotContext.h"
#include "util/BitView.h"
#include "DiskPlotInfo.h"

template<uint32 _numBuckets>
class IOBitWriter
{
    uint64*          _buffers;
    size_t           _size;                   // Size per buffer 
    uint32           _count;                  // How many buffers we have. Must have least 2 buffers
    uint32           _curBuffer         = 0;
    uint64           _remainderField    = 0;
    uint32           _remainderBitCount = 0;
    DiskBufferQueue* _queue;
};

template<TableId table, uint32 _numBuckets>
class DiskFp
{
public:
    using InInfo = DiskPlotInfo<table-1, _numBuckets>;
    using Info   = DiskPlotInfo<table, _numBuckets>;

    static constexpr uint32 _k = Info::_k;

    using Entry    = FpEntry<table>;
    using TMeta    = typename TableMetaType<table>::MetaOut;
    using TAddress = uint64;

public:

    //-----------------------------------------------------------
    inline DiskFp( DiskPlotContext& context )
        : _context( context )
        , _ioQueue( *context.ioQueue )
        , _inFxId ( FileId::FX0 )
        , _outFxId( FileId::FX1 )
        // , _bucket ( 0 )
    {

    }

    //-----------------------------------------------------------
    inline void Run()
    {

    }

private:
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
        const int64 inEntryCount = (int64)_context.bucketCounts[(int)table-1][bucket];
        const bool  isLastBucket = bucket + 1 == _numBuckets;

        const byte* packedEntries = GetReadBufferForBucket( bucket );

        // Expand entries to be 64-bit aligned, sort them, then unpack them to individual components
        ExpandEntries( packedEntries, _entries, inEntryCount );
        SortEntries( _entries, inEntryCount );
        UnpackEntries( bucket, _entries, inEntryCount, _y[0], _map[0], _meta[0] );
        
        // Expand cross bucket entries
        const int64 crossBucketEntryCount = isLastBucket ? 0 : BB_DP_CROSS_BUCKET_MAX_ENTRIES;
        if( crossBucketEntryCount )
        {
            const size_t packedEntrySize = InInfo::EntrySizePackedBits;
            const size_t bitCapacity     = RoundUpToNextBoundary( crossBucketEntryCount * packedEntrySize, 64 );

            ExpandEntries( packedEntries, 0, bitCapacity, _entries + inEntryCount, crossBucketEntryCount );
            SortEntries( _entries+inEntryCount, crossBucketEntryCount );
            UnpackEntries( bucket + 1, _entries + inEntryCount, crossBucketEntryCount, 
                           _y[0] + inEntryCount, _map[0] + inEntryCount, _meta[0] + inEntryCount );
        }

        // Write reverse-map
        if( table == TableId::Table2 )
        {
            // Write sorted x back to disk
        }
        else
        {
            // WriteReverseMap( _map[0], inEntryCount );
        }

        // Match groups
        // const int64 outEntryCount = MatchEntries( _y[0], _pairs[0], inEntryCount + crossBucketEntryCount );

        // Write pairs
        // EncodeAndWritePairs( _pairs[0], outEntryCount );

        // FxGen
        TMeta* inMeta = ( table == TableId::Table2 ) ? (TMeta*)_map[0] : _meta[0];
        // FxGen( outEntryCount, _pairs[0], _y[0], _y[1], _meta[0], _meta[1] );

        // Write new entries to disk
        // WriteEntriesToDisk( outEntryCount, _y[1], _meta[1] );

        // Save bucket length before y-sort since the pairs remain unsorted
    }

    //-----------------------------------------------------------
    inline void UnpackEntries( const uint32 bucket, const Entry* packedEntries, const int64 entryCount, uint64* outY, uint64* outMap, TMeta* outMeta )
    {
        AnonMTJob::Run( _pool, _fxThreadCount, [=]( AnonMTJob* self ) {

            constexpr uint32 metaMultipler = InInfo::MetaMultiplier;

            const uint32 yBits = InInfo::YBitSize;
            const uint64 yMask = 0xFFFFFFFFFFFFFFFFull >> (64-yBits);

            const uint64 bucketBits = ((uint64)bucket) << yBits;

            int64 count, offset, end;
            GetThreadOffsets( self, entryCount, count, offset, end );

            uint64* y    = outY;
            uint64* map  = outMap;
            TMeta*  meta = outMeta;
            
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
                    outMap[i] = e.meta;
                }
            }
        });
    }

    //-----------------------------------------------------------
    inline void SortEntries( Entry* entries, Entry* tmpEntries, const int64 entryCount )
    {
        AnonPrefixSumJob<uint32>::Run( _pool, _fxThreadCount, [=]( AnonPrefixSumJob<uint32>* self ) {
            
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
        ASSERT( entryCount > 0);

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
        AnonMTJob::Run( _pool, _fxThreadCount, [=]( AnonMTJob* self ) {
            
            constexpr uint32 packedEntrySize = InInfo::EntrySizePackedBits;

            int64 count, offset, end;
            GetThreadOffsets( self, entryCount, count, offset, end );
            
            const uint64 inputBitOffset = packedEntrySize * (uint64)offset;
            const size_t bitCapacity    = CDiv( packedEntrySize * (uint64)entryCount, 64 ) * 64;

            ExpandEntries( packedEntries, inputBitOffset, bitCapacity, expendedEntries + offset, count );
        });
    }

    //-----------------------------------------------------------
    inline static void ExpandEntries( const void* packedEntries, const uint64 inputBitOffset, const size_t bitCapacity,
                                      Entry* expendedEntries, const int64 entryCount )
    {
        constexpr uint32 yBits           = InInfo::YBitSize;
        constexpr uint32 mapBits         = InInfo::MapBits;
        constexpr uint32 metaMultipler   = InInfo::MetaMultiplier;
        constexpr uint32 packedEntrySize = InInfo::EntrySizePackedBits;
        
        BitReader reader( (uint64*)packedEntries, CDiv( (uint64)entryCount * packedEntrySize, 64 ) * 64, inputBitOffset );

              Entry* out = expendedEntries;
        const Entry* end = out + entryCount;
        
        for( ; out < end; out++ )
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
        return _readBuffers[bucket % 2];
    }

private:
    DiskPlotContext& _context    ;
    ThreadPool&      _pool       ;
    DiskBufferQueue& _ioQueue    ;
    FileId           _inFxId     ;
    FileId           _outFxId    ;
    byte**           _readBuffers;
    Fence            _readFence ;
    Fence            _writeFence;

    Duration         _readWaitTime = Duration::zero();

    // Working buffers, all of them have enough to hold  entries for a single bucket + cross bucket entries
    Entry*  _entries;   // Unpacked entries
    uint64* _y   [2];
    uint64* _map [1];
    TMeta*  _meta[2];
    Pair*   _pair[2];

    uint32  _fxThreadCount = 1;
    // For simplicity when doing mult-threaded bucket processing
    // int64            _bucketEntryCount;
    // int64            _lengths[BB_DP_MAX_JOBS];
    // int64            _offsets[BB_DP_MAX_JOBS];
    // int64            _ends   [BB_DP_MAX_JOBS];
    // uint32           _bucket    ;
};