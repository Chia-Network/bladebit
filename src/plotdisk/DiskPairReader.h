#pragma once
#include "plotdisk/DiskPlotInfo.h"
#include "plotdisk/DiskPlotContext.h"
#include "util/StackAllocator.h"

template<uint32 _numBuckets>
struct DiskMapReader
{
    static constexpr uint32 _k         = _K;
    static constexpr uint32 _savedBits = bblog2( _numBuckets - 1 );
    static constexpr uint32 _mapBits   = _k + 1 + _k - _savedBits;

    //-----------------------------------------------------------
    DiskMapReader( DiskPlotContext& context, TableId table, IAllocator& allocator )
        : _context( context )
        , _table  ( table   )
    {
        const uint64 maxBucketEntries = (uint64)DiskPlotInfo<TableId::Table1, _numBuckets>::MaxBucketEntries;
        const size_t blockSize        = context.ioQueue->BlockSize( FileId::MAP2 );
        const size_t bufferSize       = CDivT( (size_t)maxBucketEntries * _mapBits, blockSize * 8 ) * blockSize;

        _loadBuffers[0] = allocator.Alloc( bufferSize, blockSize );
        _loadBuffers[1] = allocator.Alloc( bufferSize, blockSize );
        _loadBuffers[2] = allocator.Alloc( bufferSize, blockSize );
        _loadBuffers[3] = allocator.Alloc( bufferSize, blockSize );

        _unpackdMaps[0] = allocator.CAlloc<uint64>( maxBucketEntries );
        _unpackdMaps[1] = allocator.CAlloc<uint64>( maxBucketEntries );

        ASSERT( _numBuckets == context.numBuckets + 1 );
        for( uint32 i = 0; i < _numBuckets; i++ )
            _buffers[i] = _loadBuffers[i & 3]; // i & 3 == i % 4

        memcpy( _bucketCounts, context.bucketCounts[(int)table], sizeof( uint32 ) * ( _numBuckets - 1 ) );

        const uint32 overflowEntries = (uint32)( _context.entryCounts[(int)_table] - (1ull << _k ) );
        _bucketCounts[_numBuckets-1] =  overflowEntries;
        _bucketCounts[_numBuckets-2] -= overflowEntries;

    }

    //-----------------------------------------------------------
    void LoadNextEntries( const uint64 entryCount )
    {
        if( _bucketsLoaded >= _numBuckets )
            return;

        DiskBufferQueue& ioQueue = *_context.ioQueue;

        const FileId mapId           = FileId::MAP2 + (FileId)_table - 1;

        const size_t blockSize       = ioQueue.BlockSize( FileId::T1 );
        const size_t blockSizeBits   = blockSize * 8;

        uint64 bucketLength = _bucketCounts[_bucketsLoaded];
        ASSERT( bucketLength );

        // Need to load current bucket?
        if( _bucketEntryOffset == 0 )
        {   
            const size_t loadSize = CDivT( (size_t)bucketLength * _mapBits, blockSizeBits ) * blockSize;
            ioQueue.ReadFile( mapId, _bucketsLoaded, _buffers[_bucketsLoaded], loadSize );
        }

        _bucketEntryOffset += entryCount;

        // It is possible to cross 2 buckets, so we need to account for that.
        // But we ought never have to load more than 2 buckets on one go.
        while( _bucketEntryOffset > bucketLength )
        {
            _bucketEntryOffset -= bucketLength;
            _bucketsLoaded++;
            ASSERT( _bucketsLoaded < _numBuckets );

            // Upade bucket length and load the new bucket
            bucketLength = _bucketCounts[_bucketsLoaded];
            ASSERT( bucketLength );

            const size_t loadSize = CDivT( (size_t)bucketLength * _mapBits, blockSizeBits ) * blockSize;
            ioQueue.ReadFile( mapId, _bucketsLoaded, _buffers[_bucketsLoaded], loadSize );
        }
    }

    //-----------------------------------------------------------
    void ReadEntries( const uint64 entryCount, uint64* outMap )
    {
        ASSERT( _bucketsRead < _numBuckets );

        AnonMTJob::Run( *_context.threadPool, [=]( AnonMTJob* self ) {

            uint64  entriesToRead    = entryCount;
            uint64* outWriter        = outMap;
            uint64  readBucketOffset = _bucketReadOffset;
            uint32  bucketsRead      = _bucketsRead;

            while( entriesToRead )
            {
                // Do we need to unpack the buffer first?
                if( _bucketsUnpacked <= bucketsRead )
                {
                    const uint64 bucketLength = _bucketCounts[_bucketsUnpacked];
                    
                    int64 count, offset, end;
                    GetThreadOffsets( self, (int64)bucketLength, count, offset, end );
                    ASSERT( count > 0 );

                    uint64* unpackedMap = _unpackdMaps[_bucketsUnpacked & 1];
                    BitReader reader( (uint64*)_buffers[_bucketsUnpacked], _mapBits * bucketLength, offset * 8 );

                    const uint32 idxShift     = _k+1;
                    const uint64 finalIdxMask = ( 1ull << idxShift ) - 1;
                    for( int64 i = offset; i < end; i++ )
                    {
                        const uint64 packedMap = reader.ReadBits64( (uint32)_mapBits );
                        const uint64 map       = packedMap & finalIdxMask;
                        const uint32 dstIdx    = (uint32)( packedMap >> idxShift );

                        ASSERT( dstIdx < ( 1ull << _k ) / ( _numBuckets - 1 ) );
                        unpackedMap[dstIdx] = map;
                    }

                    if( self->IsControlThread() )
                    {
                        self->LockThreads();
                        _bucketsUnpacked++;
                        self->ReleaseThreads();
                    }
                    else
                        self->WaitForRelease();
                }

                const uint64  bucketLength = _bucketCounts[bucketsRead];
                const uint64  readCount    = std::min( bucketLength - readBucketOffset, entriesToRead );

                const uint64* readMap = _unpackdMaps[bucketsRead & 1];

                // Simply copy the unpacked map to the destination buffer
                uint64 count, offset, end;
                GetThreadOffsets( self, readCount, count, offset, end );

                memcpy( outWriter + offset, readMap + offset, count * sizeof( uint64 ) );

                // Update read entries
                entriesToRead -= readCount;
                outWriter     += readCount;

                readBucketOffset += readCount;
                if( readBucketOffset == bucketLength )
                {
                    readBucketOffset = 0;
                    bucketsRead++;
                }
            }

            if( self->IsControlThread() )
            {
                _bucketReadOffset = readBucketOffset;
                _bucketsRead      = bucketsRead;
            }
        });
    }

private:
    DiskPlotContext& _context;
    TableId          _table;
    uint32           _bucketsLoaded     = 0;
    uint32           _bucketsRead       = 0;
    uint32           _bucketsUnpacked   = 0;
    uint64           _bucketEntryOffset = 0;
    uint64           _bucketReadOffset  = 0;     // Entries that have actually bean read/unpacked in a bucket

    uint64*          _unpackdMaps[2];
    void*            _loadBuffers[4];            // We only actually use 4 buffers for loading. We do double-buffering,
                                                 // but since a single load may go across bucket boundaries, we need 4.
    void*            _buffers[_numBuckets];      // Keeps track of the buffers used when a particular bucket was loaded

    uint32           _bucketCounts[_numBuckets]; // We copy the bucket counts because maps have 1 more bucket for
                                                 // overflow entries.
};


// Since pairs (back-pointers) and their corresponding
// maps, have assymetric bucket counts (or not synchronized with each other),
// as the maps have been sorted on y and written back to their original buckets,
// and pairs were never sorted on y.
template<uint32 _numBuckets>
struct DiskPairAndMapReader
{
    static constexpr uint32 _k         = _K;
    static constexpr uint32 _savedBits = bblog2( _numBuckets );
    static constexpr uint32 _pairBits  = _k + 1 - _savedBits + 9;

    //-----------------------------------------------------------
    DiskPairAndMapReader( DiskPlotContext& context, Fence& fence, const TableId table, IAllocator& allocator )
        : _context  ( context )
        , _fence    ( fence   )
        , _table    ( table   )
        , _mapReader( context, table, allocator )
    {
        DiskBufferQueue& ioQueue = *_context.ioQueue;

        const uint64 maxBucketEntries = (uint64)DiskPlotInfo<TableId::Table1, _numBuckets>::MaxBucketEntries;
        const size_t blockSize        = ioQueue.BlockSize( FileId::T1 );

        const size_t bufferSize       = blockSize + CDivT( (size_t)maxBucketEntries * _pairBits, blockSize * 8 ) * blockSize;

        _pairBuffers[0] = allocator.Alloc( bufferSize, blockSize );
        _pairBuffers[1] = allocator.Alloc( bufferSize, blockSize );

        size_t prevOverflowBits = 0;
        for( uint32 i = 0; i < _numBuckets; i++ )
        {
            const size_t bucketLength           = _context.ptrTableBucketCounts[(int)_table][i];
            const size_t bucketBitSize          = bucketLength * _pairBits - prevOverflowBits;
            const size_t bucketByteSize         = CDiv( bucketBitSize, 8 );
            const size_t bucketBlockAlignedSize = CDivT( bucketByteSize, blockSize ) * blockSize;

            const size_t overflowBits = (bucketBlockAlignedSize * 8) - bucketBitSize;
            ASSERT( overflowBits < blockSize * 8 );

            _pairBucketLoadSize[i] = bucketBlockAlignedSize;
            _pairOverflowBits  [i] = (uint32)overflowBits;

            prevOverflowBits = overflowBits;
        }
    }

    //-----------------------------------------------------------
    void LoadNextBucket()
    {
        if( _bucketsLoaded >= _numBuckets )
            return;

        ASSERT( _table > TableId::Table1 );

        DiskBufferQueue& ioQueue = *_context.ioQueue;
        
        const FileId fileId    = FileId::T1 + (FileId)_table;
        const size_t blockSize = ioQueue.BlockSize( FileId::T1 );

        const uint32 bucket    = _bucketsLoaded++;
        const uint32 loadIdx   = bucket & 1; // bucket & 1 == bucket % 2

        // First block is for previous bucket's left-over bits
        void* buffer = (byte*)_pairBuffers[loadIdx] + blockSize;

        ioQueue.ReadFile( fileId, 0, buffer, _pairBucketLoadSize[bucket] );

        // Load accompanying map entries
        _mapReader.LoadNextEntries( _context.ptrTableBucketCounts[(int)_table][bucket] );

        ioQueue.SignalFence( _fence, bucket+1 );
        ioQueue.CommitCommands();
    }

    //-----------------------------------------------------------
    void UnpackBucket( const uint32 bucket, Pair* outPairs, uint64* outMap )
    {
        DiskBufferQueue& ioQueue = *_context.ioQueue;

        const uint32 loadIdx      = bucket & 1; // Same as % 2
        const size_t blockSize    = ioQueue.BlockSize( FileId::T1 );
        const size_t blockBitSize = blockSize * 8;

        _fence.Wait( bucket + 1 );

        const byte* pairBuffer = (byte*)_pairBuffers[loadIdx];

        // If we have overflow bits (bits that belong to the next bucket),
        // copy the last block we loaded to the next bucket's buffer.
        if( _pairOverflowBits[bucket] > 0 )
        {
            const size_t bucketByteSize = _pairBucketLoadSize[bucket];
            const uint32 nextLoadIdx    = ( loadIdx + 1 ) & 1;

            memcpy( _pairBuffers[nextLoadIdx], pairBuffer + bucketByteSize, blockSize );
        }
        const uint32 startBit = bucket == 0 ? (uint32)blockBitSize : (uint32)(blockBitSize - _pairOverflowBits[bucket-1] );
        ASSERT( startBit <= blockSize*8 );

        const size_t fullBitSize = _pairBucketLoadSize[bucket] * 8 + blockBitSize - startBit;
        
        const int64 bucketLength = (int64)_context.ptrTableBucketCounts[(int)_table][bucket];

        AnonMTJob::Run( *_context.threadPool, [=]( AnonMTJob* self ) {
            

            int64 count, offset, end;
            GetThreadOffsets( self, bucketLength, count, offset, end );

            const size_t bitOffset = startBit + (size_t)offset * _pairBits;
            BitReader reader( (uint64*)pairBuffer, fullBitSize, bitOffset );

            // const uint64 pairOffset = _entriesLoaded;
            const uint32 lBits  = _k - _savedBits + 1;
            const uint32 rBits  = 9;

            for( int64 i = offset; i < end; i++ )
            {
                const uint32 left  = /*pairOffset+*/ (uint32)reader.ReadBits64( lBits );
                const uint32 right = left +  (uint32)reader.ReadBits64( rBits );

                outPairs[i] = { .left = left, .right = right };
            }
        });

        // #TODO: Try not to start another threaded job, and instead do it in the same MT job?
        _mapReader.ReadEntries( (uint64)bucketLength, outMap );
    }

private:
    DiskPlotContext& _context;
    Fence&           _fence;

    DiskMapReader<_numBuckets+1> _mapReader;

    void*            _pairBuffers       [2];
    uint32           _pairOverflowBits  [_numBuckets];
    uint64           _pairBucketLoadSize[_numBuckets];

    TableId          _table;
    uint32           _bucketsLoaded = 0;
    uint64           _entriesLoaded = 0;
};


