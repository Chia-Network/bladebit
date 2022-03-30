#pragma once
#include "plotdisk/DiskPlotInfo.h"
#include "plotdisk/DiskPlotContext.h"
#include "util/StackAllocator.h"
#include "util/BitView.h"

/// #NOTE: We actually have _numBuckets+1 because there's an
//         implicit overflow bucket that may contain entries.
template<uint32 _numBuckets>
struct DiskMapReader
{
    static constexpr uint32 _k         = _K;
    static constexpr uint32 _savedBits = bblog2( _numBuckets );
    static constexpr uint32 _mapBits   = _k + 1 + _k - _savedBits;

    //-----------------------------------------------------------
    DiskMapReader( DiskPlotContext& context, const uint32 threadCount, const TableId table, IAllocator& allocator )
        : _context    ( context     )
        , _table      ( table       )
        , _threadCount( threadCount )
    {
        const uint64 maxKEntries      = ( 1ull << _k );
        const uint64 maxBucketEntries = maxKEntries / _numBuckets;
        const size_t blockSize        = context.ioQueue->BlockSize( FileId::MAP2 );
        const size_t bufferSize       = CDivT( (size_t)maxBucketEntries * _mapBits, blockSize * 8 ) * blockSize;

        _loadBuffers[0] = allocator.Alloc( bufferSize, blockSize );
        _loadBuffers[1] = allocator.Alloc( bufferSize, blockSize );
        _loadBuffers[2] = allocator.Alloc( bufferSize, blockSize );
        _loadBuffers[3] = allocator.Alloc( bufferSize, blockSize );

        _unpackdMaps[0] = allocator.CAlloc<uint64>( maxBucketEntries );
        _unpackdMaps[1] = allocator.CAlloc<uint64>( maxBucketEntries );

        ASSERT( _numBuckets == context.numBuckets );
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

        uint64 bucketLength = GetBucketLength( _bucketsLoaded );
        ASSERT( bucketLength );

        // Need to load current bucket?
        if( _bucketEntryOffset == 0 )
        {   
            const size_t loadSize = CDivT( (size_t)bucketLength * _mapBits, blockSizeBits ) * blockSize;
            ioQueue.ReadFile( mapId, _bucketsLoaded, GetBucketBuffer( _bucketsLoaded ), loadSize );
            // _fence.Signal( _bucketsLoaded + 1 );
        }

        _bucketEntryOffset += entryCount;

        // It is possible to cross 2 buckets, so we need to account for that.
        // But we ought never have to load more than 2 buckets on one go.
        while( _bucketEntryOffset > bucketLength )
        {
            _bucketEntryOffset -= bucketLength;
            _bucketsLoaded++;
            ASSERT( _bucketsLoaded <= _numBuckets );

            // Upade bucket length and load the new bucket
            bucketLength = GetBucketLength( _bucketsLoaded );
            ASSERT( bucketLength );

            const size_t loadSize = CDivT( (size_t)bucketLength * _mapBits, blockSizeBits ) * blockSize;
            ioQueue.ReadFile( mapId, _bucketsLoaded, GetBucketBuffer( _bucketsLoaded ), loadSize );
            // _fence.Signal( _bucketsLoaded + 1 );
        }
    }

    //-----------------------------------------------------------
    void ReadEntries( const uint64 entryCount, uint64* outMap )
    {
        ASSERT( _bucketsRead <= _numBuckets );

        // #TODO: Check here if we have to unpack a bucket and then check our own fence.

        AnonMTJob::Run( *_context.threadPool, _threadCount, [=]( AnonMTJob* self ) {

            uint64  entriesToRead    = entryCount;
            uint64* outWriter        = outMap;
            uint64  readBucketOffset = _bucketReadOffset;
            uint32  bucketsRead      = _bucketsRead;

            while( entriesToRead )
            {
                // Do we need to unpack the buffer first?
                if( _bucketsUnpacked <= bucketsRead )
                {
                    const uint64 bucketLength = GetBucketLength( _bucketsUnpacked );
                    
                    int64 count, offset, end;
                    GetThreadOffsets( self, (int64)bucketLength, count, offset, end );
                    ASSERT( count > 0 );

                    uint64* unpackedMap = _unpackdMaps[_bucketsUnpacked & 1];
                    BitReader reader( (uint64*)GetBucketBuffer( _bucketsUnpacked ), _mapBits * bucketLength, offset * _mapBits );

                    const uint32 idxShift     = _k+1;
                    const uint64 finalIdxMask = ( 1ull << idxShift ) - 1;

                    for( int64 i = offset; i < end; i++ )
                    {
                        const uint64 packedMap = reader.ReadBits64( (uint32)_mapBits );
                        const uint64 map       = packedMap & finalIdxMask;
                        const uint32 dstIdx    = (uint32)( packedMap >> idxShift );

                        ASSERT( dstIdx < bucketLength );
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

                const uint64  bucketLength = GetBucketLength( bucketsRead );
                const uint64  readCount    = std::min( bucketLength - readBucketOffset, entriesToRead );

                const uint64* readMap      = _unpackdMaps[bucketsRead & 1] + readBucketOffset;

                // Simply copy the unpacked map to the destination buffer
                uint64 count, offset, end;
                GetThreadOffsets( self, readCount, count, offset, end );

                if( count )
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
                self->LockThreads();
                _bucketReadOffset = readBucketOffset;
                _bucketsRead      = bucketsRead;
                self->ReleaseThreads();
            }
            else
                self->WaitForRelease();
        });
    }

private:

    //-----------------------------------------------------------
    inline uint64 GetBucketLength( const uint32 bucket )
    {
        const uint64 maxKEntries      = ( 1ull << _k );
        const uint64 maxBucketEntries = maxKEntries / _numBuckets;

        // All buckets before the last bucket (and before the overflow bucket) have the same entry count which is 2^k / numBuckets
        if( bucket < _numBuckets - 1 )
            return maxBucketEntries;
        
        const uint64 tableEntryCount = _context.entryCounts[(int)_table];

        // Last, non-overflow bucket?
        if( bucket == _numBuckets - 1 )
            return tableEntryCount > maxKEntries ? 
                     maxBucketEntries :
                     maxBucketEntries - ( maxKEntries - tableEntryCount );
        
        // Last bucket
        return tableEntryCount > maxKEntries ? tableEntryCount - maxKEntries : 0;
    }

    //-----------------------------------------------------------
    inline void* GetBucketBuffer( const uint32 bucket )
    {
        return _loadBuffers[bucket & 3];
    }

private:
    DiskPlotContext& _context;
    TableId          _table;
    uint32           _threadCount       = 0;
    uint32           _bucketsLoaded     = 0;
    uint32           _bucketsRead       = 0;
    uint32           _bucketsUnpacked   = 0;
    uint64           _bucketEntryOffset = 0;
    uint64           _bucketReadOffset  = 0;     // Entries that have actually bean read/unpacked in a bucket

    uint64*          _unpackdMaps[2];
    void*            _loadBuffers[4];            // We do double-buffering, but since a single load may go across bucket boundaries, we need 4.
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
    DiskPairAndMapReader( DiskPlotContext& context, const uint32 threadCount, Fence& fence, const TableId table, IAllocator& allocator, bool noMap )
        : _context    ( context )
        , _fence      ( fence   )
        , _table      ( table   )
        , _threadCount( threadCount )
        , _mapReader  ( context, threadCount, table, allocator )
        , _noMap      ( noMap )
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
        if( !_noMap )
            _mapReader.LoadNextEntries( _context.ptrTableBucketCounts[(int)_table][bucket] );

        ioQueue.SignalFence( _fence, bucket+1 );
        ioQueue.CommitCommands();
    }

    //-----------------------------------------------------------
    uint64 UnpackBucket( const uint32 bucket, Pair* outPairs, uint64* outMap )
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

        AnonMTJob::Run( *_context.threadPool, _threadCount, [=]( AnonMTJob* self ) {
            

            int64 count, offset, end;
            GetThreadOffsets( self, bucketLength, count, offset, end );

            const size_t bitOffset = startBit + (size_t)offset * _pairBits;
            BitReader reader( (uint64*)pairBuffer, fullBitSize, bitOffset );

            // const uint64 pairOffset = _entriesLoaded;
            const uint32 lBits  = _k - _savedBits + 1;
            const uint32 rBits  = 9;

            for( int64 i = offset; i < end; i++ )
            {
                const uint32 left  = (uint32)reader.ReadBits64( lBits );
                const uint32 right = left +  (uint32)reader.ReadBits64( rBits );

                outPairs[i] = { .left = left, .right = right };
            }
        });

        // #TODO: Try not to start another threaded job, and instead do it in the same MT job?
        if( !_noMap )
            _mapReader.ReadEntries( (uint64)bucketLength, outMap );

        return (uint64)bucketLength;
    }

private:
    DiskPlotContext& _context;
    Fence&           _fence;

    DiskMapReader<_numBuckets> _mapReader;

    void*            _pairBuffers       [2];
    uint32           _pairOverflowBits  [_numBuckets];
    uint64           _pairBucketLoadSize[_numBuckets];

    TableId          _table;
    uint32           _threadCount;
    uint32           _bucketsLoaded = 0;
    uint64           _entriesLoaded = 0;
    bool             _noMap;
};



/// Reads T-sized elements with fs block size alignment.
/// Utility class used to hide block alignment stuff from the user.
/// Loads are double-buffered and meant to be used as if laoding a whole bucket.
/// After a maximum of 2 LoadEntries are called, 
/// ReadEntries must be used afterwards/ to read that buffer and make it available again.
/// There's no check for this at runtime in release mode, so it must be used correctly by the user.
template<typename T>
class BlockReader
{
public:
    
    //-----------------------------------------------------------
    BlockReader( const FileId fileId, DiskBufferQueue* ioQueue, const uint64 maxLength, 
                 IAllocator& allocator, const size_t blockSize )
        : _fileId( fileId )
        , _entriesPerBlock( blockSize / sizeof( T ) )
        , _ioQueue( ioQueue )
    {
        // #TODO: Check that sizeof( T ) is power of 2
        ASSERT( _entriesPerBlock * sizeof( T ) == blockSize );

        const size_t allocCount = RoundUpToNextBoundaryT( maxLength, _entriesPerBlock );
        _loadBuffer[0] = allocator.CAlloc<T>( allocCount, blockSize );
        _loadBuffer[1] = allocator.CAlloc<T>( allocCount, blockSize );
    }

    //-----------------------------------------------------------
    BlockReader()
    {}

    // #NOTE: User is responsible for using a Fence after this call
    //-----------------------------------------------------------
    void LoadEntries( uint64 count )
    {
        ASSERT( _loadIdx >= _readIdx );
        ASSERT( _loadIdx - _readIdx <= 2 );
        ASSERT( count );

        const uint32 loadIdx    = _loadIdx & 1; // % 2
        const uint32 preloadIdx = _loadIdx & 3; // % 4


        // Substract entries that are already loaded
        if( _preloadedEntries[preloadIdx] )
        {
            ASSERT( count >= _preloadedEntries[preloadIdx] ); // #TODO: Support count < _preloadedEntries
                                                              // This would mean continually passing left-over entries forward to the next buffers
                                                              // #TODO: Refactor this. Way too confusing.
                                                              // Maybe just only allow 1 left over?
            count -=  std::min( count, _preloadedEntries[preloadIdx] );

            if( count == 0 )
            {
                ASSERT( 0 );    // #TODO:: Handle this as per above
            }
        }

        uint64 loadCount = count;
        if( count )
        {
            T* buffer = _loadBuffer[loadIdx] + _entriesPerBlock;

            loadCount = RoundUpToNextBoundaryT( count, _entriesPerBlock );

            _ioQueue->ReadFile( _fileId, 0, buffer, loadCount * sizeof( uint32 ) );
            _ioQueue->CommitCommands();
        }

        _loadCount[loadIdx] = count;

        // Save overflow count for the next load
        _loadIdx++;
        _preloadedEntries[_loadIdx & 3] = loadCount - count;
    }

    //-----------------------------------------------------------
    T* ReadEntries()
    {
        const uint32 readIdx = _readIdx & 1;
        
        T* buffer = _loadBuffer[readIdx];

        // Set the offset if we have any overflow entries
        const uint64 loadCount = RoundUpToNextBoundaryT( _loadCount[readIdx], _entriesPerBlock );
        const uint64 overflow  = loadCount - _loadCount[readIdx];

        // Copy any overflow entries we may have loaded to the next buffer
        if( overflow )
        {
            T* dst = _loadBuffer[(readIdx + 1) & 1] + _entriesPerBlock - overflow;
            memcpy( dst, buffer + loadCount - overflow, overflow * sizeof( T ) );
        }

        const uint64 offset = GetPreloadedEntryCount();
        buffer += _entriesPerBlock - offset;

        // #TODO: Copy over more from preloaded if our preload

        _readIdx++;
        return buffer;
    }

private:
    //-----------------------------------------------------------
    inline uint64 GetPreloadedEntryCount()
    {
        return _preloadedEntries[_readIdx & 3];  // % 4
    }

private:
    FileId           _fileId              = FileId::None;
    uint64           _entriesPerBlock     = 0;
    DiskBufferQueue* _ioQueue             = nullptr;
    T*               _loadBuffer      [2];
    uint64           _loadCount       [2] = { 0 };
    uint64           _preloadedEntries[4] = { 0 };  // Overflow entries loaded by the previous load (due to alignment), kept for the next load
                                                    //  We use 4-sized array for this so that each load can safely store the read count without overwrites.
    //uint64         _copyCount[2]                  // How many trailing entries to copy to the next buffer after load
    uint64           _overflowEntries  = 0;         
    uint32           _loadIdx          = 0;
    uint32           _readIdx          = 0;
};

