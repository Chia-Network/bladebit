#pragma once
#include "plotdisk/DiskPlotInfo.h"
#include "plotdisk/DiskPlotContext.h"
#include "util/StackAllocator.h"
#include "util/BitView.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"

/// #NOTE: We actually have _numBuckets+1 because there's an
//         implicit overflow bucket that may contain entries.
template<typename TMap, uint32 _numBuckets, uint32 _finalIdxBits>
struct DiskMapReader
{
    static constexpr uint32 _k         = _K;
    static constexpr uint32 _savedBits = bblog2( _numBuckets );
    static constexpr uint32 _mapBits   = _k - _savedBits + _finalIdxBits; // ( origin address | final address )

    //-----------------------------------------------------------
    DiskMapReader() {}

    //-----------------------------------------------------------
    DiskMapReader( DiskPlotContext& context, const uint32 threadCount, const TableId table, const FileId fileId, 
                   IAllocator& allocator, const uint64* realBucketLengths = nullptr )
        : _context    ( &context    )
        , _table      ( table       )
        , _fileId     ( fileId      )
        , _threadCount( threadCount )
    {
        const size_t blockSize        = context.ioQueue->BlockSize( fileId );
        const uint64 maxKEntries      = ( 1ull << _k );
        const uint64 maxBucketEntries = maxKEntries / _numBuckets;
        
        Allocate(  allocator, blockSize );

        if( realBucketLengths )
            memcpy( _bucketLengths, realBucketLengths, sizeof( _bucketLengths ) );
        else
        {
            for( uint32 b = 0; b < _numBuckets-1; b++ )
                _bucketLengths[b] = maxBucketEntries;
            
            _bucketLengths[_numBuckets-1] = GetVirtualBucketLength( _numBuckets-1 );
            _bucketLengths[_numBuckets]   = GetVirtualBucketLength( _numBuckets   );
        }

        ASSERT( _numBuckets == context.numBuckets );
    }

    // Allocation-checker dummy
    //-----------------------------------------------------------
    DiskMapReader( IAllocator& allocator, const size_t blockSize ) 
    {
        Allocate( allocator, blockSize );
    }

    //-----------------------------------------------------------
    void LoadNextEntries( const uint64 entryCount )
    {
        if( _bucketsLoaded >= _numBuckets )
            return;

        DiskBufferQueue& ioQueue = *_context->ioQueue;

        const size_t blockSize     = ioQueue.BlockSize( _fileId );
        const size_t blockSizeBits = blockSize * 8;

        uint64 bucketLength = GetVirtualBucketLength( _bucketsLoaded );
        ASSERT( bucketLength );

        // Need to load current bucket?
        if( _bucketEntryOffset == 0 )
        {
            // Load length (actual bucket length) might be different than the virtual length
            const size_t loadSize = CDivT( (size_t)_bucketLengths[_bucketsLoaded] * _mapBits, blockSizeBits ) * blockSize;
            ioQueue.ReadFile( _fileId, _bucketsLoaded, GetBucketBuffer( _bucketsLoaded ), loadSize );
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
            bucketLength = GetVirtualBucketLength( _bucketsLoaded );
            ASSERT( bucketLength );

            const size_t loadSize = CDivT( (size_t)_bucketLengths[_bucketsLoaded] * _mapBits, blockSizeBits ) * blockSize;
            ioQueue.ReadFile( _fileId, _bucketsLoaded, GetBucketBuffer( _bucketsLoaded ), loadSize );
            // _fence.Signal( _bucketsLoaded + 1 );
        }
    }

    //-----------------------------------------------------------
    void ReadEntries( const uint64 entryCount, TMap* outMap )
    {
        ASSERT( _bucketsRead <= _numBuckets );

        // #TODO: Check here if we have to unpack a bucket and then check our own fence.
        AnonMTJob::Run( *_context->threadPool, _threadCount, [=]( AnonMTJob* self ) {

            uint64  entriesToRead    = entryCount;
            TMap*   outWriter        = outMap;
            uint64  readBucketOffset = _bucketReadOffset;
            uint32  bucketsRead      = _bucketsRead;

            while( entriesToRead )
            {
                // Unpack the whole bucket into it's destination indices
                if( _bucketsUnpacked <= bucketsRead )
                {
                    #if _DEBUG
                        const uint64 virtBucketLength = GetVirtualBucketLength( _bucketsUnpacked );
                    #endif

                    const uint64 bucketLength = _bucketLengths[_bucketsUnpacked];
                    
                    int64 count, offset, end;
                    GetThreadOffsets( self, (int64)bucketLength, count, offset, end );
                    ASSERT( count > 0 );

                    TMap* unpackedMap = _unpackdMaps[_bucketsUnpacked & 1];
                    BitReader reader( (uint64*)GetBucketBuffer( _bucketsUnpacked ), _mapBits * bucketLength, offset * _mapBits );

                    const uint32 idxShift     = _finalIdxBits;
                    const uint64 finalIdxMask = ( 1ull << idxShift ) - 1;

                    for( int64 i = offset; i < end; i++ )
                    {
                        const uint64 packedMap = reader.ReadBits64( (uint32)_mapBits );
                        const uint64 map       = packedMap & finalIdxMask;
                        const uint64 dstIdx    = packedMap >> idxShift;

                        ASSERT( dstIdx < virtBucketLength );
                        unpackedMap[dstIdx] = (TMap)map;
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

                const uint64 bucketLength = GetVirtualBucketLength( bucketsRead );
                const uint64 readCount    = std::min( bucketLength - readBucketOffset, entriesToRead );

                const TMap* readMap      = _unpackdMaps[bucketsRead & 1] + readBucketOffset;

                // Simply copy the unpacked map to the destination buffer
                int64 count, offset, end;
                GetThreadOffsets( self, (int64)readCount, count, offset, end );

                if( count )
                    memcpy( outWriter + offset, readMap + offset, (size_t)count * sizeof( TMap ) );

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

    //-----------------------------------------------------------
    inline uint64 GetVirtualBucketLength( const uint32 bucket )
    {
        const uint64 maxKEntries      = ( 1ull << _k );
        const uint64 maxBucketEntries = maxKEntries / _numBuckets;

        // All buckets before the last bucket (and before the overflow bucket) have the same entry count which is 2^k / numBuckets
        if( bucket < _numBuckets - 1 )
            return maxBucketEntries;
        
        const uint64 tableEntryCount = _context->entryCounts[(int)_table];

        // Last, non-overflow bucket?
        if( bucket == _numBuckets - 1 )
            return tableEntryCount > maxKEntries ? 
                     maxBucketEntries :
                     maxBucketEntries - ( maxKEntries - tableEntryCount );
        
        // Last bucket
        return tableEntryCount > maxKEntries ? tableEntryCount - maxKEntries : 0;
    }

private:
    //-----------------------------------------------------------
    inline void Allocate( IAllocator& allocator, const size_t blockSize )
    {
        const uint64 maxKEntries      = ( 1ull << _k );
        const uint64 maxBucketEntries = maxKEntries / _numBuckets;
        const size_t bufferSize       = CDivT( (size_t)maxBucketEntries * _mapBits, blockSize * 8 ) * blockSize;

        _loadBuffers[0] = allocator.Alloc( bufferSize, blockSize );
        _loadBuffers[1] = allocator.Alloc( bufferSize, blockSize );
        _loadBuffers[2] = allocator.Alloc( bufferSize, blockSize );
        _loadBuffers[3] = allocator.Alloc( bufferSize, blockSize );

        _unpackdMaps[0] = allocator.CAlloc<TMap>( maxBucketEntries );
        _unpackdMaps[1] = allocator.CAlloc<TMap>( maxBucketEntries );
    }    

    //-----------------------------------------------------------
    inline void* GetBucketBuffer( const uint32 bucket )
    {
        return _loadBuffers[bucket & 3];
    }

private:
    DiskPlotContext* _context           = nullptr;
    TableId          _table             = TableId::Table1;
    FileId           _fileId            = FileId::None;
    uint32           _threadCount       = 0;
    uint32           _bucketsLoaded     = 0;
    uint32           _bucketsRead       = 0;
    uint32           _bucketsUnpacked   = 0;
    uint64           _bucketEntryOffset = 0;
    uint64           _bucketReadOffset  = 0;        // Entries that have actually bean read/unpacked in a bucket

    TMap*            _unpackdMaps[2];
    void*            _loadBuffers[4];               // We do double-buffering, but since a single load may go across bucket boundaries, we need 4.
    uint64           _bucketLengths[_numBuckets+1]; // Real, non-virtual bucket lengths. This is only necessary for buckets that have been pruned (ex. in Phase 3)
};


// Since pairs (back-pointers) and their corresponding
// maps, have assymetric bucket counts (or not synchronized with each other),
// as the maps have been sorted on y and written back to their original buckets,
// and pairs were never sorted on y.
template<uint32 _numBuckets, bool _bounded = false>
struct DiskPairAndMapReader
{
    static constexpr uint32 _extraBuckets = _bounded ? 0 : 1;
    static constexpr uint32 _k            = _K;
    static constexpr uint32 _savedBits    = bblog2( _numBuckets );
    static constexpr uint32 _lBits        = _k + 1 - _savedBits;
    static constexpr uint32 _rBits        = 9;
    static constexpr uint32 _pairBits     = _lBits + _rBits;
    
    using MapReader = DiskMapReader<uint64, _numBuckets, _k + _extraBuckets>;

    // #TODO: Don't do this nonesense, just forget about not having nullables and just use pointers...
    //-----------------------------------------------------------
    DiskPairAndMapReader() {}

    //-----------------------------------------------------------
    DiskPairAndMapReader( DiskPlotContext& context, const uint32 threadCount, Fence& fence, const TableId table, IAllocator& allocator, bool noMap )
        : _context    ( &context )
        , _fence      ( &fence   )
        , _table      ( table   )
        , _threadCount( threadCount )
        , _mapReader  ( context, threadCount, table, FileId::MAP2 + (FileId)table - 1, allocator )
        , _noMap      ( noMap )
    {
        DiskBufferQueue& ioQueue = *_context->ioQueue;

        const size_t blockSize = ioQueue.BlockSize( FileId::T1 + (FileId)table );

        Allocate( allocator, blockSize );

        size_t prevOverflowBits = 0;
        for( uint32 i = 0; i < _numBuckets; i++ )
        {
            const size_t bucketLength           = _context->ptrTableBucketCounts[(int)_table][i];
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

    // Allocation size-check dummy
    //-----------------------------------------------------------
    DiskPairAndMapReader( IAllocator& allocator, const size_t blockSize )
    {
        Allocate( allocator, blockSize );
        MapReader reader( allocator, blockSize );
    }

    //-----------------------------------------------------------
    void LoadNextBucket()
    {
        if( _bucketsLoaded >= _numBuckets )
        {
            return;
        }

        ASSERT( _table > TableId::Table1 );

        DiskBufferQueue& ioQueue = *_context->ioQueue;
        
        const FileId fileId    = FileId::T1 + (FileId)_table;
        const size_t blockSize = ioQueue.BlockSize( FileId::T1 + (FileId)_table);

        const uint32 bucket    = _bucketsLoaded++;
        const uint32 loadIdx   = bucket & 1; // bucket & 1 == bucket % 2

        // First block is for previous bucket's left-over bits
        void* buffer = (byte*)_pairBuffers[loadIdx] + blockSize;

        ioQueue.ReadFile( fileId, 0, buffer, _pairBucketLoadSize[bucket] );

        // Load accompanying map entries
        if( !_noMap )
            _mapReader.LoadNextEntries( _context->ptrTableBucketCounts[(int)_table][bucket] );

        ioQueue.SignalFence( *_fence, bucket+1 );
        ioQueue.CommitCommands();
    }

    //-----------------------------------------------------------
    uint64 UnpackBucket( const uint32 bucket, Pair* outPairs, uint64* outMap, Duration& ioWait )
    {
        DiskBufferQueue& ioQueue = *_context->ioQueue;

        const uint32 loadIdx      = bucket & 1; // Same as % 2
        const size_t blockSize    = ioQueue.BlockSize( FileId::T1 + (FileId)_table );
        const size_t blockBitSize = blockSize * 8;

        _fence->Wait( bucket + 1, ioWait );

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
        
        const int64 bucketLength = (int64)_context->ptrTableBucketCounts[(int)_table][bucket];

        AnonMTJob::Run( *_context->threadPool, _threadCount, [=]( AnonMTJob* self ) {
            
            Pair* pairs = outPairs;

            int64 count, offset, end;
            GetThreadOffsets( self, bucketLength, count, offset, end );

            const size_t bitOffset = startBit + (size_t)offset * _pairBits;
            BitReader reader( (uint64*)pairBuffer, fullBitSize, bitOffset );

            for( int64 i = offset; i < end; i++ )
            {
                Pair pair;
                pair.left  = (uint32)reader.ReadBits64( _lBits );
                pair.right = pair.left +  (uint32)reader.ReadBits64( _rBits );

                pairs[i] = pair;
            }
        });

        // #TODO: Try not to start another threaded job, and instead do it in the same MT job?
        if( !_noMap )
            _mapReader.ReadEntries( (uint64)bucketLength, outMap );

        return (uint64)bucketLength;
    }

private:
    //-----------------------------------------------------------
    inline void Allocate( IAllocator& allocator, const size_t blockSize )
    {
        const uint64 maxBucketEntries = (uint64)DiskPlotInfo<TableId::Table1, _numBuckets>::MaxBucketEntries;
        const size_t bufferSize       = blockSize + CDivT( (size_t)maxBucketEntries * _pairBits, blockSize * 8 ) * blockSize;

        _pairBuffers[0] = allocator.Alloc( bufferSize, blockSize );
        _pairBuffers[1] = allocator.Alloc( bufferSize, blockSize );        
    }

private:
    DiskPlotContext*  _context = nullptr;
    Fence*            _fence   = nullptr;

    MapReader         _mapReader;

    void*            _pairBuffers       [2];
    uint32           _pairOverflowBits  [_numBuckets];
    uint64           _pairBucketLoadSize[_numBuckets];

    uint64           _entriesLoaded = 0;
    uint32           _bucketsLoaded = 0;
    uint32           _threadCount   = 0;
    TableId          _table         = (TableId)0;
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
    BlockReader() {}

    //-----------------------------------------------------------
    BlockReader( const FileId fileId, DiskBufferQueue* ioQueue, const uint64 maxLength, 
                 IAllocator& allocator, const size_t blockSize, const uint64 retainOffset )
        : _fileId         ( fileId )
        , _entriesPerBlock( blockSize / sizeof( T ) )
        , _ioQueue        ( ioQueue )
    {
        // #TODO: Check that sizeof( T ) is power of 2
        ASSERT( _entriesPerBlock * sizeof( T ) == blockSize );
        Allocate( allocator, blockSize, maxLength, retainOffset );
    }

    // Size-check dummy
    //-----------------------------------------------------------
    BlockReader( IAllocator& allocator, const size_t blockSize, const uint64 maxLength, const uint64 retainOffset )
        : _entriesPerBlock( blockSize / sizeof( T ) )
    {
        Allocate( allocator, blockSize, maxLength, retainOffset );
    }

    // #NOTE: User is responsible for using a Fence after this call
    //-----------------------------------------------------------
    void LoadEntries( uint64 count )
    {
        ASSERT( _loadIdx >= _readIdx );
        ASSERT( _loadIdx - _readIdx <= 2 );
        ASSERT( count );

        const uint32 loadIdx    = _loadIdx & 1; // % 2
        const uint32 preloadIdx = _loadIdx & 3; // % 4

        // Substract entries that are already pre-loaded
        const uint64 numPreloaded = _preloadedEntries[preloadIdx];

        if( numPreloaded )
        {
            count -= std::min( count, numPreloaded );

            if( count == 0 )
            {
                // #TODO: Test this scenario, have not hit it during testing.
                // <= than our preloaded entries were requested.
                // Move any left-over preloaded entries to the next load
                _preloadedEntries[(_loadIdx+1) & 3] = numPreloaded - count;
                _copyCount[loadIdx]                 = numPreloaded - count;
                _loadCount[loadIdx]                 = 0;
                return;
            }
        }

        T* buffer = _loadBuffer[loadIdx] + _entriesPerBlock;

        const uint64 loadCount = RoundUpToNextBoundaryT( count, _entriesPerBlock );

        _ioQueue->ReadFile( _fileId, 0, buffer, loadCount * sizeof( T ) );
        _ioQueue->CommitCommands();

        const uint64 overflowCount = loadCount - count;

        _loadCount[loadIdx] = loadCount;
        _copyCount[loadIdx] = overflowCount;

        // Save overflow as preloaded entries for the next load
        _loadIdx++;
        _preloadedEntries[_loadIdx & 3] = overflowCount;
    }

    //-----------------------------------------------------------
    T* ReadEntries()
    {
        const uint32 readIdx = _readIdx & 1;

        const uint64 copyCount   = _copyCount[readIdx];
        const uint64 loadedCount = _loadCount[readIdx];
        const uint64 preloaded   = GetPreloadedEntryCount();

        T* buffer = _loadBuffer[readIdx] + _entriesPerBlock - preloaded;
        
        // Copy any overflow entries we may have loaded to the next buffer
        if( copyCount )
        {
            const uint64 entryCount = preloaded + loadedCount;
            T* dst = _loadBuffer[(readIdx + 1) & 1] + _entriesPerBlock - copyCount;
            memcpy( dst, buffer + entryCount - copyCount, copyCount * sizeof( T ) );
        }

        _readIdx++;
        return buffer;
    }

private:
    //-----------------------------------------------------------
    inline void Allocate( IAllocator& allocator, const size_t blockSize, const uint64 maxLength, const uint64 retainOffset )
    {
        const size_t prefixZoneCount = RoundUpToNextBoundaryT( retainOffset, _entriesPerBlock );

        // Add another retain offset here because we space for retained entries at the start and at the end
        const size_t allocCount = prefixZoneCount + RoundUpToNextBoundaryT( _entriesPerBlock + maxLength + retainOffset, _entriesPerBlock );

        _loadBuffer[0] = allocator.CAlloc<T>( allocCount, blockSize ) + prefixZoneCount;
        _loadBuffer[1] = allocator.CAlloc<T>( allocCount, blockSize ) + prefixZoneCount;
    }

    //-----------------------------------------------------------
    inline uint64 GetPreloadedEntryCount()
    {
        return _preloadedEntries[_readIdx & 3];  // % 4
    }

private:
    FileId           _fileId              = FileId::None;
    uint64           _entriesPerBlock     = 0;
    DiskBufferQueue* _ioQueue             = nullptr;
    T*               _loadBuffer      [2] = { nullptr };
    uint64           _loadCount       [2] = { 0 };  // How many entries actually loaded from disk
    uint64           _copyCount       [2] = { 0 };  // Trailing entries to copy over to the next buffer
    uint64           _preloadedEntries[4] = { 0 };  // Overflow entries already loaded by the previous load (due to alignment), kept for the next load
                                                    //  We use 4-sized array for this so that each load can safely store the read count without overwrites.
    uint32           _loadIdx          = 0;
    uint32           _readIdx          = 0;
};

template<typename T>
class IP3LMapReader
{
public:
    virtual void LoadNextBucket() = 0;
    virtual T*   ReadLoadedBucket() = 0;
};

/// Utility for reading maps for P3 where the left table buckets
/// have to load more entries than the bucket has in order to allow
/// the pairs to point correctly to cross-bucket entries.
/// The extra entries loaded from the next bucket have to be carried over
/// to the start of the bucket of the next load.
template<uint32 _numBuckets, uint64 _retainCount, typename T>
class SingleFileMapReader : public IP3LMapReader<T>
{
public:
    //-----------------------------------------------------------
    SingleFileMapReader() {}

    //-----------------------------------------------------------
    SingleFileMapReader( const FileId fileId, DiskBufferQueue* ioQueue, IAllocator& allocator, 
                         const uint64 maxLength, const size_t blockSize,
                         const uint32 bucketCounts[_numBuckets] )
        : _reader( fileId, ioQueue, maxLength, allocator, blockSize, _retainCount )
    {
        memcpy( _bucketCounts, bucketCounts, sizeof( _bucketCounts ) );
    }

    // Size-check dummy
    //-----------------------------------------------------------
    SingleFileMapReader( IAllocator& allocator, const size_t blockSize, const uint64 maxLength )
        : _reader( allocator, blockSize, maxLength, _retainCount )
    {}

    //-----------------------------------------------------------
    void LoadNextBucket() override
    {
        ASSERT( _bucketsLoaded < _numBuckets );
        
        const uint64 bucketLength = _bucketCounts[_bucketsLoaded];
        const uint64 loadCount    = _bucketsLoaded == 0 ? bucketLength + _retainCount :
                                    _bucketsLoaded == _numBuckets - 1 ? bucketLength - _retainCount :
                                    bucketLength;

        _reader.LoadEntries( loadCount );
        _bucketsLoaded++;
    }

    //-----------------------------------------------------------
    T* ReadLoadedBucket() override
    {
        T* entries    = _reader.ReadEntries();
        T* dstEntries = entries;

        if( _bucketsRead > 0 )
        {
            // Copy over the retained entries from the last buffer
            dstEntries -= _retainCount;
            memcpy( dstEntries, _retainedEntries, _retainCount * sizeof( uint32 ) );
        }

        if( _bucketsRead < _numBuckets - 1 )
        {
            // Retain our last entries for the next buffer
            const uint64 entryCount = _bucketCounts[_bucketsRead] + ( _bucketsRead == 0 ? _retainCount : 0 );

            memcpy( _retainedEntries, entries + entryCount - _retainCount, _retainCount * sizeof( uint32 ) );
        }

        _bucketsRead++;
        return dstEntries;
    }

private:
    BlockReader<T> _reader;
    uint32         _bucketsLoaded = 0;
    uint32         _bucketsRead   = 0;
    uint32         _bucketCounts   [_numBuckets ];
    T              _retainedEntries[_retainCount];
};


#pragma GCC diagnostic pop

