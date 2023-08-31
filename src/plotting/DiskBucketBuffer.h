#pragma once
#include "DiskBufferBase.h"

/**
 * A disk-backed buffer which read/writes in buckets and slices. Where a slice is a a portion
 * of data that belongs to a bucket. The number of slices is equal to n_buckets * n_buckets.
 * Where each bucket has n_buckets slices.
 * The data layout can be visualized as a grid, where each cell of the grid represents a slice.
 * And depending on the manner of writing, each row or column of the grid represents a bucket.
 * The manner or writing and reading is swapped between tables. When horizontal (row-writes) are
 * performed, then column reads must subsequently be performed.
 * This is because each write consists of a row of slices, each for a different bucket. Therefore if we previously
 * wrote as a row, (full sequential write), then when we've finished writing all of the rows, all of a bucket's
 * data will be found in a column (vertically). If we write vertically, then
 * the opposite is true, and the bucket's data is found in a single row (horizontally).
 */
class DiskBucketBuffer : public DiskBufferBase
{
    DiskBucketBuffer( DiskQueue& queue, FileStream& stream, const char* name, uint32 bucketCount, size_t sliceCapacity );

public:
    static DiskBucketBuffer* Create( DiskQueue& queue, const char* fileName,
                                     uint32 bucketCount, size_t sliceCapacity,
                                     FileMode mode, FileAccess access, FileFlags flags );

    static size_t GetSingleBucketBufferSize( DiskQueue& queue, uint32 bucketCount, size_t sliceCapacity );
    static size_t GetReserveAllocSize( DiskQueue& queue, uint32 bucketCount, size_t sliceCapacity );

    virtual ~DiskBucketBuffer();

    void ReserveBuffers( IAllocator& allocator ) override;

    void Swap() override;

    /** 
     * Submit next write buffer and track the actual
     * size of each submitted slice.
    */
    // void Submit( const Span<size_t> sliceSizes );

    /**
     * Submit next write buffer w/ fixed source stride.
     * sliceStride must be <= the slice capacity.
     * It ought to be used when the slices are tracked by the
     * user separately, and it will be read with a slice override.
    */
    void Submit( size_t sliceStride );

    /**
     * Assumes the sliceStride is the same as the maximum slice capacity.
    */
    inline void Submit() { Submit( GetSliceStride() ); }

    /**
     * Read next bucket
     */
    void ReadNextBucket() override;

    inline size_t GetSliceStride() const { return _sliceCapacity; }

    inline size_t GetBucketRowStride() const { return _sliceCapacity * _bucketCount; }

    template<typename T>
    inline Span<T> GetNextWriteBufferAs()
    {
        return Span<T>( reinterpret_cast<T*>( GetNextWriteBuffer() ), GetBucketRowStride() );
    }

    template<typename T>
    inline Span<T> GetNextReadBufferAs()
    {
        size_t totalSize = 0;
        for( auto sz : _readSliceSizes[_nextReadLock] )
            totalSize += sz;

        return Span<T>( reinterpret_cast<T*>( GetNextReadBuffer() ), totalSize / sizeof( T ) );
    }

    Span<byte> PeekReadBuffer( uint32 bucket );

    void OverrideReadSlices( uint32 bucket, size_t elementSize, const uint32* sliceSizes, uint32 stride );

private:
    void HandleCommand( const DiskQueueDispatchCommand& cmd ) override;
    void CmdWriteSlices( const DiskBucketBufferCommand& cmd );
    void CmdReadSlices( const DiskBucketBufferCommand& cmd );

private:
    size_t _sliceCapacity;         // Maximum size of each slice

    bool   _verticalWrite = false;
    // size_t _writeSliceStride;      // Offset to the start of the next slices when writing
    // size_t _readSliceStride;       // Offset to the start of the next slice when reading (these are swapped between tables).

    std::vector<std::vector<size_t>> _writeSliceSizes = {};
    std::vector<std::vector<size_t>> _readSliceSizes  = {};
};
