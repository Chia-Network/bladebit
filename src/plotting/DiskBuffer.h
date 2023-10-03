#pragma once
#include "DiskBufferBase.h"

/**
 * Sequential disk buffer that whose actions are dispatched on a DiskQueue.
 * This performs block-aligned reads and writes.
 */
class DiskBuffer : public DiskBufferBase
{
    DiskBuffer( DiskQueue& queue, FileStream& stream, const char* name, uint32 bucketCount, size_t bufferSize );

public:
    static DiskBuffer* Create( DiskQueue& queue, const char* fileName,
                               uint32 bucketCount, size_t bufferSize,
                               FileMode mode, FileAccess access, FileFlags flags );

    virtual ~DiskBuffer();

    void ReserveBuffers( IAllocator& allocator ) override;

    static size_t GetReserveAllocSize( DiskQueue& queue, size_t bufferSize ); 

    inline size_t GetAlignedBufferSize() const
    {
        return _alignedBufferSize;
    }

    void ReadNextBucket() override;
    void Swap() override;

    void Submit( size_t size );

    template<typename T>
    inline Span<T> GetNextWriteBufferAs()
    {
        return Span<T>( reinterpret_cast<T*>( GetNextWriteBuffer() ), GetAlignedBufferSize() );
    }

    template<typename T>
    inline Span<T> GetNextReadBufferAs()
    {
        return Span<T>( reinterpret_cast<T*>( GetNextReadBuffer() ), GetAlignedBufferSize() / sizeof( T ) );
    }

protected:
    void HandleCommand( const DiskQueueDispatchCommand& cmd ) override;

private:
    /// Command handlers
    void CmdWrite( const DiskBufferCommand& cmd );
    void CmdRead( const DiskBufferCommand& cmd );

private:
    size_t _bufferSize;                 // Requested buffer size
    size_t _alignedBufferSize;          // Block-aligned requested buffer size

    std::vector<size_t> _bucketSizes;   // The actual (unaligned) size of each bucket.
};
