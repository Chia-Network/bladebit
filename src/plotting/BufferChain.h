#pragma once
#include "threading/Fence.h"
#include "util/Span.h"

class IAllocator;

/// Maintains a chain of buffers which is to be used (and re-used) with first-out, first-in semantics.
///  #NOTE: The caller is expected to free the buffers, as we don't own them, just use them.
class BufferChain
{
    BufferChain( uint32 bufferCount, size_t bufferSize );

public:
    ~BufferChain();

    static BufferChain* Create( IAllocator& allocator, uint32 bufferCount, size_t bufferSize, size_t bufferAlignment, bool dryRun );

    /// Get the pointer to a buffer that will be used for a certain index
    /// without actually waiting for it to be available.
    byte* PeekBuffer( uint32 index );

    /// Blocks calling thread until the next buffer in the chain
    /// is ready for use, and returns it.
    byte* GetNextBuffer();

    /// Releases the earliest locked buffer
    void ReleaseNextBuffer();

    /// Blocks the calling thread until all outstanding buffers have been released,
    /// and resets its state to the first buffer index again.
    void Reset();

    inline size_t BufferSize() const { return _bufferSize; }

    inline uint32 BufferCount() const { return (uint32)_buffers.Length(); }

private:
    Fence        _fence;
    Span<byte*>  _buffers;
    IAllocator*  _allocator           = nullptr;
    size_t       _bufferSize          = 0;  // Size of each individual buffer
    uint32       _nextBufferToLock    = 0;
    uint32       _nextBufferToRelease = 0;
};
