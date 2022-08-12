#pragma once
#include "DiskBufferQueue.h"
#include "util/StackAllocator.h"

template<typename T>
class BlockWriter
{
public:
    //-----------------------------------------------------------
    inline BlockWriter() {}

    //-----------------------------------------------------------
    inline BlockWriter( IAllocator& allocator, const FileId fileId, Fence& fence, const size_t blockSize, const size_t elementCount )
        : _fileId   ( fileId )
        , _fence    ( &fence )
        , _blockSize( (uint32)blockSize )
    {
        static_assert( sizeof( T ) == 1 || ( sizeof( T ) & 1) == 0 );

        const size_t allocSize = blockSize + RoundUpToNextBoundaryT( elementCount * sizeof( T ), blockSize );

        // Sanity check, should never happen as block sizes are power of 2 and
        // we don't expect to use this with non Pow2 elements.
        FatalIf( blockSize / sizeof( T ) * sizeof( T ) != blockSize, "Unexpected block size." );

        _buffers[0] = allocator.AllocT<T>( allocSize, blockSize );
        _buffers[1] = allocator.AllocT<T>( allocSize, blockSize );

        _remainder = allocator.AllocT<T>( blockSize );
    }

    //-----------------------------------------------------------
    inline T* GetNextBuffer( Duration& writeWaitTime )
    {
        ASSERT( _buffers[0] );

        if( _bufferIdx > 1 )
            _fence->Wait( _bufferIdx - 2, writeWaitTime );

        return _buffers[_bufferIdx & 1] + _remainderCount;
    }

    //-----------------------------------------------------------
    inline void SubmitBuffer( DiskBufferQueue& queue, const size_t elementCount )
    {
        ASSERT( elementCount );

        const size_t blockSize        = _blockSize;
        const size_t elementsPerBlock = blockSize / sizeof( T );

        T* buf = _buffers[_bufferIdx & 1];

        if( _remainderCount )
            bbmemcpy_t<T>( buf, _remainder, _remainderCount );

        const size_t totalElements = _remainderCount + elementCount;
        const size_t writeCount    = totalElements / elementsPerBlock * elementsPerBlock;
        _remainderCount = totalElements - writeCount;

        if( _remainderCount )
            bbmemcpy_t<T>( _remainder, buf + writeCount, _remainderCount );

        queue.WriteFile( _fileId, 0, buf, writeCount * sizeof( T ) );
        queue.SignalFence( *_fence, _bufferIdx++ );
        queue.CommitCommands();
    }

    //-----------------------------------------------------------
    inline void SubmitFinalBlock( DiskBufferQueue& queue )
    {
        if( _remainderCount )
        {
            queue.WriteFile( _fileId, 0, _remainder, queue.BlockSize( _fileId ) );
            _remainderCount = 0;
        }

        queue.SignalFence( *_fence, _bufferIdx );
        queue.CommitCommands();
    }

    //-----------------------------------------------------------
    inline void WaitForFinalWrite()
    {
        _fence->Wait( _bufferIdx );
    }

private:
    FileId _fileId         = FileId::None;
    int32  _bufferIdx      = 0;
    uint32 _blockSize      = 1;
    T*     _buffers[2]     = { nullptr };
    T*     _remainder      = nullptr; 
    size_t _remainderCount = 0;             // Number of elements saved in our remainder buffer
    Fence* _fence          = nullptr;
};
