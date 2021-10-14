#include "DiskBufferQueue.h"

#define NULL_BUFFER -1

DiskBufferQueue::DiskBufferQueue( const char* workDir, byte* workBuffer, size_t workBufferSize, size_t chunkSize )
    : _workDir       ( workDir        )
    , _workBuffer    ( workBuffer     )
    , _workBufferSize( workBufferSize )
    , _chunkSize     ( chunkSize      )
    , _files         (  nullptr,    0 )
{
    ASSERT( workDir    );
    ASSERT( workBuffer );
    ASSERT( chunkSize <= workBufferSize / 2 );

    const int bufferCount = (int)( workBufferSize / chunkSize );

    // Populate free list of buffers
    int* freeList = (int*)malloc( bufferCount * sizeof( int ) );

    for( int i = 0; i < bufferCount-1; i++ )
        freeList[i] = i+1;

    freeList[bufferCount-1] = NULL_BUFFER;
    _nextBuffer = 0;
    _bufferList = Span<int>( freeList, (size_t)bufferCount );
}


DiskBufferQueue::~DiskBufferQueue()
{
    free( _bufferList.values );
}


byte* DiskBufferQueue::GetBuffer()
{
    int nextIdx = _nextBuffer.load( std::memory_order_acquire );

    if( nextIdx == NULL_BUFFER )
    {
        // #TODO: Spin for a while, then suspend, waiting to be signaled that a buffer has been published.
        //        For now always spin until we get a free buffer
        do
        {
            nextIdx = _nextBuffer.load( std::memory_order_acquire );
        } while( nextIdx == NULL_BUFFER );
    }

    ASSERT( nextIdx != NULL_BUFFER );

    // Grab the buffer
    // # NOTE: This should be safe as in a single-consumer scenario.
    int newNextBuffer = _bufferList[nextIdx];
    while( !_nextBuffer.compare_exchange_weak( nextIdx, newNextBuffer,
                                                std::memory_order_release,
                                                std::memory_order_relaxed ) )
    {
        nextIdx       = _nextBuffer.load( std::memory_order_acquire );
        newNextBuffer = _bufferList[nextIdx];
    }

    return _workBuffer + (size_t)nextIdx * _chunkSize;
}