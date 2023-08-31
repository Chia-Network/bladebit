#include "BufferChain.h"
#include "util/IAllocator.h"

BufferChain::BufferChain( uint32 bufferCount, size_t bufferSize )
    : _buffers    ( new byte*[bufferCount], bufferCount )
    , _bufferSize ( bufferSize )
{}

BufferChain::~BufferChain()
{
    delete[] _buffers.Ptr();
}

BufferChain* BufferChain::Create( IAllocator& allocator, uint32 bufferCount, size_t bufferSize, size_t bufferAlignment, bool dryRun )
{
    PanicIf( !bufferSize, "" );
    PanicIf( !bufferCount, "" );
    PanicIf( !bufferAlignment, "" );

    BufferChain* self = nullptr;
    if( !dryRun )
        self = new BufferChain( bufferCount, bufferSize );

    for( uint32 i = 0; i < bufferCount; i++ )
    {
        byte* buffer = allocator.AllocT<byte>( bufferSize, bufferAlignment );

        if( !dryRun )
            self->_buffers[i] = buffer;
    }

    return self;
}

byte* BufferChain::PeekBuffer( const uint32 index )
{
    return _buffers[index % (uint32)_buffers.Length()];
}

byte* BufferChain::GetNextBuffer()
{
    const uint32 bufferCount = (uint32)_buffers.Length();

    PanicIf( _nextBufferToRelease > _nextBufferToLock, "" );
    PanicIf( _nextBufferToLock - _nextBufferToRelease > bufferCount, "" );

    if( _nextBufferToLock >= bufferCount )
    {
        _fence.Wait( _nextBufferToLock - bufferCount + 1 );
    }

    return PeekBuffer( _nextBufferToLock++ );
}

void BufferChain::ReleaseNextBuffer()
{
    PanicIf( _nextBufferToRelease >= _nextBufferToLock, "" );
    PanicIf(_nextBufferToLock - _nextBufferToRelease > (uint32)_buffers.Length(), "" );

    _fence.Signal( ++_nextBufferToRelease );
}

void BufferChain::Reset()
{
    // Wait for the last buffer to be released
    _fence.Wait( _nextBufferToLock );

    // Reset state
    _fence.Reset( 0 );
    _nextBufferToRelease = 0;
    _nextBufferToLock    = 0;
}
