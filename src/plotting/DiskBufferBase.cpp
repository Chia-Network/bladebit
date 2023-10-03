#include "DiskBufferBase.h"
#include "DiskQueue.h"
#include "util/IAllocator.h"
#include "util/StackAllocator.h"
#include <filesystem>

bool DiskBufferBase::MakeFile( DiskQueue& queue, const char* name,
                               FileMode mode, FileAccess access, FileFlags flags, FileStream& file )
{
    ASSERT( !file.IsOpen() );

    std::string path = std::filesystem::path( queue.Path() ).append( name ).string();

    return file.Open( path.c_str(), mode, access, flags );
}

DiskBufferBase::DiskBufferBase( DiskQueue& queue, FileStream& stream,
                                const char* name, uint32 bucketCount )
    : _queue      ( &queue )
    , _file       ( std::move( stream ) )
    , _name       ( name )
    , _bucketCount( bucketCount )
{}

DiskBufferBase::~DiskBufferBase()
{
    _file.Close();
    std::string path = std::filesystem::path( _queue->Path() ).append( _name ).string();
    ::remove( path.c_str() );

    // #TODO: Track the allocator used, and only release if we have that reference.
    // if( _writeBuffers[0] ) bbvirtfreebounded( _writeBuffers[0] );
    // if( _writeBuffers[1] ) bbvirtfreebounded( _writeBuffers[1] );
    // if( _readBuffers[0] ) bbvirtfreebounded( _readBuffers[0] );
    // if( _readBuffers[1] ) bbvirtfreebounded( _readBuffers[1] );
}

void DiskBufferBase::ReserveBufferForInstance( DiskBufferBase* self, IAllocator& allocator, const size_t size, const size_t alignment )
{
    if( self )
    {
        PanicIf( self->_writeBuffers[0], "Buffers already reserved for '%s'.", self->_name.c_str() );
    }

    byte* w0 = allocator.AllocT<byte>( size, alignment );
    byte* w1 = allocator.AllocT<byte>( size, alignment );
    byte* r0 = allocator.AllocT<byte>( size, alignment );
    byte* r1 = allocator.AllocT<byte>( size, alignment );

    if( self )
    {
        self->_writeBuffers[0] = w0;
        self->_writeBuffers[1] = w1;
        self->_readBuffers [0] = r0;
        self->_readBuffers [1] = r1;
    }
}

size_t DiskBufferBase::GetReserveAllocSize( const size_t size, const size_t alignment )
{
    DummyAllocator allocator;
    ReserveBufferForInstance( nullptr, allocator, size, alignment );

    return allocator.Size();
}

void DiskBufferBase::ReserveBuffers( IAllocator& allocator, const size_t size, const size_t alignment )
{
    ReserveBufferForInstance( this, allocator, size, alignment );
}

void DiskBufferBase::AssignBuffers( void* readBuffers[2], void* writeBuffers[2] )
{
    AssignReadBuffers( readBuffers );
    AssignWriteBuffers( writeBuffers );
}

void DiskBufferBase::AssignReadBuffers( void* readBuffers[2] )
{
    // PanicIf( _readBuffers[0], "Read buffers already assigned for '%s'.", _name.c_str() );
    _readBuffers [0] = (byte*)readBuffers [0];
    _readBuffers [1] = (byte*)readBuffers [1];
}

void DiskBufferBase::AssignWriteBuffers( void* writeBuffers[2] )
{
    // PanicIf( _writeBuffers[0], "Write buffers already assigned for '%s'.", _name.c_str() );
    _writeBuffers[0] = (byte*)writeBuffers[0];
    _writeBuffers[1] = (byte*)writeBuffers[1];
}


void DiskBufferBase::ShareBuffers( const DiskBufferBase& other )
{
    _writeBuffers[0] = other._writeBuffers[0];
    _writeBuffers[1] = other._writeBuffers[1];
    _readBuffers [0] = other._readBuffers [0];
    _readBuffers [1] = other._readBuffers [1];
}

void DiskBufferBase::Swap()
{
//    FatalIf( !_file.Seek( 0, SeekOrigin::Begin ), "Failed to seek '%s'.", _name.c_str() );
    WaitForLastWriteToComplete();

    _nextWriteBucket = 0;
    _nextReadBucket  = 0;
    _nextWriteLock   = 0;
    _nextReadLock    = 0;

    _readFence .Reset();
    _writeFence.Reset();
}

void* DiskBufferBase::GetNextWriteBuffer()
{
    PanicIf( _nextWriteLock >= _bucketCount, "Write bucket overflow." );
    PanicIf( (int64)_nextWriteLock - (int64)_nextWriteBucket >= 2, "Invalid write buffer lock for '%s'.", _name.c_str() );

    void* buf = _writeBuffers[_nextWriteLock % 2];
    PanicIf( !buf, "No write buffer reserved for '%s'.", _name.c_str() );

    if( _nextWriteLock++ >= 2 )
        WaitForWriteToComplete( _nextWriteLock-2 );

    return buf;
}

void* DiskBufferBase::PeekReadBufferForBucket( uint32 bucket )
{
    PanicIf( _nextReadLock >= _bucketCount, "Read bucket overflow." );
    return _readBuffers[bucket % 2];
}

void* DiskBufferBase::PeekWriteBufferForBucket( const uint32 bucket )
{
    PanicIf( _nextWriteLock >= _bucketCount, "Write bucket overflow." );
    return _writeBuffers[bucket % 2];
}

void DiskBufferBase::WaitForWriteToComplete( const uint32 bucket )
{
    _writeFence.Wait( bucket + 1 );
}

void DiskBufferBase::WaitForLastWriteToComplete()
{
    if( _nextWriteBucket < 1 )
        return;

    WaitForWriteToComplete( _nextWriteBucket-1 );
}

void* DiskBufferBase::GetNextReadBuffer()
{
    PanicIf( _nextReadLock >= _bucketCount, "Read bucket overflow." );
    PanicIf( _nextReadLock >= _nextReadBucket, "Invalid read buffer lock for '%s'.", _name.c_str() );

    void* buf = _readBuffers[_nextReadLock % 2];
    PanicIf( !buf, "No read buffer reserved for '%s'.", _name.c_str() );

    WaitForReadToComplete( _nextReadLock++ );
    return buf;
}

void DiskBufferBase::WaitForReadToComplete( const uint32 bucket )
{
    _readFence.Wait( bucket + 1 );
}

void DiskBufferBase::WaitForNextReadToComplete()
{
    FatalIf( _nextReadBucket < 1, "Nothing yet read for '%s'.", _name.c_str() );
    FatalIf( _nextReadLock >= _nextReadBucket, "Invalid read buffer lock for '%s'.", _name.c_str() );

    Panic( "Unsupported. Nothing to see here." );

    // # TODO: Don't use this as is, it is not intuitive and can causes errors.
    //         Use GetNextReadBuffer() or WaitForReadToComplete() instead.
    WaitForReadToComplete( _nextReadBucket-1 );
}

uint32 DiskBufferBase::BeginWriteSubmission()
{
    FatalIf( (int64)_nextWriteLock - (int64)_nextWriteBucket > 2, "Invalid write lock state for '%s'.", _name.c_str() );
    return _nextWriteBucket;
}

void DiskBufferBase::EndWriteSubmission()
{
    _queue->SignalFence( _writeFence, ++_nextWriteBucket );
}

uint32 DiskBufferBase::BeginReadSubmission()
{
    FatalIf( _nextReadBucket >= _bucketCount, "'%s' Read bucket overflow.", Name() );
    return _nextReadBucket;
}

void DiskBufferBase::EndReadSubmission()
{
    _queue->SignalFence( _readFence, ++_nextReadBucket );
}
