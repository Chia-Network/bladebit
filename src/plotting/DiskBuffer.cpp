#include "DiskBuffer.h"
#include "DiskQueue.h"
#include "plotdisk/jobs/IOJob.h"

DiskBuffer* DiskBuffer::Create( DiskQueue& queue, const char* fileName, uint32 bucketCount,
                                size_t bufferSize, FileMode mode, FileAccess access, FileFlags flags )
{
    FileStream file;
    if( !DiskBufferBase::MakeFile( queue, fileName, mode, access, flags, file ) )
        return nullptr;

    return new DiskBuffer( queue, file, fileName, bucketCount, bufferSize );
}

DiskBuffer::DiskBuffer( DiskQueue& queue, FileStream& stream, const char* name,
                        uint32 bucketCount, size_t bufferSize )
    : DiskBufferBase( queue, stream, name, bucketCount )
    , _bufferSize( bufferSize )
    , _alignedBufferSize( RoundUpToNextBoundaryT( bufferSize, _file.BlockSize() ) )
{
    _bucketSizes.resize( bucketCount );
}

DiskBuffer::~DiskBuffer() {}

void DiskBuffer::ReserveBuffers( IAllocator& allocator )
{
    DiskBufferBase::ReserveBuffers( allocator, _alignedBufferSize, _file.BlockSize() );
}

size_t DiskBuffer::GetReserveAllocSize( DiskQueue& queue, size_t bufferSize )
{
    const size_t alignment = queue.BlockSize();
    
    return DiskBufferBase::GetReserveAllocSize( RoundUpToNextBoundaryT( bufferSize, alignment ), alignment );
}

void DiskBuffer::Swap()
{
    DiskBufferBase::Swap();

    FatalIf( !_file.Seek( 0, SeekOrigin::Begin ), "Failed to seek to file start on '%s/%s' with error %d.",
            _queue->Path(), Name(), (int32)_file.GetError() );
}

void DiskBuffer::ReadNextBucket()
{
    FatalIf( _nextReadBucket >= _bucketCount, "'%s' Read bucket overflow.", Name() );

    // Read whole bucket
    DiskQueueDispatchCommand dcmd = {};
    auto& cmd = dcmd.bufferCmd;
    cmd.type = DiskBufferCommand::Read;

    auto& c = cmd.read;
    c.bucket = _nextReadBucket;

    _queue->EnqueueDispatchCommand( this, dcmd );
    _queue->SignalFence( _readFence, ++_nextReadBucket );
}

void DiskBuffer::Submit( const size_t size )
{
    FatalIf( (int64)_nextWriteLock - (int64)_nextWriteBucket > 2, "Invalid write lock state for '%s'.", _name.c_str() );
    FatalIf( size > _alignedBufferSize, "Write submission too large for '%s'.", _name.c_str() );

    DiskQueueDispatchCommand dcmd = {};
    auto& cmd = dcmd.bufferCmd;
    cmd.type = DiskBufferCommand::Write;

    auto& c = cmd.write;
    c.bucket = _nextWriteBucket;
    _queue->EnqueueDispatchCommand( this, dcmd );

    // Signal completion
    _queue->SignalFence( _writeFence, ++_nextWriteBucket );
}

void DiskBuffer::HandleCommand( const DiskQueueDispatchCommand& cmd )
{
    const auto& c = cmd.bufferCmd;

    switch( c.type )
    {
        case DiskBufferCommand::None:
            ASSERT( 0 );
            break;
        case DiskBufferCommand::Write:
            CmdWrite( c );
            break;
        case DiskBufferCommand::Read:
            CmdRead( c );
            break;
    }
}

void DiskBuffer::CmdWrite( const DiskBufferCommand& cmd )
{
    const auto& c = cmd.write;

    // Write a full block-aligned bucket
    int err = 0;
    if( !IOJob::WriteToFileUnaligned( _file, _writeBuffers[c.bucket % 2], _alignedBufferSize, err ) )
    {
        Fatal( "Failed to write bucket to '%s/%s' with error %d.", _queue->Path(), Name(), err );
    }
}

void DiskBuffer::CmdRead( const DiskBufferCommand& cmd )
{
    const auto& c = cmd.read;

    // Read a full block-aligned bucket
    int err = 0;
    if( !IOJob::ReadFromFileUnaligned( _file, _readBuffers[c.bucket % 2], _alignedBufferSize, err ) )
    {
        Fatal( "Failed to read bucket from '%s/%s' with error %d.", _queue->Path(), Name(), err );
    }
}
