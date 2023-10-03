#include "DiskBucketBuffer.h"
#include "DiskQueue.h"
#include "plotdisk/jobs/IOJob.h"
#include "util/IAllocator.h"
#include "util/StackAllocator.h"
#include <filesystem>

DiskBucketBuffer::DiskBucketBuffer( DiskQueue& queue, FileStream& stream, const char* name,
                                    uint32 bucketCount, size_t sliceCapacity )
    : DiskBufferBase( queue, stream, name, bucketCount )
    , _sliceCapacity( RoundUpToNextBoundaryT( sliceCapacity, queue.BlockSize() ) )
    // , _writeSliceStride( _sliceCapacity )   // Start writing horizontally
    // , _readSliceStride( _sliceCapacity * bucketCount )
{
    ASSERT( bucketCount > 0 );

    _writeSliceSizes.resize( bucketCount );
    _readSliceSizes .resize( bucketCount );
    for( size_t bucket = 0; bucket < bucketCount; bucket++ )
    {
        _writeSliceSizes[bucket].resize( bucketCount );
        _readSliceSizes [bucket].resize( bucketCount );
    }
}

DiskBucketBuffer::~DiskBucketBuffer()
{}

DiskBucketBuffer*
DiskBucketBuffer::Create( DiskQueue& queue, const char* fileName,
                          uint32 bucketCount, size_t sliceCapacity,
                          FileMode mode, FileAccess access, FileFlags flags )
{
    FileStream file;
    if( !DiskBufferBase::MakeFile( queue, fileName, mode, access, flags, file ) )
        return nullptr;

    return new DiskBucketBuffer( queue, file, fileName, bucketCount, sliceCapacity );
}

size_t DiskBucketBuffer::GetSingleBucketBufferSize( DiskQueue& queue, uint32 bucketCount, size_t sliceCapacity )
{
    return RoundUpToNextBoundaryT( sliceCapacity, queue.BlockSize() ) * bucketCount;
}

size_t DiskBucketBuffer::GetReserveAllocSize( DiskQueue& queue, uint32 bucketCount, size_t sliceCapacity )
{
    return DiskBufferBase::GetReserveAllocSize(
        GetSingleBucketBufferSize( queue, bucketCount, sliceCapacity ),
        queue.BlockSize() );
}

void DiskBucketBuffer::ReserveBuffers( IAllocator& allocator )
{   
    DiskBufferBase::ReserveBuffers( allocator, GetBucketRowStride(), _queue->BlockSize() );
}

void DiskBucketBuffer::Swap()
{
    DiskBufferBase::Swap();

    // std::swap( _writeSliceStride, _readSliceStride );
    _verticalWrite = !_verticalWrite;
    std::swap( _writeSliceSizes, _readSliceSizes );
}

void DiskBucketBuffer::Submit( const size_t sliceStride )
{
    PanicIf( sliceStride > _sliceCapacity, "Invalid slice stride %llu is greater than capacity %llu for %s.", 
        (llu)sliceStride, (llu)_sliceCapacity, Name() );

    const uint32 bucket = BeginWriteSubmission();

    DiskQueueDispatchCommand dcmd = {};
    auto& cmd = dcmd.bucketBufferCmd;

    cmd.type = DiskBucketBufferCommand::Write;
    auto& c = cmd.write;

    c.sliceStride = sliceStride;
    c.bucket      = bucket;
    c.vertical    = _verticalWrite;

    _queue->EnqueueDispatchCommand( this, dcmd );

    // Record slice sizes (write 1 column cell per row)
    // At the end of a table a bucket row will have
    // all the slice sizes of a given bucket.
    for( uint32 row = 0; row < _bucketCount; row++ )
    {
        _writeSliceSizes[row][bucket] = sliceStride;//sliceSizes[row];
    }

    // Signal completion
    EndWriteSubmission();
}

void DiskBucketBuffer::ReadNextBucket()
{
    const uint32 bucket = BeginReadSubmission();

    DiskQueueDispatchCommand dcmd = {};
    auto& cmd = dcmd.bucketBufferCmd;

    cmd.type = DiskBucketBufferCommand::Read;
    auto& c = cmd.read;
    c.bucket   = bucket;
    c.vertical = _verticalWrite; // If the last write was NOT vertical, then the read is vertical.

    _queue->EnqueueDispatchCommand( this, dcmd );

    EndReadSubmission();
}

Span<byte> DiskBucketBuffer::PeekReadBuffer( const uint32 bucket )
{
    size_t totalSize = 0;
    for( auto sz : _readSliceSizes[bucket] )
        totalSize += sz;

    return Span<byte>( _readBuffers[bucket % 2], totalSize );
}

void DiskBucketBuffer::OverrideReadSlices( const uint32 bucket, const size_t elementSize, const uint32* sliceSizes, const uint32 stride )
{
    size_t totalSize = 0;

    auto& readSlices = _readSliceSizes[bucket];
    ASSERT( readSlices.size() == _bucketCount );

    for( size_t i = 0; i < _bucketCount; i++ )
    {
        readSlices[i] = *sliceSizes * elementSize;
        sliceSizes += stride;
    }
}


///
/// These are executed from the DiskQueue thread
///
void DiskBucketBuffer::HandleCommand( const DiskQueueDispatchCommand& cmd )
{
    const auto& c = cmd.bucketBufferCmd;

    switch( c.type )
    {
        default:
            Panic( "Unexpected." );
            break;

        case DiskBucketBufferCommand::Write:
            CmdWriteSlices( c );
            break;

        case DiskBucketBufferCommand::Read:
            CmdReadSlices( c );
            break;
    }
}

void DiskBucketBuffer::CmdWriteSlices( const DiskBucketBufferCommand& cmd )
{
    auto & c = cmd.write;
    int err = 0;

    const byte*  src       = (byte*)_writeBuffers[c.bucket % 2];
    const size_t srcStride = c.sliceStride;
    const size_t dstStride = c.vertical ? GetBucketRowStride() : GetSliceStride(); 

    // Offset to the starting location
    int64 offset = (int64)(c.vertical ? _sliceCapacity * c.bucket : GetBucketRowStride() * c.bucket );

    // Seek to starting location
    for( uint32 i = 0; i < _bucketCount; i++ )
    {
        // Seek to next slice
        FatalIf( !_file.Seek( offset, SeekOrigin::Begin ),
                    "Failed to seek to slice %u start on '%s/%s' with error %d.",
                    i, _queue->Path(), Name(), (int32)_file.GetError() );
        offset += (int64)dstStride;

        // Write slice
        if( !IOJob::WriteToFileUnaligned( _file, src, srcStride, err ) )
        {
            Fatal( "Failed to write slice on '%s/%s' with error %d.", _queue->Path(), Name(), err );
        }

        src += srcStride;
    }
}

void DiskBucketBuffer::CmdReadSlices( const DiskBucketBufferCommand& cmd )
{
    const auto& c = cmd.read;

    int err = 0;

    byte* dst = _readBuffers[c.bucket % 2];

    const size_t rowStride   = GetBucketRowStride();
    const size_t sliceStride = GetSliceStride();

    // Use the last slice as a temp buffer (to avoid the slower memmove on most copies)
    byte* tmpBuffer = dst + sliceStride * (_bucketCount-1);

    for( size_t i = 0; i < _bucketCount; i++ )
    {
        // Seek to starting location of the slice
        const size_t colOffset = c.vertical ? sliceStride * c.bucket : sliceStride * i;
        const size_t rowOffset = c.vertical ? rowStride * i          : rowStride * c.bucket;

        if( !_file.Seek( (int64)(rowOffset + colOffset), SeekOrigin::Begin ) )
        {
            Fatal( "Failed to seek to slice %u start on '%s/%s' with error %d.",
                   i, _queue->Path(), Name(), (int32)_file.GetError() );
        }

        // Read a full block-aligned slice
        if( !IOJob::ReadFromFileUnaligned( _file, tmpBuffer, sliceStride, err ) )
        {
            if( err != 0 || i + 1 < _bucketCount )
            {
                Fatal( "Failed to read slice from '%s/%s' with error %d.", _queue->Path(), Name(), err );
            }
        }

        // Copy read buffer to actual location
        const size_t sliceSize = _readSliceSizes[c.bucket][i];

        if( i + 1 < _bucketCount )
            memcpy( dst, tmpBuffer, sliceSize );
        else
            memmove( dst, tmpBuffer, sliceSize );   // Last copy overlaps since it's the same as the temp buffer

        dst += sliceSize;
    }
}
