#pragma once
#include "io/FileStream.h"
#include "threading/Fence.h"
#include "DiskQueue.h"
#include "util/Span.h"


class IAllocator;

/**
 * Dual-buffered base class for DiskQueue-based writing and reading.
 */
class DiskBufferBase
{
    friend class DiskQueue;

protected:
    static bool MakeFile( DiskQueue& queue, const char* name, FileMode mode, FileAccess access, FileFlags flags, FileStream& file );

    DiskBufferBase( DiskQueue& queue, FileStream& stream, const char* name, uint32 bucketCount );

    virtual void HandleCommand( const DiskQueueDispatchCommand& cmd ) = 0;

    static void ReserveBufferForInstance( DiskBufferBase* self, IAllocator& allocator, size_t size, size_t alignment );
    void ReserveBuffers( IAllocator& allocator, size_t size, size_t alignment );

    static size_t GetReserveAllocSize( size_t size, const size_t alignment );
public:

    virtual void ReserveBuffers( IAllocator& allocator ) = 0;

    /// Assigns already existing buffers to be used as I/O buffers
    void AssignBuffers( void* readBuffers[2], void* writeBuffers[2] );
    void AssignReadBuffers( void* readBuffers[2] );
    void AssignWriteBuffers( void* writeBuffers[2] );

    /// Takes the same buffers that another DiskBufferBase uses and shares them.
    void ShareBuffers( const DiskBufferBase& other );

    /// Read next bucket
    virtual void ReadNextBucket() = 0;

    /// Waits for the last write to finish
    /// and marks completion of writing and reading a table.
    virtual void Swap();

    void* GetNextWriteBuffer();
    void* GetNextReadBuffer();

    void* PeekReadBufferForBucket( uint32 bucket );

    /// Gets the write buffer for a certain bucket without waiting for it (unsafe)
    void* PeekWriteBufferForBucket( uint32 bucket );

    void WaitForWriteToComplete( uint32 bucket );
    void WaitForLastWriteToComplete();

    void WaitForReadToComplete( uint32 bucket );
    void WaitForNextReadToComplete();


    inline const char* Name() const
    {
        return _name.c_str();
    }

    inline FileStream& File() const
    {
        return const_cast<DiskBufferBase*>( this )->_file;
    }

    /// Helpers
    inline bool TryReadNextBucket()
    {
        if( _nextReadBucket >= _bucketCount )
            return false;

        ReadNextBucket();
        return true;
    }

    inline uint32 GetNextReadBucketId() const
    {
        ASSERT( _nextReadBucket < _bucketCount );
        return _nextReadBucket; 
    }

public:
    virtual ~DiskBufferBase();

protected:
    /**
     * Returns the bucket about to be written.
    */
    uint32 BeginWriteSubmission();
    void   EndWriteSubmission();

    uint32 BeginReadSubmission();
    void   EndReadSubmission();

protected:
    DiskQueue*  _queue;
    FileStream  _file;
    std::string _name;

    uint32      _bucketCount;
    Fence       _writeFence;
    Fence       _readFence;

    byte*       _writeBuffers[2]  = {};
    byte*       _readBuffers [2]  = {};

    uint32      _nextWriteBucket = 0;   // Next bucket that will be written to disk
    uint32      _nextReadBucket  = 0;   // Next bucket that will be read from disk
    uint32      _nextWriteLock   = 0;   // Next write bucket buffer index that will be locked (for user use)
    uint32      _nextReadLock    = 0;   // Next read bucket buffer index that will be locked (for user use)
};
