#pragma once

#include "io/FileStream.h"

class Thread;

class DiskBufferQueue
{

    struct FileSet
    {
        const char*      name;
        Span<FileStream> files;
    };

    enum class CommandType : uint
    {
        Void = 0,
        WriteFile,
    };

    struct Command
    {
        CommandType type;

        union
        {
            struct
            {
                size_t size;
                byte*  buffer;
            } write;

            struct
            {
                const uint* sizes;
                const byte* buffers;
            } buckets;
        };
    };

public:

    DiskBufferQueue( const char* workDir, byte* workBuffer, size_t workBufferSize, size_t chunkSize );

    ~DiskBufferQueue();

    uint CreateFile( const char* name, uint bucketCount );

    //void WriteBuckets( uint id, )
    
    void WriteFile( uint id, uint bucket, const byte* buffer, size_t size );

    // Obtain a chunk buffer for use.
    // May block until there's a buffer available if there was none.
    // This assumes a single consumer
    byte* GetBuffer();

    // Release/return a chunk buffer that was in use, gotten by GetBuffer()
    // These returns the buffer back to the queue so that it is in use.
    // This command is serialized and should be added after any writing/reding has finished
    // with said buffer
    void ReleaseBuffer( byte* buffer );

private:

    const char*   _workDir;
    byte*         _workBuffer;
    size_t        _workBufferSize;
    size_t        _chunkSize;

    std::atomic<int> _nextBuffer;   // Next available buffer in the list. If < 0, then we have none.
    Span<int>        _bufferList;   // Free list of buffers

    Span<FileSet> _files;

    Thread*       _dispatchThread;
};