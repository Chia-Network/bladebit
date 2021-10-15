#pragma once

#include "io/FileStream.h"
#include "threading/AutoResetSignal.h"
#include "threading/ThreadPool.h"
#include "plotshared/MTJob.h"

class Thread;
#define BB_DISK_QUEUE_MAX_CMDS 256

enum class FileId
{
    None = 0,
    Y,
    META_LO, META_HI,
    X,
    T2_L, T2_R,
    T3_L, T3_R,
    T4_L, T4_R,
    T5_L, T5_R,
    T6_L, T6_R,
    T7_L, T7_R,
    F7

    ,_COUNT
};

struct FileSet
{
    const char*      name;
    Span<FileStream> files;
};

struct WriteBucketsJob
{
    FileSet* fileSet;

    uint*    sizes;
    byte*    buffers;
};

struct WriteToFileJob : MTJob<WriteToFileJob>
{
    const byte* buffer;
    byte*       blockBuffer;
    size_t      size;
    FileStream* file;

    void Run() override;
};


class DiskBufferQueue
{
    struct Command
    {
        enum CommandType
        {
            Void = 0,
            WriteFile,
            WriteBuckets,
            ReleaseBuffer
        };

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
                FileId      fileId;
            } buckets;

            struct
            {
                byte* buffer;
            } releaseBuffer;
        };
    };

public:

    DiskBufferQueue( const char* workDir, byte* workBuffer, 
                     size_t workBufferSize, size_t chunkSize,
                     uint ioThreadCount );

    ~DiskBufferQueue();

    uint CreateFile( const char* name, uint bucketCount );

    void WriteBuckets( FileId id, const byte* buckets, const uint* sizes );
    
    //void WriteFile( uint id, uint bucket, const byte* buffer, size_t size );

    void CommitCommands();

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

    Command* GetCommandObject();

    static void CommandThreadMain( DiskBufferQueue* self );
    void CommandMain();
    void ExecuteCommand( Command& cmd );

private:

    const char*      _workDir;
    byte*            _workBuffer;
    size_t           _workBufferSize;
    size_t           _chunkSize;

    ThreadPool       _threadPool;

    std::atomic<int> _nextBuffer;   // Next available buffer in the list. If < 0, then we have none.
    Span<int>        _bufferList;   // Free list of buffers

    FileSet          _files[(size_t)FileId::_COUNT];

    Thread           _dispatchThread;
    Command          _commands[BB_DISK_QUEUE_MAX_CMDS];

    int              _cmdWritePos = 0;
    int              _cmdsPending = 0;
    std::atomic<int> _cmdCount    = 0;

    AutoResetSignal _cmdReadySignal;
    AutoResetSignal _cmdConsumedSignal;


};