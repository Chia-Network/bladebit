#pragma once

#include "io/FileStream.h"
#include "threading/AutoResetSignal.h"
#include "threading/ThreadPool.h"
#include "plotshared/MTJob.h"
#include "plotshared/WorkHeap.h"

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

struct WriteBucketsJob : MTJob<WriteBucketsJob>
{
    FileSet*       fileSet;
    const uint*    sizes;
    const byte*    buffers;

    void Run() override;
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
            ReleaseBuffer,
            MemoryFence,
        };

        CommandType type;

        union
        {
            struct
            {
                const byte*  buffer;
                size_t size;
                FileId fileId;
                uint   bucket;
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

            struct
            {
                AutoResetSignal* signal;    // #TODO: Should this be a manual reset signal?
            } fence;
        };
    };

public:

    DiskBufferQueue( const char* workDir, byte* workBuffer, 
                     size_t workBufferSize, uint ioThreadCount,
                     bool useDirectIO = true );
    ~DiskBufferQueue();


    void WriteBuckets( FileId id, const void* buckets, const uint* sizes );
    
    void WriteFile( FileId id, uint bucket, const void* buffer, size_t size );

    void CommitCommands();

    // Obtain a buffer allocated from the work heap.
    // May block until there's a buffer available if there was none.
    // This assumes a single consumer.
    byte* GetBuffer( size_t size );

    // Release/return a chunk buffer that was in use, gotten by GetBuffer()
    // These returns the buffer back to the queue so that it is in use.
    // This command is serialized and should be added after any writing/reading has finished
    // with said buffer
    void ReleaseBuffer( void* buffer );

    // Add a memory fence into the command stream.
    // The signal will be set once this command is reached.
    // This ensures the caller that all commands before the 
    // fence have been processed.
    void AddFence( AutoResetSignal& signal );

    inline size_t BlockSize() const { return _blockSize; }

private:

    void InitFileSet( FileId fileId, const char* name, uint bucketCount, char* pathBuffer, size_t workDirLength );

    Command* GetCommandObject();

    static void CommandThreadMain( DiskBufferQueue* self );
    void CommandMain();
    void ExecuteCommand( Command& cmd );

    void CmdWriteBuckets( const Command& cmd );
    void CndWriteFile( const Command& cmd );

    void WriteToFile( FileStream& file, size_t size, const byte* buffer, byte* blockBuffer, const char* fileName, uint bucket );

private:

    const char*      _workDir;              // Temporary directory in which we will store our temporary files
    WorkHeap         _workHeap;             // Reserved memory for performing plot work and I/O
    
    // Handles to all files needed to create a plot
    FileSet          _files[(size_t)FileId::_COUNT];
    byte*            _blockBuffer   = nullptr;
    size_t           _blockSize     = 0;

    // I/O thread stuff
    Thread            _dispatchThread;
    
//     Command           _commands[BB_DISK_QUEUE_MAX_CMDS];
    SPCQueue<Command, BB_DISK_QUEUE_MAX_CMDS> _commands;

    bool              _userDirectIO;
    ThreadPool        _threadPool;

    int               _cmdWritePos   = 0;
    int               _cmdsPending   = 0;
    std::atomic<int>  _cmdCount      = 0;

    AutoResetSignal   _cmdReadySignal;
    AutoResetSignal   _cmdConsumedSignal;


};