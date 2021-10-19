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

template<typename T, int Capacity>
class ProducerConsumerQueue
{
public:
    struct Iterator
    {
        int _iteration;     // One of potentially 2
        int _endIndex[2];

        bool Finished() const;
        void Next();
    };

    ProducerConsumerQueue( T* buffer );


    void Produce();
    T*   Consume();

    Iterator Begin() const;

    T& Get( const Iterator& iter );

    // Block and wait to be signaled that the
    // producer thread has added something to the buffer
    void WaitForProduction();

private:
    int              _writePosition;
    std::atomic<int> _count;
    T                _buffer[Capacity];
    AutoResetSignal  _signal;
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

//     uint CreateFile( const char* name, uint bucketCount );

    void WriteBuckets( FileId id, const byte* buckets, const uint* sizes );
    
    //void WriteFile( uint id, uint bucket, const byte* buffer, size_t size );

    void CommitCommands();

    // Obtain a buffer allocated from the work heap.
    // May block until there's a buffer available if there was none.
    // This assumes a single consumer.
    byte* GetBuffer( size_t size );

    // Release/return a chunk buffer that was in use, gotten by GetBuffer()
    // These returns the buffer back to the queue so that it is in use.
    // This command is serialized and should be added after any writing/reading has finished
    // with said buffer
    void ReleaseBuffer( byte* buffer );

    inline size_t BlockSize() const { return _blockSize; }

private:

    void InitFileSet( FileId fileId, const char* name, uint bucketCount, char* pathBuffer, size_t workDirLength );

    void ConsumeReleasedBuffers();

    Command* GetCommandObject();

    static void CommandThreadMain( DiskBufferQueue* self );
    void CommandMain();
    void ExecuteCommand( Command& cmd );

    void CmdWriteBuckets( const Command& cmd );

private:

    // Represents a portion of unallocated space in our heap/work buffer
    struct HeapEntry
    {
        byte*  address;
        size_t size;
    };

private:

    const char*      _workDir;              // Temporary directory in which we will store our temporary files
    WorkHeap         _workHeap;             // Our working heap
    
    // Handles to all files needed to create a plot
    FileSet          _files[(size_t)FileId::_COUNT];
    byte*            _blockBuffer   = nullptr;
    size_t           _blockSize     = 0;

    // I/O thread stuff
    Thread            _dispatchThread;
    Command           _commands[BB_DISK_QUEUE_MAX_CMDS];

    ThreadPool        _threadPool;

    int               _cmdWritePos   = 0;
    int               _cmdsPending   = 0;
    std::atomic<int>  _cmdCount      = 0;

    ProducerConsumerQueue<HeapEntry> _releasedBuffers;  // Queue used to return buffers back to the consumer

    AutoResetSignal   _cmdReadySignal;
    AutoResetSignal   _cmdConsumedSignal;


};