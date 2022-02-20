#pragma once

#include "io/FileStream.h"
#include "threading/Fence.h"
#include "threading/ThreadPool.h"
#include "threading/MTJob.h"
#include "plotting/WorkHeap.h"
#include "plotting/Tables.h"
#include "FileId.h"

class Thread;

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
            ReadFile,
            SeekFile,
            SeekBucket,
            DeleteFile,
            DeleteBucket,
            ReleaseBuffer,
            SignalFence,
            WaitForFence,
        };

        CommandType type;

        union
        {
            struct
            {
                byte*  buffer;
                size_t size;
                FileId fileId;
                uint   bucket;
            } file;

            struct
            {
                const uint* sizes;
                const byte* buffers;
                FileId      fileId;
            } buckets;

            struct
            {
                FileId     fileId;
                uint       bucket;
                int64      offset;
                SeekOrigin origin;

            } seek;

            struct
            {
                byte* buffer;
            } releaseBuffer;

            struct
            {
                Fence* signal;
                int64  value;
            } fence;

            struct
            {
                FileId fileId;
                uint   bucket;
            } deleteFile;
        };
    };

    struct FileDeleteCommand
    {
        FileId fileId;
        int64  bucket;  // If < 0, delete all buckets
    };

public:

    DiskBufferQueue( const char* workDir, byte* workBuffer, 
                     size_t workBufferSize, uint ioThreadCount,
                     bool useDirectIO );
    
    ~DiskBufferQueue();

    bool InitFileSet( FileId fileId, const char* name, uint bucketCount );

    void OpenPlotFile( const char* fileName, const byte* plotId, const byte* plotMemo, uint16 plotMemoSize );

    void ResetHeap( const size_t heapSize, void* heapBuffer );

    void WriteBuckets( FileId id, const void* buckets, const uint* sizes );

    void WriteFile( FileId id, uint bucket, const void* buffer, size_t size );

    void ReadFile( FileId id, uint bucket, void* dstBuffer, size_t readSize );

    void SeekFile( FileId id, uint bucket, int64 offset, SeekOrigin origin );

    void SeekBucket( FileId id, int64 offset, SeekOrigin origin );

    void DeleteFile( FileId id, uint bucket );

    void DeleteBucket( FileId id );

    void CommitCommands();

    // Obtain a buffer allocated from the work heap.
    // May block until there's a buffer available if there was none.
    // This assumes a single consumer.
    inline byte* GetBuffer( size_t size, bool blockUntilFreeBuffer = true ) { return _workHeap.Alloc( size, _blockSize, blockUntilFreeBuffer ); }

    // Release/return a chunk buffer that was in use, gotten by GetBuffer()
    // These returns the buffer back to the queue so that it is in use.
    // This command is serialized and should be added after 
    // any writing/reading has finished with said buffer
    void ReleaseBuffer( void* buffer );

    // Add a memory fence into the command stream.
    // The signal will be set once this command is reached.
    // This ensures the caller that all commands before the 
    // fence have been processed.
    void SignalFence( Fence& fence );

    // Signal a fence with a specific value.
    void SignalFence( Fence& fence, uint32 value );

    // Instructs the command dispatch thread to wait until the specified fence has been signalled
    void WaitForFence( Fence& fence );

    void CompletePendingReleases();

    inline size_t BlockSize()     const { return _blockSize; }
    inline bool   UseDirectIO()   const { return _useDirectIO; }

    inline const WorkHeap& Heap() const { return _workHeap; }

    inline size_t PlotHeaderSize() const { return _plotHeaderSize; }

    inline uint64 PlotTablePointersAddress() const { return _plotTablesPointers; }
    
private:

    Command* GetCommandObject( Command::CommandType type );

    static void CommandThreadMain( DiskBufferQueue* self );
    void CommandMain();

    static void DeleterThreadMain( DiskBufferQueue* self );
    void DeleterMain();

    void ExecuteCommand( Command& cmd );

    void CmdWriteBuckets( const Command& cmd );
    void CndWriteFile( const Command& cmd );
    void CmdReadFile( const Command& cmd );
    void CmdSeekBucket( const Command& cmd );

    void WriteToFile( FileStream& file, size_t size, const byte* buffer, byte* blockBuffer, const char* fileName, uint bucket );
    void ReadFromFile( FileStream& file, size_t size, byte* buffer, byte* blockBuffer, const char* fileName, uint bucket );

    void CmdDeleteFile( const Command& cmd );
    void CmdDeleteBucket( const Command& cmd );

    void DeleteFileNow( const FileId fileId, uint32 bucket );
    void DeleteBucketNow( const FileId fileId );

    static const char* DbgGetCommandName( Command::CommandType type );


private:
    std::string      _workDir;              // Temporary directory in which we will store our temporary files
    WorkHeap         _workHeap;             // Reserved memory for performing plot work and I/O
    
    // Handles to all files needed to create a plot
    FileSet          _files[(size_t)FileId::_COUNT];
    byte*            _blockBuffer    = nullptr;
    size_t           _blockSize      = 0;
    
    char*            _filePathBuffer = nullptr; // For deleting files

    size_t           _plotHeaderSize     = 0;
    byte*            _plotHeaderbuffer   = nullptr;
    uint64           _plotTablesPointers = 0;            // Offset in the plot file to the tables pointer table


    // I/O thread stuff
    Thread            _dispatchThread;
    
    SPCQueue<Command, BB_DISK_QUEUE_MAX_CMDS> _commands;

    bool              _useDirectIO;
    // ThreadPool        _threadPool;

    AutoResetSignal   _cmdReadySignal;
    AutoResetSignal   _cmdConsumedSignal;

    // File deleter thread
    Thread            _deleterThread;                   // For deleting files.
    AutoResetSignal   _deleteSignal;                    // We do this in a separate thread as to not
    GrowableSPCQueue<FileDeleteCommand> _deleteQueue;   // block other commands when the kernel is clearing cached IO buffers for the files.
    bool              _deleterExit = false;

    
};
