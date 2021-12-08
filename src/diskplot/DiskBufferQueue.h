#pragma once

#include "io/FileStream.h"
#include "threading/AutoResetSignal.h"
#include "threading/ThreadPool.h"
#include "plotshared/MTJob.h"
#include "plotshared/WorkHeap.h"
#include "plotshared/Tables.h"

class Thread;

enum class FileId
{
    None = 0,
    Y0, Y1,
    META_A_0, META_B_0,
    META_A_1, META_B_1,
    X,

    T2_L, T2_R, 
    T3_L, T3_R, 
    T4_L, T4_R, 
    T5_L, T5_R, 
    T6_L, T6_R, 
    T7_L, T7_R, 

    SORT_KEY2,
    SORT_KEY3,
    SORT_KEY4,
    SORT_KEY5,
    SORT_KEY6,
    SORT_KEY7,
    
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
            ReadFile,
            SeekFile,
            SeekBucket,
            ReleaseBuffer,
            MemoryFence,
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
                AutoResetSignal* signal;    // #TODO: Should this be a manual reset signal?
            } fence;
        };
    };

public:

    DiskBufferQueue( const char* workDir, byte* workBuffer, 
                     size_t workBufferSize, uint ioThreadCount,
                     bool useDirectIO );
    
    ~DiskBufferQueue();

    void ResetHeap( const size_t heapSize, void* heapBuffer );


    void WriteBuckets( FileId id, const void* buckets, const uint* sizes );
    
    void WriteFile( FileId id, uint bucket, const void* buffer, size_t size );

    void ReadFile( FileId id, uint bucket, void* dstBuffer, size_t readSize );

    void SeekFile( FileId id, uint bucket, int64 offset, SeekOrigin origin );
    
    void SeekBucket( FileId id, int64 offset, SeekOrigin origin );

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

    void CompletePendingReleases();

    inline size_t BlockSize() const { return _blockSize; }
    inline bool UseDirectIO() const { return _useDirectIO; }
    
private:

    void InitFileSet( FileId fileId, const char* name, uint bucketCount, char* pathBuffer, size_t workDirLength );

    Command* GetCommandObject( Command::CommandType type );

    static void CommandThreadMain( DiskBufferQueue* self );
    void CommandMain();
    void ExecuteCommand( Command& cmd );

    void CmdWriteBuckets( const Command& cmd );
    void CndWriteFile( const Command& cmd );
    void CmdReadFile( const Command& cmd );
    void CmdSeekBucket( const Command& cmd );

    void WriteToFile( FileStream& file, size_t size, const byte* buffer, byte* blockBuffer, const char* fileName, uint bucket );
    void ReadFromFile( FileStream& file, size_t size, byte* buffer, byte* blockBuffer, const char* fileName, uint bucket );

    static const char* DbgGetCommandName( Command::CommandType type );


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

    bool              _useDirectIO;
    ThreadPool        _threadPool;

    AutoResetSignal   _cmdReadySignal;
    AutoResetSignal   _cmdConsumedSignal;


};

//-----------------------------------------------------------
inline FileId TableIdToSortKeyId( const TableId table )
{
    switch( table )
    {
        case TableId::Table2: return FileId::SORT_KEY2;
        case TableId::Table3: return FileId::SORT_KEY3;
        case TableId::Table4: return FileId::SORT_KEY4;
        case TableId::Table5: return FileId::SORT_KEY5;
        case TableId::Table6: return FileId::SORT_KEY6;
        case TableId::Table7: return FileId::SORT_KEY7;
    
        default:
            ASSERT( 0 );
            break;
    }
    
    ASSERT( 0 );
    return FileId::None;
}

//-----------------------------------------------------------
inline FileId TableIdToBackPointerFileId( const TableId table )
{
    switch( table )
    {
        case TableId::Table2: return FileId::T2_L;
        case TableId::Table3: return FileId::T3_L;
        case TableId::Table4: return FileId::T4_L;
        case TableId::Table5: return FileId::T5_L;
        case TableId::Table6: return FileId::T6_L;
        case TableId::Table7: return FileId::T7_L;
    
        default:
            ASSERT( 0 );
            break;
    }
    
    ASSERT( 0 );
    return FileId::None;
}

