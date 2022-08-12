#pragma once

#include "io/IStream.h"
#include "threading/Fence.h"
#include "threading/ThreadPool.h"
#include "threading/MTJob.h"
#include "plotting/WorkHeap.h"
#include "plotting/Tables.h"
#include "FileId.h"

class Thread;
class IIOTransform;

enum FileSetOptions
{
    None        = 0,
    DirectIO    = 1 << 0,   // Use direct IO/unbuffered file IO
    Cachable    = 1 << 1,   // Use a in-memory cache for the file
    UseTemp2    = 1 << 2,   // Open the file set the high-frequency temp directory

    Interleaved = 1 << 3,   // Write in interleaved mode. That is all slices written to a single bucket.
    
    Alternating = 1 << 4,   // Alternate between bucket writing/reading modes. This allows for lower cache size.

    BlockAlign  = 1 << 5,   // Only write in block-aligned segments. Keeping a block-sized buffer for left overs.
                            // The last write flushes the whole block.
                            // This can be very memory-costly on file systems with large block sizes
                            // as interleaved buckets will need many block buffers.
                            // This must be used with DirectIO.
};
ImplementFlagOps( FileSetOptions );

struct FileSetInitData
{
    // Cachable
    void*  cache            = nullptr;  // Cache buffer
    size_t cacheSize        = 0;        // Cache size in bytes

    // For alternating mode
    uint64 maxSliceSize = 0;        // Maximum size (in bytes) of a bucket slice
};

struct FileSet
{
    const char*        name         = nullptr;
    Span<IStream*>     files;
    Span<IStream*>     readFiles;                            // When FileSetOptions::Alternating is enabled, we have to keep separate read streams
    void*              blockBuffer  = nullptr;               // For FileSetOptions::BlockAlign
    uint64             maxSliceSize = 0;                     // Maximum size (in bytes) of a bucket slice, for FileSetOptions::Alternating
    Span<Span<size_t>> readSliceSizes ;
    Span<Span<size_t>> writeSliceSizes;
    uint32             readBucket   = 0;                     // Current read/write bucket that generated slices. Valid when writing in interleaved mode and alternating mode
    uint32             writeBucket  = 0;
    FileSetOptions     options      = FileSetOptions::None;

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
            WriteBucketElements,
            ReadBucket,
            ReadFile,
            SeekFile,
            SeekBucket,
            DeleteFile,
            DeleteBucket,
            ReleaseBuffer,
            SignalFence,
            WaitForFence,
            TruncateBucket,

            DBG_WriteSliceSizes,    // Read/Write slice sizes to disk. Used for skipping tables
            DBG_ReadSliceSizes
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
                const uint* writeSizes;     // Size that will actually be written to disk (usually padded and block-aligned)
                const uint* sliceSizes;     // Actual number of slices that we have per-bucket. This used to store it for reading-back buckets.
                const byte* buffers;
                FileId      fileId;
                uint32      elementSize;    // Size of each element in the buffer
                bool        interleaved;    // Write interleaved or not?
            } buckets;

            struct {
                Span<byte>* buffer;
                FileId      fileId;
                uint32      elementSize;
                bool        interleaved;    // Read in interleaved mode
            } readBucket;

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

            struct
            {
                FileId  fileId;
                ssize_t position;
            } truncateBucket;

            #if _DEBUG
                struct
                {
                    TableId table;
                    FileId  fileId;
                } dbgSliceSizes;
            #endif
        };
    };

    struct FileDeleteCommand
    {
        FileId fileId;
        int64  bucket;  // If < 0, delete all buckets
    };

#if _DEBUG || BB_IO_METRICS_ON
public:
    struct IOMetric
    {
        size_t   size;
        Duration time;
        size_t   count;
    };
#endif

public:
    DiskBufferQueue( const char* workDir1, const char* workDir2, const char* plotDir,
                     byte* workBuffer, size_t workBufferSize, uint ioThreadCount,
                     int32 threadBindId = -1 );

    ~DiskBufferQueue();

    bool InitFileSet( FileId fileId, const char* name, uint bucketCount, const FileSetOptions options, const FileSetInitData* optsData  );

    bool InitFileSet( FileId fileId, const char* name, uint bucketCount );

    void SetTransform( FileId fileId, IIOTransform& transform );

    void OpenPlotFile( const char* fileName, const byte* plotId, const byte* plotMemo, uint16 plotMemoSize );

/// Commands
    void FinishPlot( Fence& fence );

    void ResetHeap( const size_t heapSize, void* heapBuffer );

    void WriteBuckets( FileId id, const void* buckets, const uint* writeSizes, const uint32* sliceSizes = nullptr );

    void WriteBucketElements( const FileId id, const bool interleaved, const void* buckets, const size_t elementSize, const uint32* writeCounts, const uint32* sliceCounts = nullptr );

    template<typename T>
    void WriteBucketElementsT( const FileId id, const bool interleaved, const T* buckets, const uint32* writeCounts, const uint32* sliceCounts = nullptr );

    void WriteFile( FileId id, uint bucket, const void* buffer, size_t size );

    void ReadBucketElements( const FileId id, const bool interleaved, Span<byte>& buffer, const size_t elementSize );

    template<typename T>
    void ReadBucketElementsT( const FileId id, const bool interleaved, Span<T>& buffer );

    void ReadFile( FileId id, uint bucket, void* dstBuffer, size_t readSize );

    void SeekFile( FileId id, uint bucket, int64 offset, SeekOrigin origin );

    void SeekBucket( FileId id, int64 offset, SeekOrigin origin );

    void DeleteFile( FileId id, uint bucket );

    void DeleteBucket( FileId id );

    void TruncateBucket( FileId id, const ssize_t position );

    // Add a memory fence into the command stream.
    // The signal will be set once this command is reached.
    // This ensures the caller that all commands before the
    // fence have been processed.
    void SignalFence( Fence& fence );

    // Signal a fence with a specific value.
    void SignalFence( Fence& fence, uint32 value );

    // Instructs the command dispatch thread to wait until the specified fence has been signalled
    void WaitForFence( Fence& fence );

    void CommitCommands();

// Helpers
    // Obtain a buffer allocated from the work heap.
    // May block until there's a buffer available if there was none.
    // This assumes a single consumer.
    inline byte* GetBuffer( size_t size, bool blockUntilFreeBuffer = true )
    { 
        return _workHeap.Alloc( size, 1 /*_blockSize*/, blockUntilFreeBuffer, &_ioBufferWaitTime ); 
    }

    inline byte* GetBuffer( const size_t size, const size_t alignment, bool blockUntilFreeBuffer = true )
    { 
        return _workHeap.Alloc( size, alignment, blockUntilFreeBuffer, &_ioBufferWaitTime ); 
    }

    #if _DEBUG || BB_TEST_MODE
        // const Span<Span<size_t>> SliceSizes( const FileId fileId ) const { return _files[(int)fileId].sliceSizes; }
    #endif

    #if _DEBUG
        void DebugWriteSliceSizes( const TableId table, const FileId fileId );
        void DebugReadSliceSizes( const TableId table, const FileId fileId );
    #endif

    // byte* GetBufferForId( const FileId fileId, const uint32 bucket, const size_t size, bool blockUntilFreeBuffer = true );

    // Release/return a chunk buffer that was in use, gotten by GetBuffer()
    // These returns the buffer back to the queue so that it is in use.
    // This command is serialized and should be added after 
    // any writing/reading has finished with said buffer
    void ReleaseBuffer( void* buffer );

    void CompletePendingReleases();

    inline size_t BlockSize() const { return _blockSize; }
    size_t BlockSize( FileId fileId ) const;
    
    inline const WorkHeap& Heap() const { return _workHeap; }

    inline size_t PlotHeaderSize() const { return _plotHeaderSize; }

    inline uint64 PlotTablePointersAddress() const { return _plotTablesPointers; }

    inline double IOBufferWaitTime() const { return TicksToSeconds( _ioBufferWaitTime ); }
    inline void ResetIOBufferWaitCounter() { _ioBufferWaitTime = Duration::zero(); }


    #if _DEBUG || BB_IO_METRICS_ON
    //-----------------------------------------------------------
    inline const IOMetric& GetReadMetrics() const { return _readMetrics; }
    inline const IOMetric& GetWriteMetrics() const { return _writeMetrics;}

    //-----------------------------------------------------------
    inline double GetAverageReadThroughput() const
    {
        const double elapsed    = TicksToSeconds( _readMetrics.time ) / (double)_readMetrics.count; 
        const double throughput = (double)_readMetrics.size / (double)_readMetrics.count / elapsed;
        return throughput;
    }

    //-----------------------------------------------------------
    inline double GetAverageWriteThroughput() const
    {
        const double elapsed    = TicksToSeconds( _writeMetrics.time ) / (double)_writeMetrics.count; 
        const double throughput = (double)_writeMetrics.size / (double)_writeMetrics.count / elapsed;
        return throughput;
    }

    //-----------------------------------------------------------
    inline void DumpDiskMetrics( const TableId table )
    {
        const double readThroughput  = GetAverageReadThroughput();
        const auto&  reads           = GetReadMetrics();
        const double writeThroughput = GetAverageWriteThroughput();
        const auto&  writes          = GetWriteMetrics();

        Log::Line( " Table %u I/O Metrics:", (uint32)table+1 );
        
        Log::Line( "  Average read throughput %.2lf MiB ( %.2lf MB ) or %.2lf GiB ( %.2lf GB ).", 
            readThroughput BtoMB, readThroughput / 1000000.0, readThroughput BtoGB, readThroughput / 1000000000.0 );
        Log::Line( "  Total size read: %.2lf MiB ( %.2lf MB ) or %.2lf GiB ( %.2lf GB ).",
            (double)reads.size BtoMB, (double)reads.size / 1000000.0, (double)reads.size BtoGB, (double)reads.size / 1000000000.0 );
        Log::Line( "  Total read commands: %llu.", (llu)reads.count );
        
        Log::Line( "  Average write throughput %.2lf MiB ( %.2lf MB ) or %.2lf GiB ( %.2lf GB ).", 
            writeThroughput BtoMB, writeThroughput / 1000000.0, writeThroughput BtoGB, writeThroughput / 1000000000.0 );
        Log::Line( "  Total size written: %.2lf MiB ( %.2lf MB ) or %.2lf GiB ( %.2lf GB ).",
            (double)writes.size BtoMB, (double)writes.size / 1000000.0, (double)writes.size BtoGB, (double)writes.size / 1000000000.0 );
        Log::Line( "  Total write commands: %llu.", (llu)writes.count );
        Log::Line( "" );

        ClearReadMetrics();
        ClearWriteMetrics();
    }
    
    //-----------------------------------------------------------
    inline void DumpReadMetrics( const TableId table )
    {
        const double readThroughput  = GetAverageReadThroughput();
        const auto&  reads           = GetReadMetrics();

        Log::Line( " Table %u Disk Read Metrics:", (uint32)table+1 );
        
        Log::Line( "  Average read throughput %.2lf MiB ( %.2lf MB ) or %.2lf GiB ( %.2lf GB ).", 
            readThroughput BtoMB, readThroughput / 1000000.0, readThroughput BtoGB, readThroughput / 1000000000.0 );
        Log::Line( "  Total size read: %.2lf MiB ( %.2lf MB ) or %.2lf GiB ( %.2lf GB ).",
            (double)reads.size BtoMB, (double)reads.size / 1000000.0, (double)reads.size BtoGB, (double)reads.size / 1000000000.0 );
        Log::Line( "  Total read commands: %llu.", (llu)reads.count );

        ClearReadMetrics();
    }

    //-----------------------------------------------------------
    inline void DumpWriteMetrics(  const TableId table )
    {
        const double writeThroughput = GetAverageWriteThroughput();
        const auto&  writes          = GetWriteMetrics();

        Log::Line( " Table %u Disk Write Metrics:", (uint32)table+1 );
        
        Log::Line( "  Average write throughput %.2lf MiB ( %.2lf MB ) or %.2lf GiB ( %.2lf GB ).", 
            writeThroughput BtoMB, writeThroughput / 1000000.0, writeThroughput BtoGB, writeThroughput / 1000000000.0 );
        Log::Line( "  Total size written: %.2lf MiB ( %.2lf MB ) or %.2lf GiB ( %.2lf GB ).",
            (double)writes.size BtoMB, (double)writes.size / 1000000.0, (double)writes.size BtoGB, (double)writes.size / 1000000000.0 );
        Log::Line( "  Total write commands: %llu.", (llu)writes.count );
        Log::Line( "" );
        
        ClearWriteMetrics();
    }

    //-----------------------------------------------------------
    inline void ClearReadMetrics()
    {
        _readMetrics = {};
    }

    //-----------------------------------------------------------
    inline void ClearWriteMetrics()
    {
        _writeMetrics = {};
    }
    #else
    inline void DumpWriteMetrics( const TableId table ) {}
    inline void DumpReadMetrics( const TableId table  ) {}
    inline void DumpDiskMetrics( const TableId table  ){}
    inline void ClearWriteMetrics(){}
    inline void ClearReadMetrics(){}
    #endif
    
private:

    Command* GetCommandObject( Command::CommandType type );

    static void CommandThreadMain( DiskBufferQueue* self );
    void CommandMain();

    static void DeleterThreadMain( DiskBufferQueue* self );
    void DeleterMain();

    void ExecuteCommand( Command& cmd );

    void CmdWriteBuckets( const Command& cmd, const size_t elementSize );
    void CndWriteFile( const Command& cmd );
    void CmdReadBucket( const Command& cmd );
    void CmdReadFile( const Command& cmd );
    void CmdSeekBucket( const Command& cmd );

    void WriteToFile( IStream& file, size_t size, const byte* buffer, byte* blockBuffer, const char* fileName, uint bucket );
    void ReadFromFile( IStream& file, size_t size, byte* buffer, byte* blockBuffer, const size_t blockSize, const bool directIO, const char* fileName, const uint bucket );

    void CmdDeleteFile( const Command& cmd );
    void CmdDeleteBucket( const Command& cmd );

    void CmdTruncateBucket( const Command& cmd );

    void CloseFileNow( const FileId fileId, const uint32 bucket );
    void DeleteFileNow( const FileId fileId, const uint32 bucket );
    void DeleteBucketNow( const FileId fileId );

    static const char* DbgGetCommandName( Command::CommandType type );

    #if _DEBUG
        void CmdDbgWriteSliceSizes( const Command& cmd );
        void CmdDbgReadSliceSizes( const Command& cmd );
    #endif


private:
    std::string      _workDir1;     // Temporary 1 directory in which we will store our long-lived temporary files
    std::string      _workDir2;     // Temporary 2 directory in which we will store our short-live, high-req I/O temporary files
    std::string      _plotDir;      // Temporary plot directory
    std::string      _plotFullName; // Full path of the plot file without '.tmp'

    WorkHeap         _workHeap;     // Reserved memory for performing plot work and I/O // #TODO: Remove this
    
    // Handles to all files needed to create a plot
    FileSet          _files[(size_t)FileId::_COUNT];
    size_t           _blockSize          = 0;
    byte*            _t1BlockBuffer      = nullptr;         // Temporary temp1 dir block buffer user for slice reading
    byte*            _t2BlockBuffer      = nullptr;         // Temporary temp2 dir block buffer user for slice reading
    
    char*            _filePathBuffer     = nullptr;         // For creating file sets

    size_t           _plotHeaderSize     = 0;
    byte*            _plotHeaderbuffer   = nullptr;
    uint64           _plotTablesPointers = 0;               // Offset in the plot file to the tables pointer table

    Duration         _ioBufferWaitTime = Duration::zero();  // Total time spent waiting for IO buffers.

    // I/O thread stuff
    Thread            _dispatchThread;
    
    SPCQueue<Command, BB_DISK_QUEUE_MAX_CMDS> _commands;

    // ThreadPool        _threadPool;

    AutoResetSignal   _cmdReadySignal;
    AutoResetSignal   _cmdConsumedSignal;

    // File deleter thread
    Thread            _deleterThread;                   // For deleting files.
    AutoResetSignal   _deleteSignal;                    // We do this in a separate thread as to not
    GrowableSPCQueue<FileDeleteCommand> _deleteQueue;   // block other commands when the kernel is clearing cached IO buffers for the files.
    char*             _delFilePathBuffer = nullptr;     // For deleting file sets
    bool              _deleterExit       = false;
    int32             _threadBindId;

#if _DEBUG || BB_IO_METRICS_ON
    IOMetric _readMetrics  = {};
    IOMetric _writeMetrics = {};
#endif
};


//-----------------------------------------------------------
template<typename T>
inline void DiskBufferQueue::WriteBucketElementsT( const FileId id, const bool interleaved, const T* buckets, const uint32* writeCounts, const uint32* sliceCounts  )
{
    WriteBucketElements( id, interleaved, (byte*)buckets, sizeof( T ), writeCounts, sliceCounts );
}

//-----------------------------------------------------------
template<typename T>
inline void DiskBufferQueue::ReadBucketElementsT( const FileId id, const bool interleaved, Span<T>& buffer )
{
    ReadBucketElements( id, interleaved, reinterpret_cast<Span<byte>&>( buffer ), sizeof( T ) );
}