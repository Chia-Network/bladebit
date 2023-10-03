#pragma once

#include "GpuStreams.h"
#include <functional>

class DiskQueue;

struct GpuStreamDescriptor
{
    size_t      entrySize;
    size_t      entriesPerSlice;
    uint32      sliceCount;
    uint32      sliceAlignment;
    uint32      bufferCount;
    IAllocator* deviceAllocator;
    IAllocator* pinnedAllocator;
    DiskQueue*  diskQueue;          // DiskQueue to use when disk offload mode is enabled.
    const char* diskFileName;       // File name to use when disk offload mode is enabled. The diskQueue must be set.
    bool        bucketedDiskBuffer; // If true, a DiskBucketBuffer will be used instead of a DiskBuffer.
    bool        directIO;           // If true, direct I/O will be used when using disk offload mode.
};

typedef std::function<void()> GpuCallbackDispath;

class GpuQueue
{
    friend struct IGpuBuffer;
    friend struct GpuDownloadBuffer;
    friend struct GpuUploadBuffer;

    enum class CommandType
    {
        None = 0,
        Copy,
        CopyArray,
        Callback,
    };

    struct Command
    {
        CommandType type;

        union
        {
            struct CopyInfo* copy;

            struct {
                GpuDownloadCallback callback;
                size_t              copySize;
                void*               dstbuffer;
                void*               userData;
            } callback;
        };
    };

public:

    enum Kind
    {
        Downloader,
        Uploader
    };

    GpuQueue( Kind kind );
    virtual ~GpuQueue();

    static size_t CalculateSliceSizeFromDescriptor( const GpuStreamDescriptor& desc );
    static size_t CalculateBufferSizeFromDescriptor( const GpuStreamDescriptor& desc );

    //GpuDownloadBuffer CreateDownloadBuffer( void* dev0, void* dev1, void* pinned0, void* pinned1, size_t size = 0, bool dryRun = false );
    //GpuDownloadBuffer CreateDownloadBuffer( const size_t size, bool dryRun = false );
    GpuDownloadBuffer CreateDirectDownloadBuffer( size_t size, IAllocator& devAllocator, size_t alignment, bool dryRun = false );
    GpuDownloadBuffer CreateDownloadBuffer( size_t size, IAllocator& devAllocator, IAllocator& pinnedAllocator, size_t alignment, bool dryRun = false );
    GpuDownloadBuffer CreateDownloadBuffer( size_t size, uint32 bufferCount, IAllocator& devAllocator, IAllocator& pinnedAllocator, size_t alignment, bool dryRun = false );

    GpuDownloadBuffer CreateDownloadBuffer( const GpuStreamDescriptor& desc, bool dryRun = false );

    /// Create with descriptor and override entry size
    inline GpuDownloadBuffer CreateDownloadBuffer( const GpuStreamDescriptor& desc, size_t entrySize, bool dryRun = false )
    {
        GpuStreamDescriptor copy = desc;
        copy.entrySize = entrySize;

        return CreateDownloadBuffer( copy, dryRun );
    }

    template<typename T>
    inline GpuDownloadBuffer CreateDownloadBufferT( const GpuStreamDescriptor& desc, bool dryRun = false )
    {
        return CreateDownloadBuffer( desc, sizeof( T ), dryRun );
    }

    /// Create with descriptor and override entry size
    GpuUploadBuffer CreateUploadBuffer( const GpuStreamDescriptor& desc, bool dryRun = false );

    // inline GpuUploadBuffer CreateUploadBuffer( const GpuStreamDescriptor& desc, bool size_t entrySize, bool dryRun = false )
    // {
    //     GpuStreamDescriptor copy = desc;
    //     copy.entrySize = entrySize;

    //     return CreateUploadBuffer( copy, dryRun );
    // }

    template<typename T>
    inline GpuUploadBuffer CreateUploadBufferT( const GpuStreamDescriptor& desc, bool dryRun = false )
    {
        GpuStreamDescriptor copy = desc;
        copy.entrySize = sizeof(T);

        return CreateUploadBuffer( copy, dryRun );
        // return CreateUploadBuffer( desc, sizeof( T ), dryRun );
    }


    template<typename T>
    inline GpuDownloadBuffer CreateDirectDownloadBuffer( const size_t count, IAllocator& devAllocator, size_t alignment = alignof( T ), bool dryRun = false )
    {
        return CreateDirectDownloadBuffer( count * sizeof( T ), devAllocator, alignment, dryRun );
    }

    template<typename T>
    inline GpuDownloadBuffer CreateDownloadBufferT( const size_t count, IAllocator& devAllocator, IAllocator& pinnedAllocator, size_t alignment = alignof( T ), bool dryRun = false )
    {
        return CreateDownloadBuffer( count * sizeof( T ), devAllocator, pinnedAllocator, alignment, dryRun );
    }

    template<typename T>
    inline GpuDownloadBuffer CreateDownloadBufferT( const size_t count, uint32 bufferCount, IAllocator& devAllocator, IAllocator& pinnedAllocator, size_t alignment = alignof( T ), bool dryRun = false )
    {
        return CreateDownloadBuffer( count * sizeof( T ), bufferCount, devAllocator, pinnedAllocator, alignment, dryRun );
    }

    //GpuUploadBuffer CreateUploadBuffer( void* dev0, void* dev1, void* pinned0, void* pinned1, size_t size = 0, bool dryRun = false );
    //GpuUploadBuffer CreateUploadBuffer( const size_t size, bool dryRun = false );
    GpuUploadBuffer CreateUploadBuffer( const size_t size, IAllocator& devAllocator, IAllocator& pinnedAllocator, size_t alignment, bool dryRun = false );

    template<typename T>
    inline GpuUploadBuffer CreateUploadBufferT( const size_t count, IAllocator& devAllocator, IAllocator& pinnedAllocator, size_t alignment, bool dryRun = false )
    {
        return CreateUploadBuffer( count * sizeof( T ), devAllocator, pinnedAllocator, alignment, dryRun );
    }

    inline cudaStream_t GetStream() const { return _stream; }

protected:

    struct IGpuBuffer* CreateGpuBuffer( size_t size, IAllocator& devAllocator, IAllocator& pinnedAllocator, size_t alignment, bool dryRun );
    struct IGpuBuffer* CreateGpuBuffer( const GpuStreamDescriptor& desc, bool dryRun );
    
    void DispatchHostFunc( GpuCallbackDispath func, cudaStream_t stream, cudaEvent_t lockEvent, cudaEvent_t completedEvent );

    static void CopyPendingDownloadStream( void* userData );

    [[nodiscard]]
    Command& GetCommand( CommandType type );
    void SubmitCommands();

    // Copy threads
    static void QueueThreadEntryPoint( GpuQueue* self );
    void QueueThreadMain();

    void ExecuteCommand( const Command& cpy );

    bool ShouldExitQueueThread();

protected:
    cudaStream_t             _stream         = nullptr;
    cudaStream_t             _preloadStream  = nullptr;
    cudaStream_t             _callbackStream = nullptr;


    Thread                   _queueThread;
    //Fence                    _bufferReadySignal;
    Semaphore                _bufferReadySignal;
    Fence                    _bufferCopiedSignal;
    Fence                    _syncFence;
    SPCQueue<Command, BBCU_BUCKET_COUNT*6> _queue;
    Kind                     _kind;

    AutoResetSignal          _waitForExitSignal;
    std::atomic<bool>        _exitQueueThread    = false;

    // Support multiple threads to grab commands
    std::atomic<uint64> _cmdTicketOut    = 0;
    std::atomic<uint64> _cmdTicketIn     = 0;
    std::atomic<uint64> _commitTicketOut = 0;
    std::atomic<uint64> _commitTicketIn  = 0;
};
