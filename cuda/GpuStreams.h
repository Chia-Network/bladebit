#pragma once
#include "CudaUtil.h"
#include "CudaPlotConfig.h"
#include "threading/Thread.h"
#include "threading/Fence.h"
#include "threading/Semaphore.h"
#include "util/SPCQueue.h"

//#define GPU_BUFFER_COUNT


// Represents a double-buffered device buffer, which can be used with a GpuStreamQueue to 
// make fast transfers (via intermediate pinned memory)

class IAllocator;

enum class GpuStreamKind : uint32
{
    Download = 0,
    Upload
};

typedef void (*GpuDownloadCallback)( void* hostBuffer, size_t downloadSize, void* userData );

struct GpuDownloadBuffer
{
    // Blocks the target stream buffer is available for kernel use
    void* LockDeviceBuffer( cudaStream_t stream );

    template<typename T>
    inline T* LockDeviceBuffer( cudaStream_t stream )
    {
        return reinterpret_cast<T*>( LockDeviceBuffer( stream ) );
    }

    // Blocks until the next device buffer is available for kernel use
    void* GetDeviceBuffer();

    template<typename T>
    inline T* GetDeviceBuffer()
    {
        return reinterpret_cast<T*>( GetDeviceBuffer() );
    }

    // Begin next async download and signal completion in the specified stream
    // so that the stream waits until it is signalled to continue work
    void Download( void* hostBuffer, size_t size, cudaStream_t workStream, bool directOverride = false );

    template<typename T>
    inline void DownloadT( T* hostBuffer, size_t count, cudaStream_t workStream, bool directOverride = false )
    {
        Download( hostBuffer, count * sizeof( T ), workStream, directOverride );
    }

    // Begin next async download call given the last device buffer used
    void Download( void* hostBuffer, size_t size );

    // Download and copy to another destination in a background thread
    void DownloadAndCopy( void* hostBuffer, void* finalBuffer, size_t size, cudaStream_t workStream  );

    template<typename T>
    inline void DownloadAndCopyT( T* hostBuffer, T* finalBuffer, const size_t count, cudaStream_t workStream  )
    {
        DownloadAndCopy( hostBuffer, finalBuffer, count * sizeof( T ), workStream  );
    }
    
    template<typename T>
    inline void DownloadT( T* hostBuffer, size_t count )
    {
        Download( hostBuffer, count * sizeof( T ) );
    }

    void DownloadTempAndCopy( void* hostBuffer, size_t size, cudaStream_t workStream );

    template<typename T>
    inline void DownloadTempAndCopyT( T* hostBuffer, const size_t size, cudaStream_t workStream )
    {
        return DownloadTempAndCopy( hostBuffer, size * sizeof( T ), workStream );
    }

    void DownloadWithCallback( void* hostBuffer, size_t size, GpuDownloadCallback callback, void* userData, cudaStream_t workStream = nullptr, bool directOverride = false );
    
    // Performs a direct host-to-pinned buffer copy,
    // and then a 2-dimensional copy from pinned buffer to host buffer
    //  - width    : Size in bytes of each row to copy
    //  - height   : How many rows to copy
    //  - dstStride: Offset, in bytes, to the start of each row in the destination buffer
    //  - srcStride: Offset, in bytes, to the start of each row in the source buffer
    void Download2D( void* hostBuffer, size_t width, size_t height, size_t dstStride, size_t srcStride, cudaStream_t workStream = nullptr, bool directOverride = false );

    void Download2DWithCallback( void* hostBuffer, size_t width, size_t height, size_t dstStride, size_t srcStride,
                                 GpuDownloadCallback callback, void* userData, cudaStream_t workStream = nullptr, bool directOverride = false );

    // Values of width, dstStride and srcStride are in element counts for this version
    template<typename T>
    inline void Download2DT( T* hostBuffer, size_t width, size_t height, size_t dstStride, size_t srcStride, cudaStream_t workStream = nullptr, bool directOverride = false )
    {
        Download2D( hostBuffer, width * sizeof( T ), height, dstStride * sizeof( T ), srcStride * sizeof( T ), workStream, directOverride );
    }

    // Performs several gpu-to-pinned downloads, then copies the pinned data as a contiguous buffer
    // to the destination host buffer
    void DownloadAndPackArray( void* hostBuffer, uint32 length, size_t srcStride, const uint32* counts, uint32 elementSize );

    template<typename T>
    inline void DownloadAndPackArray( T* hostBuffer, uint32 length, size_t srcStride, const uint32* counts )
    {
        DownloadAndPackArray( (void*)hostBuffer, length, srcStride, counts, sizeof( T ) );
    }

    // Wait for all downloads to complete.
    void WaitForCompletion();

    // Wait for copy to complete (when using DownloadAndCopy)
    void WaitForCopyCompletion();

    // Reset sequence id's.
    // This should only be used when no more events are pending.
    void Reset();

    class GpuQueue* GetQueue() const;

//private:
    struct IGpuBuffer* self;

private:
    void PerformDownload( void* hostBuffer, size_t width, size_t height, size_t dstStride, size_t srcStride,
                          GpuDownloadCallback callback, void* userData, cudaStream_t workStream, struct CopyInfo* copy = nullptr );

    void GetDownload2DCommand( void* hostBuffer, size_t width, size_t height, size_t dstStride, size_t srcStride, 
                               uint32& outIndex, void*& outPinnedBuffer, const void*& outDevBuffer, GpuDownloadCallback callback = nullptr, void* userData = nullptr );
};

struct GpuUploadBuffer
{
    void Upload( const void* hostBuffer, size_t size, cudaStream_t workStream );

    template<typename T>
    inline void UploadT( const T* hostBuffer, size_t count, cudaStream_t workStream )
    {
        Upload( hostBuffer, count * sizeof( T ), workStream );
    }

    void Upload( const void* hostBuffer, size_t size );

    template<typename T>
    inline void UploadT( const T* hostBuffer, size_t count )
    {
        Upload( hostBuffer, count * sizeof( T ) );
    }

    // Upload the host buffer, then copy the copyBufferSrc to the host buffer. Preloading
    // data into that hostBuffer (should be pinned) as soon as it is free so that memory is ready for the next upload.
    void UploadAndPreLoad( void* hostBuffer, size_t size, const void* copyBufferSrc, size_t copySize );
    
    template<typename T>
    inline void UploadAndPreLoadT( T* hostBuffer, const size_t count, const T* copyBufferSrc, const size_t copyCount )
    {
        UploadAndPreLoad( hostBuffer, count * sizeof( T ), copyBufferSrc, copyCount * sizeof( T ) );
    }

    void UploadArray( const void* hostBuffer, uint32 length, uint32 elementSize, uint32 srcStrideBytes, uint32 countStride, const uint32* counts, cudaStream_t workStream );
   
    template<typename T>
    inline void UploadArrayT( const T* hostBuffer, uint32 length, uint32 srcStride, uint32 countStride, const uint32* counts, cudaStream_t workStream )
    {
        UploadArray( hostBuffer, length, (uint32)sizeof( T ), srcStride * (uint32)sizeof( T ), countStride, counts, workStream );
    }


    void UploadArray( const void* hostBuffer, uint32 length, uint32 elementSize, uint32 srcStrideBytes, uint32 countStride, const uint32* counts );

    // srcStride here is in element count
    template<typename T>
    inline void UploadArrayT( const T* hostBuffer, uint32 length, uint32 srcStride, uint32 countStride, const uint32* counts )
    {
        UploadArray( hostBuffer, length, (uint32)sizeof( T ), srcStride * (uint32)sizeof( T ), countStride, counts );
    }


    void* GetUploadedDeviceBuffer( cudaStream_t workStream );

    template<typename T>
    inline T* GetUploadedDeviceBufferT( cudaStream_t workStream ) { return (T*)GetUploadedDeviceBuffer( workStream ); }

    // Waits until the earliest buffer has been uploaded to the GPU
    // and returns the device buffer.
    void* GetUploadedDeviceBuffer();

    template<typename T>
    inline T* GetUploadedDeviceBufferT() { return (T*)GetUploadedDeviceBuffer(); }

    // #TODO: Pass in the buffer used as a reference so that it can be nullified, for safety.
    void ReleaseDeviceBuffer( cudaStream_t workStream );
    // Wait for all uploads to complete.
    // After this is called, Synchronize should be called on the stream.
    //void WaitForCompletion();

    // Waits for preloaded data (via UploadAndPreLoad) to complete
    void WaitForPreloadsToComplete();

    // Reset sequence id's.
    // This should only be used when no more events are pending.
    void Reset();

    class GpuQueue* GetQueue() const;

//private:
    struct IGpuBuffer* self;

private:
    void* GetNextPinnedBuffer();
};


class GpuQueue
{
    friend struct IGpuBuffer;
    friend struct GpuDownloadBuffer;
    friend struct GpuUploadBuffer;

    enum class CommandType
    {
        None = 0,
        Copy,
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

    //void Synchronize();

    //GpuDownloadBuffer CreateDownloadBuffer( void* dev0, void* dev1, void* pinned0, void* pinned1, size_t size = 0, bool dryRun = false );
    //GpuDownloadBuffer CreateDownloadBuffer( const size_t size, bool dryRun = false );
    GpuDownloadBuffer CreateDirectDownloadBuffer( size_t size, IAllocator& devAllocator, size_t alignment, bool dryRun = false );
    GpuDownloadBuffer CreateDownloadBuffer( size_t size, IAllocator& devAllocator, IAllocator& pinnedAllocator, size_t alignment, bool dryRun = false );
    GpuDownloadBuffer CreateDownloadBuffer( size_t size, uint32 bufferCount, IAllocator& devAllocator, IAllocator& pinnedAllocator, size_t alignment, bool dryRun = false );

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
    struct IGpuBuffer* CreateGpuBuffer( size_t size, uint32 bufferCount, IAllocator* devAllocator, IAllocator* pinnedAllocator, size_t alignment, bool dryRun );
    //struct IGpuBuffer* CreateGpuBuffer( const size_t size );
    //struct IGpuBuffer* CreateGpuBuffer( void* dev0, void* dev1, void* pinned0, void* pinned1, size_t size );

    static void CopyPendingDownloadStream( void* userData );

    [[nodiscard]]
    Command& GetCommand( CommandType type );
    void SubmitCommands();

    // Copy threads
    static void CopyThreadEntryPoint( GpuQueue* self );
    virtual void CopyThreadMain();

    void ExecuteCommand( const Command& cpy );

    bool ShouldExitCopyThread();

protected:
    cudaStream_t             _stream;
    cudaStream_t             _preloadStream;
    Thread                   _copyThread;
    //Fence                    _bufferReadySignal;
    Semaphore                _bufferReadySignal;
    Fence                    _bufferCopiedSignal;
    Fence                    _syncFence;
    SPCQueue<Command, BBCU_BUCKET_COUNT*6> _queue;
    Kind                     _kind;

    AutoResetSignal          _waitForExitSignal;
    std::atomic<bool>        _exitCopyThread    = false;

    // Support multiple threads to grab commands
    std::atomic<uint64> _cmdTicketOut    = 0;
    std::atomic<uint64> _cmdTicketIn     = 0;
    std::atomic<uint64> _commitTicketOut = 0;
    std::atomic<uint64> _commitTicketIn  = 0;
};
