#pragma once
#include "CudaUtil.h"
#include "CudaPlotConfig.h"
#include "threading/Thread.h"
#include "threading/Fence.h"
#include "threading/Semaphore.h"
#include "util/SPCQueue.h"
#include "util/StackAllocator.h"
#include <functional>

class DiskBufferBase;
class DiskBuffer;
class DiskBucketBuffer;
struct GpuDownloadBuffer;
struct GpuUploadBuffer;
struct GpuQueue;

typedef std::function<void()> GpuStreamCallback;
typedef void (*GpuDownloadCallback)( void* hostBuffer, size_t downloadSize, void* userData );

struct PackedCopy
{
    struct IGpuBuffer* self;
    const  byte*       src;
           uint32      sequence;
           uint32      length;
           uint32      stride;
           uint32      elementSize;
           uint32      counts[BBCU_BUCKET_COUNT];
};

struct DiskDataInfo
{
    DiskBufferBase* diskBuffer;

    union {
        struct {
            GpuUploadBuffer* self;
            uint32           sequence;
        } uploadInfo;

        struct {
            size_t srcStride;
        } download2DInfo;

        struct {
            size_t size;
        } downloadSequentialInfo;
    };
};

struct CopyInfo
{
    struct IGpuBuffer* self;
    uint32             sequence;

    const void* srcBuffer;
    void*       dstBuffer;
    size_t      width;
    size_t      height;
    size_t      dstStride;
    size_t      srcStride;
    
    // Callback data
    GpuDownloadCallback callback;
    void*               userData;
};

// Represents a double-buffered device buffer, which can be used with a GpuStreamQueue to 
// make fast transfers (via intermediate pinned memory)

enum class GpuStreamKind : uint32
{
    Download = 0,
    Upload
};

struct IGpuBuffer
{
    size_t            size;
    uint32            bufferCount;                                 // Number of pinned/device buffers this instance contains
    void*             deviceBuffer[BBCU_GPU_BUFFER_MAX_COUNT];
    void*             pinnedBuffer[BBCU_GPU_BUFFER_MAX_COUNT];  // Pinned host buffer


    cudaEvent_t       pinnedEvent[BBCU_GPU_BUFFER_MAX_COUNT];   // Signals that the pinned buffer is ready for use

    union {
        cudaEvent_t   deviceEvents[BBCU_GPU_BUFFER_MAX_COUNT];  // Signals that the device buffer is ready for use
        cudaEvent_t   events      [BBCU_GPU_BUFFER_MAX_COUNT];  // Signals the device buffer is ready for use
    };


    union {
        cudaEvent_t workEvent      [BBCU_GPU_BUFFER_MAX_COUNT]; // Signals that the the work stream is done w/ the device buffer, and it's ready for use
        cudaEvent_t readyEvents    [BBCU_GPU_BUFFER_MAX_COUNT];  // User must signal this event when the device buffer is ready for download
    };
        cudaEvent_t completedEvents[BBCU_GPU_BUFFER_MAX_COUNT]; // Signals the buffer is ready for consumption by the device or buffer

    // For dispatching host callbacks.
    // Each buffer uses its own function?
    cudaEvent_t       callbackLockEvent;
    cudaEvent_t       callbackCompletedEvent;

    Fence             fence;                                       // Signals the pinned buffer is ready for use
    Fence             copyFence;

    cudaEvent_t       preloadEvents[BBCU_GPU_BUFFER_MAX_COUNT];


    CopyInfo copies[BBCU_BUCKET_COUNT];
    // union {
        // PackedCopy packedCopeis[BBCU_BUCKET_COUNT];    // For upload buffers
        DiskDataInfo diskData[BBCU_BUCKET_COUNT];
    // };
    // DiskBucketBuffer* diskBucketBuffer = nullptr;

    // #TODO: Remove atomic again
    uint32              lockSequence;           // Index of next buffer to lock
    uint32              outgoingSequence;       // Index of locked buffer that will be downloaded/uploaded
    std::atomic<uint32> completedSequence;      // Index of buffer that finished downloading/uploading
    std::atomic<uint32> copySequence;

    GpuQueue*       queue;      // Queue associated with this buffer
    DiskBufferBase* diskBuffer; // DiskBuffer, is any, used when using disk offload mode.
};



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

    template<typename T>
    inline void Download2DWithCallbackT( T* hostBuffer, size_t width, size_t height, size_t dstStride, size_t srcStride,
                                         GpuDownloadCallback callback, void* userData, cudaStream_t workStream = nullptr, bool directOverride = false )
    {
        Download2DWithCallback( 
            hostBuffer, width * sizeof( T ), height, dstStride * sizeof( T ), srcStride * sizeof( T ), 
            callback, userData, workStream, directOverride );
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

    DiskBufferBase* GetDiskBuffer() const;
    void AssignDiskBuffer( DiskBufferBase* diskBuffer );

    void HostCallback( std::function<void()> func );

//private:
    struct IGpuBuffer* self;

private:

    void PerformDownload2D( void* hostBuffer, size_t width, size_t height, size_t dstStride, size_t srcStride,
                            GpuDownloadCallback postCallback, void* postUserData, 
                            cudaStream_t workStream, bool directOverride );

    void PerformDownload( void* hostBuffer, size_t width, size_t height, size_t dstStride, size_t srcStride,
                          GpuDownloadCallback callback, void* userData, cudaStream_t workStream, struct CopyInfo* copy = nullptr );

    void GetDownload2DCommand( void* hostBuffer, size_t width, size_t height, size_t dstStride, size_t srcStride, 
                               uint32& outIndex, void*& outPinnedBuffer, const void*& outDevBuffer, GpuDownloadCallback callback = nullptr, void* userData = nullptr );

    void CallHostFunctionOnStream( cudaStream_t stream, std::function<void()> func );
};

struct GpuUploadBuffer
{
    void Upload( const void* hostBuffer, size_t size, cudaStream_t workStream, bool directOverride = false );

    template<typename T>
    inline void UploadT( const T* hostBuffer, size_t count, cudaStream_t workStream, bool directOverride = false  )
    {
        Upload( hostBuffer, count * sizeof( T ), workStream, directOverride );
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

    void UploadArrayForIndex( const uint32 index, const void* hostBuffer, uint32 length, 
                              uint32 elementSize, uint32 srcStride, uint32 countStride, const uint32* counts );

    // srcStride here is in element count
    template<typename T>
    inline void UploadArrayT( const T* hostBuffer, uint32 length, uint32 srcStride, uint32 countStride, const uint32* counts )
    {
        UploadArray( hostBuffer, length, (uint32)sizeof( T ), srcStride * (uint32)sizeof( T ), countStride, counts );
    }

    // Waits until the earliest buffer has been uploaded to the GPU
    // and returns the device buffer.
    void* GetUploadedDeviceBuffer( cudaStream_t workStream );

    template<typename T>
    inline T* GetUploadedDeviceBufferT( cudaStream_t workStream ) { return (T*)GetUploadedDeviceBuffer( workStream ); }

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

    void AssignDiskBuffer( DiskBufferBase* diskBuffer );
    DiskBufferBase* GetDiskBuffer() const;

    void CallHostFunctionOnStream( cudaStream_t stream, std::function<void()> func );


//private:
    struct IGpuBuffer* self;

private:
    uint32 SynchronizeOutgoingSequence();
    void* GetNextPinnedBuffer();
};

