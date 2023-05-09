#include "GpuStreams.h"
#include "util/StackAllocator.h"

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

struct IGpuBuffer
{
    size_t            size;
    uint32            bufferCount;                                 // Number of pinned/device buffers this instance contains
    void*             deviceBuffer   [BBCU_GPU_BUFFER_MAX_COUNT];
    void*             pinnedBuffer   [BBCU_GPU_BUFFER_MAX_COUNT];  // Pinned host buffer
    cudaEvent_t       events         [BBCU_GPU_BUFFER_MAX_COUNT];  // Signals the device buffer is ready for use
    cudaEvent_t       completedEvents[BBCU_GPU_BUFFER_MAX_COUNT];  // Signals the buffer is ready for consumption by the device or buffer
    cudaEvent_t       readyEvents    [BBCU_GPU_BUFFER_MAX_COUNT];  // User must signal this event when the device buffer is ready for download
    // GpuQueue::Command commands       [BBCU_GPU_BUFFER_MAX_COUNT];  // Pending copy command for downloads
    Fence             fence;                                       // Signals the pinned buffer is ready for use
    Fence             copyFence;

    cudaEvent_t       preloadEvents[BBCU_GPU_BUFFER_MAX_COUNT];

    CopyInfo copies[BBCU_BUCKET_COUNT];
    PackedCopy packedCopeis[BBCU_BUCKET_COUNT];    // For uplad buffers
    // #TODO: Remove atomic again
    uint32     lockSequence;           // Index of next buffer to lock
    uint32     outgoingSequence;       // Index of locked buffer that will be downoaded/uploaded
    std::atomic<uint32>     completedSequence;      // Index of buffer that finished downloading/uploading
    std::atomic<uint32>     copySequence;

    GpuQueue* queue;
};


///
/// DownloadBuffer
///
void* GpuDownloadBuffer::GetDeviceBuffer()
{
    const uint32 index = self->outgoingSequence % self->bufferCount;

    CudaErrCheck( cudaEventSynchronize( self->events[index] ) );

    return self->deviceBuffer[index];
}

void* GpuDownloadBuffer::LockDeviceBuffer( cudaStream_t stream )
{
    ASSERT( self->lockSequence >= self->outgoingSequence );
    ASSERT( self->lockSequence - self->outgoingSequence < self->bufferCount );

    const uint32 index = self->lockSequence % self->bufferCount;
    self->lockSequence++;

    // Wait for the device buffer to be free to be used by kernels
    CudaErrCheck( cudaStreamWaitEvent( stream, self->events[index] ) );
    return self->deviceBuffer[index];
}

void GpuDownloadBuffer::Download( void* hostBuffer, const size_t size )
{
    Download2D( hostBuffer, size, 1, size, size );
}

void GpuDownloadBuffer::Download( void* hostBuffer, const size_t size, cudaStream_t workStream, bool directOverride )
{
    Download2D( hostBuffer, size, 1, size, size, workStream, directOverride );
}

void GpuDownloadBuffer::DownloadAndCopy( void* hostBuffer, void* finalBuffer, const size_t size, cudaStream_t workStream  )
{
    ASSERT( 0 );
    // ASSERT( self->outgoingSequence < BBCU_BUCKET_COUNT );
    // ASSERT( hostBuffer );
    // ASSERT( workStream );
    // ASSERT( self->lockSequence > 0 );
    // ASSERT( self->outgoingSequence < self->lockSequence );
    // ASSERT( self->lockSequence - self->outgoingSequence <= self->bufferCount );

    // auto& cpy = self->copies[self->outgoingSequence];
    // cpy.self            = self;
    // cpy.sequence        = self->outgoingSequence;
    // cpy.copy.hostBuffer = finalBuffer;
    // cpy.copy.srcBuffer  = hostBuffer;
    // cpy.copy.size       = size;


    // const uint32 index = self->outgoingSequence % self->bufferCount;
    // self->outgoingSequence++;

    //       void* pinnedBuffer = self->pinnedBuffer[index];
    // const void* devBuffer    = self->deviceBuffer[index];

    // // Signal from the work stream when it has finished doing kernel work with the device buffer
    // CudaErrCheck( cudaEventRecord( self->readyEvents[index], workStream ) );


    // // Ensure the work stream has completed writing data to the device buffer
    // cudaStream_t stream = self->queue->_stream;

    // CudaErrCheck( cudaStreamWaitEvent( stream, self->readyEvents[index] ) );

    // // Copy
    // CudaErrCheck( cudaMemcpyAsync( hostBuffer, devBuffer, size, cudaMemcpyDeviceToHost, stream ) );
    
    // // Signal that the device buffer is free to be re-used
    // CudaErrCheck( cudaEventRecord( self->events[index], stream ) );

    // // Launch copy command
    // CudaErrCheck( cudaLaunchHostFunc( stream, []( void* userData ){

    //     const CopyInfo& c = *reinterpret_cast<CopyInfo*>( userData );
    //     IGpuBuffer* self = c.self;

    //     auto& cmd = self->queue->GetCommand( GpuQueue::CommandType::Copy );
    //     cmd.copy.info = &c;

    //     self->queue->SubmitCommands();
        
    //     // Signal the download completed
    //     self->fence.Signal( ++self->completedSequence );
    // }, &cpy ) );
}

void GpuDownloadBuffer::DownloadWithCallback( void* hostBuffer, const size_t size, GpuDownloadCallback callback, void* userData, cudaStream_t workStream, bool directOverride )
{
    Download2DWithCallback( hostBuffer, size, 1, size, size, callback, userData, workStream, directOverride );
}

void GpuDownloadBuffer::Download2D( void* hostBuffer, size_t width, size_t height, size_t dstStride, size_t srcStride, cudaStream_t workStream, bool directOverride )
{
    Download2DWithCallback( hostBuffer, width, height, dstStride, srcStride, nullptr, nullptr, workStream, directOverride );
}

void GpuDownloadBuffer::Download2DWithCallback( void* hostBuffer, size_t width, size_t height, size_t dstStride, size_t srcStride, 
                                                GpuDownloadCallback callback, void* userData, cudaStream_t workStream, bool directOverride )
{
    ASSERT( hostBuffer );
    ASSERT( workStream );
    ASSERT( self->lockSequence > 0 );
    ASSERT( self->outgoingSequence < self->lockSequence );
    ASSERT( self->lockSequence - self->outgoingSequence <= self->bufferCount );

    const uint32 index = self->outgoingSequence % self->bufferCount;

          void* pinnedBuffer = self->pinnedBuffer[index];
    const void* devBuffer    = self->deviceBuffer[index];

    const bool isDirect = directOverride || self->pinnedBuffer[0] == nullptr;           ASSERT( isDirect || self->pinnedBuffer[0] );

    // Signal from the work stream when it has finished doing kernel work with the device buffer
    CudaErrCheck( cudaEventRecord( self->readyEvents[index], workStream ) );

    // Ensure the work stream has completed writing data to the device buffer
    cudaStream_t stream = self->queue->_stream;

    CudaErrCheck( cudaStreamWaitEvent( stream, self->readyEvents[index] ) );
    
    // Ensure the pinned buffer is ready for use
    if( !isDirect )
    {
        // CudaErrCheck( cudaStreamWaitEvent( stream, self->completedEvents[index] ) );
        CudaErrCheck( cudaLaunchHostFunc( stream, []( void* userData ){
            
            IGpuBuffer* self = reinterpret_cast<IGpuBuffer*>( userData );
            if( self->copySequence++ > 1 )
            {
                self->copyFence.Wait( self->copySequence-1 );
            }
        }, self ) );
    }

    // Copy from device to pinned host buffer
    const bool   isSequentialCopy = dstStride == srcStride;
    const size_t totalSize        = height * width;
    
    if( isDirect )
    {
        if( isSequentialCopy )
            CudaErrCheck( cudaMemcpyAsync( hostBuffer, devBuffer, totalSize, cudaMemcpyDeviceToHost, stream ) );
        else
            CudaErrCheck( cudaMemcpy2DAsync( hostBuffer, dstStride, devBuffer, srcStride, width, height, cudaMemcpyDeviceToHost, stream ) );

        // Signal direct download completed
        auto& cpy = self->copies[self->outgoingSequence];
        cpy.self      = self;
        cpy.sequence  = self->outgoingSequence;
        cpy.dstBuffer = hostBuffer;
        cpy.callback  = callback;
        cpy.userData  = userData;
        cpy.height    = height;
        cpy.width     = width;

        CudaErrCheck( cudaLaunchHostFunc( stream, []( void* userData ){

            CopyInfo&   cpy  = *reinterpret_cast<CopyInfo*>( userData );
            IGpuBuffer* self = cpy.self; //reinterpret_cast<IGpuBuffer*>( userData );

            self->fence.Signal( ++self->completedSequence );

            // Dispatch callback, if one was set
            if( cpy.callback )
                cpy.callback( cpy.dstBuffer, cpy.height * cpy.width, cpy.userData );

        }, &cpy ) );
    }
    else
    {
        CudaErrCheck( cudaMemcpyAsync( pinnedBuffer, devBuffer, totalSize, cudaMemcpyDeviceToHost, stream ) );
    }
    
    // Signal that the device buffer is free to be re-used
    CudaErrCheck( cudaEventRecord( self->events[index], stream ) );

    // If not a direct copy, we need to do another copy from the pinned buffer to the unpinned host buffer
    if( !isDirect )
    {
        // Signal the copy stream that the pinned buffer is ready to be copied to the unpinned host buffer
        CudaErrCheck( cudaEventRecord( self->preloadEvents[index], stream ) );

        // Ensure the pinned buffer is ready for use
        cudaStream_t copyStream = self->queue->_preloadStream;
        
        CudaErrCheck( cudaStreamWaitEvent( copyStream, self->preloadEvents[index] ) );

        {
            auto& cpy = self->copies[self->outgoingSequence];
            cpy.self     = self;
            cpy.sequence = self->outgoingSequence;

            cpy.dstBuffer = hostBuffer;
            cpy.srcBuffer = pinnedBuffer;
            cpy.width     = width;
            cpy.height    = height;
            cpy.srcStride = srcStride;
            cpy.dstStride = dstStride;
            cpy.callback  = callback;
            cpy.userData  = userData;

            CudaErrCheck( cudaLaunchHostFunc( copyStream, []( void* userData ){

                CopyInfo&   cpy  = *reinterpret_cast<CopyInfo*>( userData );
                IGpuBuffer* self = cpy.self; //reinterpret_cast<IGpuBuffer*>( userData );

                auto& cmd = self->queue->GetCommand( GpuQueue::CommandType::Copy );
                cmd.copy = &cpy;
                self->queue->SubmitCommands();

            }, &cpy ) );
        }

        // Signal the pinned buffer is free to be re-used
        // CudaErrCheck( cudaEventRecord( self->completedEvents[index], copyStream ) );
    }


    // Signal the download completed
    // {
    //     auto& cpy = self->copies[self->outgoingSequence];
    //     cpy.self     = self;
    //     cpy.sequence = self->outgoingSequence;
        
    //     cpy.copy2d.dstBuffer = hostBuffer;
    //     cpy.copy2d.srcBuffer = pinnedBuffer;
    //     cpy.copy2d.width     = width;
    //     cpy.copy2d.height    = height;
    //     cpy.copy2d.srcStride = srcStride;
    //     cpy.copy2d.dstStride = dstStride;

    //     CudaErrCheck( cudaLaunchHostFunc( copyStream, []( void* userData ){
            
    //         CopyInfo&   cpy  = *reinterpret_cast<CopyInfo*>( userData );
    //         IGpuBuffer* self = cpy.self; //reinterpret_cast<IGpuBuffer*>( userData );

    //         const uint32 idx = cpy.sequence & self->bufferCount;
            
    //         const byte* src = (byte*)cpy.copy2d.srcBuffer;
    //               byte* dst = (byte*)cpy.copy2d.dstBuffer;
            
    //         const size_t width     = cpy.copy2d.width;
    //         const size_t height    = cpy.copy2d.height;
    //         const size_t dstStride = cpy.copy2d.dstStride;
    //         const size_t srcStride = cpy.copy2d.srcStride;

    //         auto& cmd = self->queue->GetCommand( GpuQueue::CommandType::Download2D );
    //         cmd.sequenceId = cpy.sequence;
    //         cmd.srcBuffer  = src;
    //         cmd.dstBuffer  = dst;
    //         cmd.download2d.buf       = self;
    //         cmd.download2d.width     = width;
    //         cmd.download2d.height    = height;
    //         cmd.download2d.srcStride = srcStride;
    //         cmd.download2d.dstStride = dstStride;
    //         self->queue->SubmitCommands();

    //         // for( size_t i = 0; i < height; i++ )
    //         // {
    //         //     memcpy( dst, src, width );

    //         //     dst += dstStride;
    //         //     src += srcStride;
    //         // }

    //         // self->fence.Signal( ++self->completedSequence );
    //     }, &cpy ) );
    // }
    // CudaErrCheck( cudaEventRecord( self->completedEvents[index], copyStream ) );

    // if( callback )
    // {
    //     ASSERT( width <= srcStride );
    //     ASSERT( width <= dstStride );

    //     auto& cpy = self->copies[self->outgoingSequence];
    //     cpy.self                = self;
    //     cpy.sequence            = self->outgoingSequence;
    //     cpy.callback.hostBuffer = hostBuffer;
    //     cpy.callback.size       = width * height;
    //     cpy.callback.callback   = callback;
    //     cpy.callback.userData   = userData;

    //     CudaErrCheck( cudaLaunchHostFunc( stream, []( void* userData ){
            
    //         auto& cpy  = *reinterpret_cast<CopyInfo*>( userData );
    //         auto* self = cpy.self;

    //         // Fire callback command
    //         auto& cmd = self->queue->GetCommand( GpuQueue::CommandType::Callback );
    //         cmd.dstBuffer         = cpy.callback.hostBuffer;
    //         cmd.callback.copySize = cpy.callback.size;
    //         cmd.callback.callback = cpy.callback.callback;
    //         cmd.callback.userData = cpy.callback.userData;
    //         self->queue->SubmitCommands();

    //         // Signal the download completed
    //         self->fence.Signal( ++self->completedSequence );
    //     }, &cpy ) );
    // }
    // else
    // {
    //     // Signal the download completed
    //     CudaErrCheck( cudaLaunchHostFunc( stream, []( void* userData ){

    //         IGpuBuffer* self = reinterpret_cast<IGpuBuffer*>( userData );
    //         self->fence.Signal( ++self->completedSequence );
    //     }, self ) );
    // }

    self->outgoingSequence++;
}

void GpuDownloadBuffer::GetDownload2DCommand( void* hostBuffer, size_t width, size_t height, size_t dstStride, size_t srcStride,
                                              uint32& outIndex, void*& outPinnedBuffer, const void*& outDevBuffer, GpuDownloadCallback callback, void* userData )
{
    ASSERT( width      );
    ASSERT( height     );
    ASSERT( hostBuffer );

    const uint32 index = self->outgoingSequence % self->bufferCount;

    // We need to block until the pinned buffer is available.
    if( self->outgoingSequence > self->bufferCount-1 )
        self->fence.Wait( self->outgoingSequence - self->bufferCount + 1 );

          void* pinnedBuffer = self->pinnedBuffer[index];
    const void* devBuffer    = self->deviceBuffer[index];

    //auto& cmd = self->commands[index];
    //cmd.type             = GpuQueue::CommandType::Copy2D;
    //cmd.sequenceId       = self->outgoingSequence++;
    //cmd.finishedSignal   = &self->fence;
    //cmd.dstBuffer        = hostBuffer;
    //cmd.srcBuffer        = pinnedBuffer;
    //cmd.copy2d.width     = width;
    //cmd.copy2d.height    = height;
    //cmd.copy2d.dstStride = dstStride;
    //cmd.copy2d.srcStride = srcStride;
    //cmd.copy2d.callback  = callback;
    //cmd.copy2d.userData  = userData;

    outIndex        = index;
    outPinnedBuffer = pinnedBuffer;
    outDevBuffer    = devBuffer;
}


void GpuDownloadBuffer::DownloadAndPackArray( void* hostBuffer, const uint32 length, size_t srcStride, const uint32* counts, const uint32 elementSize )
{
    ASSERT( length      );
    ASSERT( elementSize );
    ASSERT( counts      );

    uint32 totalElements = 0;
    for( uint32 i = 0; i < length; i++ )
        totalElements += counts[i];

    const size_t totalSize = (size_t)totalElements * elementSize;

    uint32      index;
    void*       pinnedBuffer;
    const void* devBuffer;
    GetDownload2DCommand( hostBuffer, totalSize, 1, totalSize, totalSize, index, pinnedBuffer, devBuffer );


    srcStride *= elementSize;

          byte* dst = (byte*)pinnedBuffer;
    const byte* src = (byte*)devBuffer;

    cudaStream_t stream = self->queue->_stream;

    // Copy all buffers from device to pinned buffer
    for( uint32 i = 0; i < length; i++ )
    {
        const size_t copySize = counts[i] * (size_t)elementSize;

        // #TODO: Determine if there's a cuda (jagged) array copy
        CudaErrCheck( cudaMemcpyAsync( dst, src, copySize, cudaMemcpyDeviceToHost, stream ) );

        src += srcStride;
        dst += copySize;
    }

    // Signal that the device buffer is free
    CudaErrCheck( cudaEventRecord( self->events[index], stream ) );

    // Submit command to do the final copy from pinned to host
    CudaErrCheck( cudaLaunchHostFunc( stream, GpuQueue::CopyPendingDownloadStream, self ) );
}

void GpuDownloadBuffer::WaitForCompletion()
{
    if( self->outgoingSequence > 0 )
    {
        //const uint32 index = (self->outgoingSequence - 1) % self->bufferCount;

        //      cudaEvent_t event = self->completedEvents[index];
        //const cudaError_t r     = cudaEventQuery( event );

        //if( r == cudaSuccess )
        //    return;

        //if( r != cudaErrorNotReady )
        //    CudaErrCheck( r );

        //CudaErrCheck( cudaEventSynchronize( event ) );
        
        self->fence.Wait( self->outgoingSequence );
    }
}

void GpuDownloadBuffer::WaitForCopyCompletion()
{
    if( self->outgoingSequence > 0 )
    {
        self->copyFence.Wait( self->outgoingSequence );
    }
}

void GpuDownloadBuffer::Reset()
{
    self->lockSequence      = 0;
    self->outgoingSequence  = 0;
    self->completedSequence = 0;
    self->copySequence      = 0;
    self->fence.Reset( 0 );
    self->copyFence.Reset( 0 );
}

GpuQueue* GpuDownloadBuffer::GetQueue() const
{
    return self->queue;
}


///
/// UploadBuffer
///
void* GpuUploadBuffer::GetNextPinnedBuffer()
{
    // Wait for the pinned host buffer to be available
    //if( self->outgoingSequence > self->bufferCount-1 )
    //    self->fence.Wait( self->outgoingSequence - self->bufferCount + 1 );
    //
    const uint32 index = self->outgoingSequence % self->bufferCount;

    void* pinnedBuffer = self->pinnedBuffer[index];

    return pinnedBuffer;
}

void GpuUploadBuffer::Upload( const void* hostBuffer, size_t size, cudaStream_t workStream )
{
    ASSERT( hostBuffer );
    ASSERT( size );
    ASSERT( self->outgoingSequence - self->lockSequence < 2 );
    // ASSERT( workStream );
    
    const uint32 index = self->outgoingSequence % self->bufferCount;
    self->outgoingSequence++;

    auto stream = self->queue->GetStream();

    // Ensure the device buffer is ready for use
    CudaErrCheck( cudaStreamWaitEvent( stream, self->events[index] ) );

    // Upload to device buffer
    CudaErrCheck( cudaMemcpyAsync( self->deviceBuffer[index], hostBuffer, size, cudaMemcpyHostToDevice, stream ) );

    // Signal work stream that the device buffer is ready to be used
    CudaErrCheck( cudaEventRecord( self->readyEvents[index], stream ) );
}

void GpuUploadBuffer::UploadAndPreLoad( void* hostBuffer, const size_t size, const void* copyBufferSrc, const size_t copySize )
{
    ASSERT(0);
    // ASSERT( size >= copySize );

    // Upload( hostBuffer, size, nullptr );

    // // Add callback for copy
    // const uint32 sequence = self->outgoingSequence - 1;
    // auto& cpy = self->copies[sequence];
    // cpy.self            = self;
    // cpy.sequence        = sequence;
    // cpy.copy.hostBuffer = hostBuffer;
    // cpy.copy.srcBuffer  = copyBufferSrc;
    // cpy.copy.size       = copySize;

    // // Launch copy command
    // CudaErrCheck( cudaLaunchHostFunc( self->queue->GetStream(), []( void* userData ){

    //     const CopyInfo& c = *reinterpret_cast<CopyInfo*>( userData );
    //     IGpuBuffer* self = c.self;

    //     auto& cmd = self->queue->GetCommand( GpuQueue::CommandType::Copy );
    //     cmd.copy.info = &c;

    //     self->queue->SubmitCommands();
    // }, &cpy ) );
}

void GpuUploadBuffer::UploadArray( const void* hostBuffer, uint32 length, uint32 elementSize, uint32 srcStride, 
                                   uint32 countStride, const uint32* counts, cudaStream_t workStream )
{
    ASSERT( hostBuffer );
    ASSERT( self->outgoingSequence - self->lockSequence < 2 );

    const uint32 index = self->outgoingSequence % self->bufferCount;
    self->outgoingSequence++;

    auto stream = self->queue->GetStream();

    // Ensure the device buffer is ready for use
    CudaErrCheck( cudaStreamWaitEvent( stream, self->events[index] ) );

    // Perform uploads
    //size_t deviceCopySize = 0;
    const byte* src = (byte*)hostBuffer;
          byte* dst = (byte*)self->deviceBuffer[index];

    for( uint32 i = 0; i < length; i++ )
    {
        const size_t size = *counts * (size_t)elementSize;
        //memcpy( dst, src, size );
        CudaErrCheck( cudaMemcpyAsync( dst, src, size, cudaMemcpyHostToDevice, stream ) );

        //deviceCopySize += size;

        dst    += size;
        src    += srcStride;
        counts += countStride;
    }

    // Copy to device buffer
    //CudaErrCheck( cudaMemcpyAsync( self->deviceBuffer[index], cpy.dstBuffer, deviceCopySize, cudaMemcpyHostToDevice, _stream ) );

    // Signal work stream that the device buffer is ready to be used
    CudaErrCheck( cudaEventRecord( self->readyEvents[index], stream ) );
}

void GpuUploadBuffer::Upload( const void* hostBuffer, const size_t size )
{
    Upload( hostBuffer, size, nullptr );
}

void GpuUploadBuffer::UploadArray( const void* hostBuffer, uint32 length,
                                   uint32 elementSize, uint32 srcStride, uint32 countStride, const uint32* counts )
{
    UploadArray( hostBuffer, length, elementSize, srcStride, countStride, counts, nullptr );
}

void* GpuUploadBuffer::GetUploadedDeviceBuffer( cudaStream_t workStream )
{
    ASSERT( workStream );

    if( self->outgoingSequence < 1 )
    {
        ASSERT( 0 );
        return nullptr;
    }

    const uint32 index = self->completedSequence % self->bufferCount;
    self->completedSequence++;

    CudaErrCheck( cudaStreamWaitEvent( workStream, self->readyEvents[index] ) );

    return self->deviceBuffer[index];
}

void* GpuUploadBuffer::GetUploadedDeviceBuffer()
{ASSERT(0); // Not allowed for now
    if( self->outgoingSequence < 1 )
    {
        ASSERT( 0 );
        return nullptr;
    }
    ASSERT( 0 );
    const uint32 index = self->completedSequence % self->bufferCount;

    // #TODO: Make this spin way.
    // #TODO: Find a better way to do this instead of having to wait on both primitives.
    // Can't check the cuda event until we're sure it's been
    // added to the stream
    self->fence.Wait( self->completedSequence + 1 );
    CudaErrCheck( cudaEventSynchronize( self->events[index] ) );

    self->completedSequence++;

    return self->deviceBuffer[index];
}

void GpuUploadBuffer::ReleaseDeviceBuffer( cudaStream_t workStream )
{
    ASSERT( self->outgoingSequence > self->lockSequence );
    ASSERT( self->outgoingSequence - self->lockSequence <= 2 );
    ASSERT( self->completedSequence > 0 );

    const uint32 index = self->lockSequence % self->bufferCount;
    self->lockSequence++;

    CudaErrCheck( cudaEventRecord( self->events[index], workStream ) );
}

void GpuUploadBuffer::WaitForPreloadsToComplete()
{
    if( self->outgoingSequence > 0 )
    {
        self->copyFence.Wait( self->outgoingSequence );
    }
}

void GpuUploadBuffer::Reset()
{
    self->lockSequence      = 0;
    self->outgoingSequence  = 0;
    self->completedSequence = 0;
    self->copySequence      = 0;
    self->fence.Reset( 0 );
    self->copyFence.Reset( 0 );
}

GpuQueue* GpuUploadBuffer::GetQueue() const
{
    return self->queue;
}


///
/// Shared GpuStream Inteface
///
GpuQueue::GpuQueue( Kind kind ) : _kind( kind )
    , _bufferReadySignal( BBCU_BUCKET_COUNT )
{
    CudaErrCheck( cudaStreamCreateWithFlags( &_stream, cudaStreamNonBlocking ) );
    CudaErrCheck( cudaStreamCreateWithFlags( &_preloadStream, cudaStreamNonBlocking ) );

    _copyThread.Run( CopyThreadEntryPoint, this );
}

GpuQueue::~GpuQueue()
{
    _exitCopyThread.store( true, std::memory_order_release );
    _bufferReadySignal.Release();
    _waitForExitSignal.Wait();
}

//void GpuQueue::Synchronize()
//{
//    (void)GetCommand( CommandType::Sync );
//    SubmitCommands();
//
//    _syncFence.Wait();
//}


//GpuDownloadBuffer GpuQueue::CreateDownloadBuffer( void* dev0, void* dev1, void* pinned0, void* pinned1, size_t size, bool dryRun )
//{
//    FatalIf( _kind != Downloader, "Attempted to create GpuDownloadBuffer on an UploadQueue" );
//    if( dryRun ) return { nullptr };
//
//    // #TODO: Set size?
//    return { CreateGpuBuffer( dev0, dev1, pinned0, pinned1, size ) };
//}

//GpuDownloadBuffer GpuQueue::CreateDownloadBuffer( const size_t size, bool dryRun )
//{
//    FatalIf( _kind != Downloader, "Attempted to create GpuDownloadBuffer on an UploadQueue" );
//    if( dryRun ) return { nullptr };
//    return { CreateGpuBuffer( size ) };
//}

GpuDownloadBuffer GpuQueue::CreateDirectDownloadBuffer( const size_t size, IAllocator& devAllocator, const size_t alignment, const bool dryRun )
{
    FatalIf( _kind != Downloader, "Attempted to create GpuDownloadBuffer on an UploadQueue" );
    GpuDownloadBuffer r = { CreateGpuBuffer( size, BBCU_DEFAULT_GPU_BUFFER_COUNT, &devAllocator, nullptr, alignment, dryRun ) };

    if( !dryRun )
        r.Reset();

    return r;
}

GpuDownloadBuffer GpuQueue::CreateDownloadBuffer( const size_t size, IAllocator& devAllocator, IAllocator& pinnedAllocator, size_t alignment, bool dryRun )
{
    FatalIf( _kind != Downloader, "Attempted to create GpuDownloadBuffer on an UploadQueue" );
    GpuDownloadBuffer r = { CreateGpuBuffer( size, devAllocator, pinnedAllocator, alignment, dryRun ) };

    if( !dryRun )
        r.Reset();

    return r;
}

GpuDownloadBuffer GpuQueue::CreateDownloadBuffer( const size_t size, const uint32 bufferCount, IAllocator& devAllocator, IAllocator& pinnedAllocator, size_t alignment, bool dryRun )
{
    FatalIf( _kind != Downloader, "Attempted to create GpuDownloadBuffer on an UploadQueue" );
    GpuDownloadBuffer r = { CreateGpuBuffer( size, bufferCount, &devAllocator, &pinnedAllocator, alignment, dryRun ) };

    if( !dryRun )
        r.Reset();

    return r;
}

GpuUploadBuffer GpuQueue::CreateUploadBuffer( const size_t size, IAllocator& devAllocator, IAllocator& pinnedAllocator, size_t alignment, bool dryRun )
{
    FatalIf( _kind != Uploader, "Attempted to create GpuUploadBuffer on an DownloadQueue" );
    GpuUploadBuffer r = { CreateGpuBuffer( size, devAllocator, pinnedAllocator, alignment, dryRun ) };

    if( !dryRun )
        r.Reset();

    return r;
}


struct IGpuBuffer* GpuQueue::CreateGpuBuffer( const size_t size, IAllocator& devAllocator, IAllocator& pinnedAllocator, size_t alignment, bool dryRun )
{
    return CreateGpuBuffer( size, BBCU_DEFAULT_GPU_BUFFER_COUNT, &devAllocator, &pinnedAllocator, alignment, dryRun );
}

struct IGpuBuffer* GpuQueue::CreateGpuBuffer( const size_t size, const uint32 bufferCount, IAllocator* devAllocator, IAllocator* pinnedAllocator, size_t alignment, bool dryRun )
{
    FatalIf( bufferCount > BBCU_GPU_BUFFER_MAX_COUNT, "GPU Buffer count overflow." );

    const size_t allocSize = RoundUpToNextBoundaryT( size, alignment );

    void* devBuffers   [BBCU_GPU_BUFFER_MAX_COUNT] = {};
    void* pinnedBuffers[BBCU_GPU_BUFFER_MAX_COUNT] = {};

    for( int32 i = 0; i < bufferCount; i++ )
    {
        devBuffers[i] = devAllocator->Alloc( allocSize, alignment );

        if( pinnedAllocator )
            pinnedBuffers[i] = pinnedAllocator->Alloc( allocSize, alignment );
    }

    if( dryRun ) return nullptr;

    struct IGpuBuffer* buf = new IGpuBuffer{};

    for( int32 i = 0; i < bufferCount; i++ )
    {
        CudaErrCheck( cudaEventCreateWithFlags( &buf->events[i]         , cudaEventDisableTiming ) );
        CudaErrCheck( cudaEventCreateWithFlags( &buf->completedEvents[i], cudaEventDisableTiming ) );
        CudaErrCheck( cudaEventCreateWithFlags( &buf->readyEvents[i]    , cudaEventDisableTiming ) );
        CudaErrCheck( cudaEventCreateWithFlags( &buf->preloadEvents[i]  , cudaEventDisableTiming ) );
        
        buf->deviceBuffer[i] = devBuffers[i];
        buf->pinnedBuffer[i] = pinnedBuffers[i];
        // buf->commands[i]     = {};

        // Events have to be disabled initially for uploads
        //if( _kind == Uploader )
        //{
        //    CudaErrCheck( cudaEventSynchronize( buf->events[i]          ) );
        //    CudaErrCheck( cudaEventSynchronize( buf->completedEvents[i] ) );
        //    CudaErrCheck( cudaEventSynchronize( buf->readyEvents[i]     ) );
        //}
    }

    buf->size        = size;
    buf->bufferCount = bufferCount;
    buf->queue       = this;

    return buf;
}

//struct IGpuBuffer* GpuQueue::CreateGpuBuffer( void* dev0, void* dev1, void* pinned0, void* pinned1, const size_t size )
//{
//    ASSERT( dev0 );
//    ASSERT( dev1 );
//    ASSERT( pinned0 );
//    ASSERT( pinned1 );
//
//    ASSERT( dev0 != dev1 );
//    ASSERT( pinned0 != pinned1 );
//
//#if _DEBUG
//    if( size )
//    {
//        ASSERT_DOES_NOT_OVERLAP( dev0   , dev1   , size );
//        ASSERT_DOES_NOT_OVERLAP( dev0   , pinned0, size );
//        ASSERT_DOES_NOT_OVERLAP( dev0   , pinned1, size );
//        ASSERT_DOES_NOT_OVERLAP( dev1   , pinned0, size );
//        ASSERT_DOES_NOT_OVERLAP( dev1   , pinned1, size );
//        ASSERT_DOES_NOT_OVERLAP( pinned0, pinned1, size );
//    }
//#endif
//
//    struct IGpuBuffer* buf = new IGpuBuffer();
//
//    CudaErrCheck( cudaEventCreateWithFlags( &buf->events[0], cudaEventDisableTiming ) );
//    CudaErrCheck( cudaEventCreateWithFlags( &buf->events[1], cudaEventDisableTiming ) );
//
//    buf->deviceBuffer[0] = dev0;
//    buf->deviceBuffer[1] = dev1;
//
//    buf->pinnedBuffer[0] = pinned0;
//    buf->pinnedBuffer[1] = pinned1;
//
//    buf->size = size;
//    buf->fence.Reset( 0 );
//
//    buf->commands[0] = {};
//    buf->commands[1] = {};
//
//    buf->outgoingSequence  = 0;
//    buf->completedSequence = 0;
//
//    buf->queue = this;
//
//    return buf;
//}

//struct IGpuBuffer* GpuQueue::CreateGpuBuffer( const size_t size )
//{
//    ASSERT( size );
//
//    void* dev0;
//    void* dev1;
//    void* pinned0;
//    void* pinned1;
//
//    CudaErrCheck( cudaMalloc( &dev0, size ) );
//    CudaErrCheck( cudaMalloc( &dev1, size ) );
//    CudaErrCheck( cudaMallocHost( &pinned0, size ) );
//    CudaErrCheck( cudaMallocHost( &pinned1, size ) );
//
//    return CreateGpuBuffer( dev0, dev1, pinned0, pinned1, size );
//}

void GpuQueue::CopyPendingDownloadStream( void* userData )
{
    auto* buf = reinterpret_cast<IGpuBuffer*>( userData );

    GpuQueue* queue = buf->queue;

    //const uint32 index = buf->completedSequence % buf->bufferCount;
    buf->completedSequence++;

    //queue->GetCommand( CommandType::Download2D ) = buf->commands[index];
    queue->SubmitCommands();
}

void GpuQueue::SubmitCommands()
{
    const uint64 ticket = _commitTicketOut++;

    // Wait for our ticket to come up
    while( _commitTicketIn.load( std::memory_order_relaxed ) != ticket );

    _queue.Commit();
    _bufferReadySignal.Release();
    //_bufferReadySignal.Signal();

    // Use our ticket
    _commitTicketIn.store( ticket+1, std::memory_order_release );
}

GpuQueue::Command& GpuQueue::GetCommand( CommandType type )
{
    const uint64 ticket = _cmdTicketOut++;

    // Wait for our ticket to come up
    while( _cmdTicketIn.load( std::memory_order_relaxed ) != ticket );
    
    Command* cmd;
    while( !_queue.Write( cmd ) )
    {
        Log::Line( "[GpuQueue] Queue is depleted. Waiting for copies to complete." );
        auto waitTimer = TimerBegin();

        // Block and wait until we have commands free in the buffer
        _bufferCopiedSignal.Wait();
        
        Log::Line( "[GpuQueue] Waited %.6lf seconds for availability.", TimerEnd( waitTimer ) );
    }

    // Use our ticket
    _cmdTicketIn.store( ticket+1, std::memory_order_release );

    ZeroMem( cmd );
    cmd->type = type;

    return *cmd;
}


///
/// Command thread
///
void GpuQueue::CopyThreadEntryPoint( GpuQueue* self )
{
    ASSERT( self );
    self->CopyThreadMain();
    self->_waitForExitSignal.Signal();
}

void GpuQueue::CopyThreadMain()
{
    const int32 CMD_BUF_SIZE = 256;
    Command buffers[CMD_BUF_SIZE];

    for( ;; )
    {
        _bufferReadySignal.Wait();

        if( ShouldExitCopyThread() )
            return;

        // 1 command per semaphore release
        int32 bufCount;
        while( ( ( bufCount = _queue.Dequeue( buffers, CMD_BUF_SIZE ) ) ) )
        // if( ( ( bufCount = _queue.Dequeue( buffers, CMD_BUF_SIZE ) ) ) )
        {
            ASSERT( bufCount <= CMD_BUF_SIZE );
            _bufferCopiedSignal.Signal();

            for( int i = 0; i < bufCount; i++ )
                ExecuteCommand( buffers[i] );
        }
    }
}

void GpuQueue::ExecuteCommand( const Command& cmd )
{

    // const uint32 index = cmd.sequenceId % BBCU_GPU_BUFFER_MAX_COUNT;

    if( cmd.type == CommandType::Copy )
    {
        auto& cpy = *cmd.copy;

        const bool   isSequentialCopy = cpy.dstStride == cpy.srcStride;
        const size_t totalSize        = cpy.height * cpy.width;

              byte* dst = (byte*)cpy.dstBuffer;
        const byte* src = (byte*)cpy.srcBuffer;
        
        if( isSequentialCopy )
            memcpy( cpy.dstBuffer, cpy.srcBuffer, totalSize );
        else
        {
            const byte* src = (byte*)cpy.srcBuffer;
                  byte* dst = (byte*)cpy.dstBuffer;

            for( size_t i = 0; i < cpy.height; i++ )
            {
                memcpy( dst, src, cpy.width );

                dst += cpy.dstStride;
                src += cpy.srcStride;
            }
        }

        cpy.self->fence.Signal( cpy.sequence+1 );
        cpy.self->copyFence.Signal( cpy.sequence+1 );

        if( cpy.callback )
            cpy.callback( cpy.dstBuffer, totalSize, cpy.userData );
    }
    else if( cmd.type == CommandType::Callback )
    {
        cmd.callback.callback( cmd.callback.dstbuffer, cmd.callback.copySize, cmd.callback.userData );
    }
    // else if( cmd.type == CommandType::Sync )
    // {
    //     _syncFence.Signal();
    //     return;
    // }
    else
    {
        ASSERT( 0 );
    }

    // Signal that the pinned buffer is free
    //cpy.finishedSignal->Signal( cpy.sequenceId + 1 );
}

inline bool GpuQueue::ShouldExitCopyThread()
{
    return _exitCopyThread.load( std::memory_order_acquire );
}
