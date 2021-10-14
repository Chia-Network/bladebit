#include "DiskBufferQueue.h"
#include "Util.h"

#define NULL_BUFFER -1

//-----------------------------------------------------------
DiskBufferQueue::DiskBufferQueue( 
    const char* workDir, byte* workBuffer, 
    size_t workBufferSize, size_t chunkSize,
    uint ioThreadCount )
    : _workDir       ( workDir        )
    , _workBuffer    ( workBuffer     )
    , _workBufferSize( workBufferSize )
    , _chunkSize     ( chunkSize      )
    , _files         ( nullptr, 0     )
{
    ASSERT( workDir    );
    ASSERT( workBuffer );
    ASSERT( chunkSize <= workBufferSize / 2 );

    const int bufferCount = (int)( workBufferSize / chunkSize );

    // Populate free list of buffers
    int* freeList = (int*)malloc( bufferCount * sizeof( int ) );

    for( int i = 0; i < bufferCount-1; i++ )
        freeList[i] = i+1;

    freeList[bufferCount-1] = NULL_BUFFER;
    _nextBuffer = 0;
    _bufferList = Span<int>( freeList, (size_t)bufferCount );
}

//-----------------------------------------------------------
DiskBufferQueue::~DiskBufferQueue()
{
    free( _bufferList.values );
}

//-----------------------------------------------------------
byte* DiskBufferQueue::GetBuffer()
{
    int nextIdx = _nextBuffer.load( std::memory_order_acquire );

    if( nextIdx == NULL_BUFFER )
    {
        // #TODO: Spin for a while, then suspend, waiting to be signaled that a buffer has been published.
        //        For now always spin until we get a free buffer
        do
        {
            nextIdx = _nextBuffer.load( std::memory_order_acquire );
        } while( nextIdx == NULL_BUFFER );
    }

    ASSERT( nextIdx != NULL_BUFFER );

    // Grab the buffer
    // # NOTE: This should be safe as in a single-producer/consumer scenario.
    int newNextBuffer = _bufferList[nextIdx];
    while( !_nextBuffer.compare_exchange_weak( nextIdx, newNextBuffer,
                                                std::memory_order_release,
                                                std::memory_order_relaxed ) )
    {
        newNextBuffer = _bufferList[nextIdx];
    }

    return _workBuffer + (size_t)nextIdx * _chunkSize;
}

//-----------------------------------------------------------
DiskBufferQueue::Command* DiskBufferQueue::GetCommandObject()
{
    int cmdCount = _cmdCount.load( std::memory_order_acquire );

    // Have to wait until there's new commands
    if( cmdCount == BB_DISK_QUEUE_MAX_CMDS )
    {
        _cmdConsumedSignal.Wait();
        cmdCount =_cmdCount.load( std::memory_order_acquire );
        ASSERT( cmdCount < BB_DISK_QUEUE_MAX_CMDS );
    }

    Command* cmd = &_commands[_cmdWritePos];
    ZeroMem( cmd );

    ++_cmdWritePos %= BB_DISK_QUEUE_MAX_CMDS;
    _cmdsPending++;

    return cmd;
}

//-----------------------------------------------------------
void DiskBufferQueue::CommitCommand()
{
    ASSERT( _cmdsPending );

    int cmdCount = _cmdCount.load( std::memory_order_acquire );
    ASSERT( cmdCount < BB_DISK_QUEUE_MAX_CMDS );

    while( !_cmdCount.compare_exchange_weak( cmdCount, cmdCount + 1,
                                             std::memory_order_release,
                                             std::memory_order_relaxed ) )

    _cmdsPending = 0;
}

//-----------------------------------------------------------
void DiskBufferQueue::CommandThreadMain( DiskBufferQueue* self )
{
    self->CommandMain();
}

//-----------------------------------------------------------
void DiskBufferQueue::CommandMain()
{
    for( ;; )
    {
        _cmdReadySignal.Wait();

        const int cmdCount = _cmdCount.load( std::memory_order_acquire );
        ASSERT( cmdCount );

        if( cmdCount < 1 )
            continue;

        const int cmdWritePos = _cmdWritePos;

        int readPos = ( cmdWritePos - cmdCount + BB_DISK_QUEUE_MAX_CMDS ) % BB_DISK_QUEUE_MAX_CMDS;
        
        int cmdEnd          = std::min( BB_DISK_QUEUE_MAX_CMDS, readPos + cmdCount );
        int secondPassCount = readPos + cmdCount - cmdEnd;

        int i = readPos;
        for( int pass = 0; pass < 2; pass++ )
        {
            for( ; i < cmdEnd; i++ )
            {
                DispatchCommand( _commands[i] );
            }

            i = 0;
            cmdEnd = secondPassCount;
        }

    }
}

//-----------------------------------------------------------
void DiskBufferQueue::DispatchCommand( Command& cmd )
{

}