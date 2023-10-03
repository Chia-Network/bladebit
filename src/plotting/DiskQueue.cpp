#include "DiskQueue.h"
#include "DiskBucketBuffer.h"
#include "io/FileStream.h"
#include "threading/Fence.h"
#include "plotdisk/jobs/IOJob.h"

DiskQueue::DiskQueue( const char* path )
    : Super()
    , _path( path )
{
    ASSERT( path );

    _blockSize = FileStream::GetBlockSizeForPath( path );
    FatalIf( _blockSize < 1, "Failed to obtain file system block size for path '%s'", path );

    StartConsumer();
}

DiskQueue::~DiskQueue()
{}

void DiskQueue::ProcessCommands( const Span<DiskQueueCommand> items )
{
    for( uint32 item = 0; item < items.Length(); item++ )
    {
        auto& cmd = items[item];

        switch( cmd.type )
        {
            case DiskQueueCommand::DispatchDiskBufferCommand:
                cmd.dispatch.sender->HandleCommand( cmd.dispatch.cmd );
                break;

            case DiskQueueCommand::Signal:
                cmd.signal.fence->Signal( (uint32)cmd.signal.value );
            break;

            default:
                ASSERT(0);
                break;
        }
    }
}

void DiskQueue::EnqueueDispatchCommand( DiskBufferBase* sender, const DiskQueueDispatchCommand& cmd )
{
    // #TODO: Don't copy and just have them send in a DiskQueueCommand?
    DiskQueueCommand c;
    c.type                    = DiskQueueCommand::DispatchDiskBufferCommand;
    c.dispatch.sender = sender;
    c.dispatch.cmd    = cmd;

    this->Submit( c );
}

void DiskQueue::SignalFence( Fence& fence, uint64 value )
{
    DiskQueueCommand c;
    c.type                    = DiskQueueCommand::Signal;
    c.signal.fence            = &fence;
    c.signal.value            = value;

    this->Submit( c );
}
