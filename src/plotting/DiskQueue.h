#pragma once
#include "threading/Thread.h"
#include "threading/AutoResetSignal.h"
#include "util/MPMCQueue.h"
#include "util/CommandQueue.h"

class IStream;
class Fence;
class DiskBufferBase;

struct DiskBufferCommand
{
    enum Type
    {
        None = 0,
        Write,
        Read,
    };

    Type type;

    union
    {
        struct {
            uint32 bucket;
        } write;

        struct {
            uint32 bucket;
        } read;
    };
};

struct DiskBucketBufferCommand
{
    enum Type
    {
        None = 0,
        Write,
        Read,
        // Seek,
        // Close,
    };

    Type type;

    union
    {
        struct {
            size_t sliceStride;
            uint32 bucket;
            bool   vertical;
        } write;

        struct {
            uint32 bucket;
            bool   vertical;
        } read;
    };
};

union DiskQueueDispatchCommand
{
    DiskBufferCommand       bufferCmd;
    DiskBucketBufferCommand bucketBufferCmd;
};

struct DiskQueueCommand
{
    static constexpr uint32 MAX_STACK_COMMANDS = 64;

    enum Type
    {
        None = 0,
        DispatchDiskBufferCommand,
        Signal,
    };

    Type type;

    union
    {
        struct {
            DiskBufferBase* sender;
            DiskQueueDispatchCommand cmd;
        } dispatch;

        struct {
            Fence* fence;
            uint64 value;
        } signal;
    };
};

class DiskQueue : public MPCommandQueue<DiskQueueCommand, DiskQueueCommand::MAX_STACK_COMMANDS>
{
    using Super = MPCommandQueue<DiskQueueCommand, DiskQueueCommand::MAX_STACK_COMMANDS>;

    friend class DiskBufferBase;
    friend class DiskBuffer;
    friend class DiskBucketBuffer;

public:
    DiskQueue( const char* path );
    ~DiskQueue();

    inline const char* Path() const { return _path.c_str(); }
    inline size_t      BlockSize() const { return _blockSize; }

protected:
    void ProcessCommands( const Span<DiskQueueCommand> items ) override;

private:
    void EnqueueDispatchCommand( DiskBufferBase* sender, const DiskQueueDispatchCommand& cmd );
    void SignalFence( Fence& fence, uint64 value );

private:
    std::string _path;          // Storage directory
    size_t      _blockSize = 0; // File system block size at path
};

