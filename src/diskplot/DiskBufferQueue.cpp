#include "DiskBufferQueue.h"
#include "Util.h"
#include "diskplot/DiskPlotConfig.h"
#include "SysHost.h"

#include "util/Log.h"

#define NULL_BUFFER -1

//-----------------------------------------------------------
DiskBufferQueue::DiskBufferQueue( 
    const char* workDir, byte* workBuffer, 
    size_t workBufferSize, uint ioThreadCount,
    bool useDirectIO )
    : _workDir          ( workDir     )
    , _workHeap         ( workBufferSize, workBuffer )
    , _userDirectIO     ( useDirectIO )
    , _threadPool       ( ioThreadCount, ThreadPool::Mode::Fixed, true )
    , _dispatchThread()
{
    ASSERT( workDir );

    // Initialize Files
    size_t workDirLen = strlen( workDir );
    char*  pathBuffer = bbmalloc<char>( workDirLen + 64 );
    FatalIf( !pathBuffer, "Out of memory." );

    memcpy( pathBuffer, workDir, workDirLen + 1 );

    if( pathBuffer[workDirLen - 1] != '/' && pathBuffer[workDirLen - 1] != '\\' )
    {
        pathBuffer[workDirLen]   = '/';
        pathBuffer[++workDirLen] = '\0';
    }

    InitFileSet( FileId::Y, "y", BB_DP_BUCKET_COUNT, pathBuffer, workDirLen );
    InitFileSet( FileId::X, "x", BB_DP_BUCKET_COUNT, pathBuffer, workDirLen );
    free( pathBuffer );

    // Initialize I/O thread
    _dispatchThread.Run( CommandThreadMain, this );
}

//-----------------------------------------------------------
DiskBufferQueue::~DiskBufferQueue()
{
}

//-----------------------------------------------------------
void DiskBufferQueue::InitFileSet( FileId fileId, const char* name, uint bucketCount, char* pathBuffer, size_t workDirLength )
{
    char* baseName = pathBuffer + workDirLength;

    FileFlags flags = FileFlags::LargeFile;
    if( _userDirectIO )
        flags |= FileFlags::NoBuffering;

    FileSet& fileSet = _files[(uint)fileId];
    
    fileSet.name         = name;
    fileSet.files.values = new FileStream[bucketCount];
    fileSet.files.length = bucketCount;

    for( uint i = 0; i < bucketCount; i++ )
    {
        FileStream& file = fileSet.files[i];

        sprintf( baseName, "%s_%u.tmp", name, i );
        if( !file.Open( pathBuffer, FileMode::Create, FileAccess::ReadWrite, flags ) )
            Fatal( "Failed to open temp work file @ %s with error: %d.", pathBuffer, file.GetError() );

        if( !_blockBuffer )
        {
            _blockSize = file.BlockSize();
            FatalIf( _blockSize < 2, "Invalid temporary file block size." );
            
            _blockBuffer = (byte*)SysHost::VirtualAlloc( _blockSize, false );
            FatalIf( !_blockBuffer, "Out of memory." );
        }
        else
        {
            if( file.BlockSize() != _blockSize )
                Fatal( "Temporary work files have differing block sizes." );
        }
    }
}

//-----------------------------------------------------------
void DiskBufferQueue::WriteBuckets( FileId id, const void* buckets, const uint* sizes )
{
    Command* cmd = GetCommandObject();
    cmd->type            = Command::WriteBuckets;
    cmd->buckets.buffers = (byte*)buckets;
    cmd->buckets.sizes   = sizes;
    cmd->buckets.fileId  = id;
}

//-----------------------------------------------------------
void DiskBufferQueue::ReleaseBuffer( void* buffer )
{
    ASSERT( buffer );
    //ASSERT( buffer >= _workHeap && buffer < _workHeap + _workHeapSize );

    Command* cmd = GetCommandObject();
    cmd->type                 = Command::ReleaseBuffer;
    cmd->releaseBuffer.buffer = (byte*)buffer;
}

//-----------------------------------------------------------
void DiskBufferQueue::AddFence( AutoResetSignal& signal )
{
    Command* cmd = GetCommandObject();
    cmd->type         = Command::MemoryFence;
    cmd->fence.signal = &signal;
}

//-----------------------------------------------------------
byte* DiskBufferQueue::GetBuffer( size_t size )
{
    return _workHeap.Alloc( size, _blockSize );
}


//-----------------------------------------------------------
DiskBufferQueue::Command* DiskBufferQueue::GetCommandObject()
{
    Command* cmd;
    while( !_commands.Write( cmd ) )
    {
        #if _DEBUG
            Log::Debug( "[DiskBufferQueue] Command buffer full. Waiting for commands." );
            auto waitTimer = TimerBegin();
        #endif

        // Block and wait until we have commands free in the buffer
        // #TODO: We should track this and let the user know that he's running slow
        _cmdConsumedSignal.Wait();
        
        #if _DEBUG
            Log::Debug( "[DiskBufferQueue] Waited %.6lf seconds for a Command to be available.", TimerEnd( waitTimer ) );
        #endif
    }

    ZeroMem( cmd );
    return cmd;

//     int cmdCount = _cmdCount.load( std::memory_order_acquire );
//     cmdCount += _cmdsPending;
// 
//     // Have to wait until there's new commands
//     if( cmdCount == BB_DISK_QUEUE_MAX_CMDS )
//     {
//         _cmdConsumedSignal.Wait();
//         cmdCount =_cmdCount.load( std::memory_order_acquire );
//         ASSERT( cmdCount < BB_DISK_QUEUE_MAX_CMDS );
//     }
// 
//     Command* cmd = &_commands[_cmdWritePos];
//     ZeroMem( cmd );
// 
//     ++_cmdWritePos %= BB_DISK_QUEUE_MAX_CMDS;
//     _cmdsPending++;
// 
//     return cmd;
}

//-----------------------------------------------------------
void DiskBufferQueue::CommitCommands()
{
//     ASSERT( _cmdsPending );
// 
//     int cmdCount = _cmdCount.load( std::memory_order_acquire );
//     ASSERT( cmdCount < BB_DISK_QUEUE_MAX_CMDS );
// 
//     while( !_cmdCount.compare_exchange_weak( cmdCount, cmdCount + _cmdsPending,
//                                              std::memory_order_release,
//                                              std::memory_order_relaxed ) );
//     _cmdsPending = 0;
    Log::Debug( "Committing %d commands.", _commands._pendingCount );
    _commands.Commit();
    _cmdReadySignal.Signal();
}

//-----------------------------------------------------------
void DiskBufferQueue::CommandThreadMain( DiskBufferQueue* self )
{
    self->CommandMain();
}

//-----------------------------------------------------------
void DiskBufferQueue::CommandMain()
{
    const int CMD_BUF_SIZE = 64;
    Command commands[CMD_BUF_SIZE];

    for( ;; )
    {
        _cmdReadySignal.Wait();

        int cmdCount;
        while( cmdCount = _commands.Dequeue( commands, CMD_BUF_SIZE ) )
        {
            _cmdConsumedSignal.Signal();

            for( int i = 0; i < cmdCount; i++ )
                ExecuteCommand( commands[i] );
        }
        
//         const int cmdCount = _cmdCount.load( std::memory_order_acquire );
//         ASSERT( cmdCount );
// 
//         if( cmdCount < 1 )
//             continue;
// 
//         const int cmdWritePos = _cmdWritePos;
// 
//         int readPos = ( cmdWritePos - cmdCount + BB_DISK_QUEUE_MAX_CMDS ) % BB_DISK_QUEUE_MAX_CMDS;
//         
//         int cmdEnd          = std::min( BB_DISK_QUEUE_MAX_CMDS, readPos + cmdCount );
//         int secondPassCount = readPos + cmdCount - cmdEnd;
// 
//         int i = readPos;
//         for( int pass = 0; pass < 2; pass++ )
//         {
//             for( ; i < cmdEnd; i++ )
//             {
//                 ExecuteCommand( _commands[i] );
//             }
// 
//             i      = 0;
//             cmdEnd = secondPassCount;
//         }
// 
//         // Release commands
//         int curCmdCount = cmdCount;
//         while( !_cmdCount.compare_exchange_weak( curCmdCount, curCmdCount - cmdCount,
//                                                  std::memory_order_release,
//                                                  std::memory_order_relaxed ) );
//         _cmdConsumedSignal.Signal();
    }
}

//-----------------------------------------------------------
void DiskBufferQueue::ExecuteCommand( Command& cmd )
{
    switch( cmd.type )
    {
        case Command::WriteBuckets:
            CmdWriteBuckets( cmd );
        break;

        case Command::ReleaseBuffer:
            Log::Debug( " _Release 0x%p", cmd.releaseBuffer.buffer );
            _workHeap.Release( cmd.releaseBuffer.buffer );
        break;

        case Command::MemoryFence:
            ASSERT( cmd.fence.signal );
            cmd.fence.signal->Signal();
        break;

        default:
            ASSERT( 0 );
        break;
    }
}

//-----------------------------------------------------------
void DiskBufferQueue::CmdWriteBuckets( const Command& cmd )
{
    const FileId fileId      = cmd.buckets.fileId;
    const uint*  sizes       = cmd.buckets.sizes;
    const byte*  buffers     = cmd.buckets.buffers;

    FileSet&     fileBuckets = _files[(int)fileId];

    const uint   bucketCount = (uint)fileBuckets.files.length;

    Log::Debug( "  >>> Write 0x%p", buffers );

    // Single-threaded for now... We don't have file handles for all the threads yet!
    const byte* buffer = buffers;
    for( uint i = 0; i < bucketCount; i++ )
    {
        const size_t bucketSize  = sizes[i];
        FileStream&  file        = fileBuckets.files[i];
        
        size_t       sizeToWrite = bucketSize / _blockSize * _blockSize;
        const size_t remainder   = bucketSize - sizeToWrite;

        while( sizeToWrite )
        {
            ssize_t sizeWritten = file.Write( buffer, sizeToWrite );
            if( sizeWritten < 1 )
            {
                const int err = file.GetError();
                Fatal( "Failed to write to '%s.%u' work file with error %d (0x%x).", fileBuckets.name, i, err, err );
            }

            ASSERT( sizeWritten <= (ssize_t)sizeToWrite );
            sizeToWrite -= (size_t)sizeWritten;
            buffer += sizeWritten;
        }
        
        if( remainder )
        {
            // #TODO: We should only write in block-sized portions?? 
            // We can't! Because our buckets might not have enough...
            // We should only flush if the buckets are at blocks sized.
            // So for now, ignore this (improper data, yes, but this is just for testing times.
            memcpy( _blockBuffer, buffer, remainder );
        }
    }

//     MTJobRunner<WriteBucketsJob> job( _threadPool );
//     
//     const uint threadCount = _threadPool.ThreadCount();
// 
//     for( uint i = 0; i < threadCount; i++ )
//     {
//         FileStream& file = fileBuckets.files[i];
//                 
//         //MTJobRunner
//     }

}

