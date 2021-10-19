#include "DiskBufferQueue.h"
#include "Util.h"
#include "diskplot/DiskPlotConfig.h"
#include "SysHost.h"
#define NULL_BUFFER -1

//-----------------------------------------------------------
DiskBufferQueue::DiskBufferQueue( 
    const char* workDir, byte* workBuffer, 
    size_t workBufferSize, size_t chunkSize,
    uint ioThreadCount )
    : _workDir          ( workDir        )
    , _workHeap         ( workBuffer     )
    , _workHeapSize     ( workBufferSize )
    , _heapTableCapacity( 256u )
    , _heapTable        ( bbcalloc<HeapEntry>( _heapTableCapacity ), 1 )
    , _threadPool       ( ioThreadCount, ThreadPool::Mode::Fixed, true )
    , _releasedBuffers  ( 64 )
    , _dispatchThread()
{
    ASSERT( workDir    );
    ASSERT( workBuffer );

    // Initialize our heap table
    _heapTable.values[0].address = _workHeap;
    _heapTable.values[0].size    = _workHeapSize;


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
    free( _heapTable.values );
}

//-----------------------------------------------------------
void DiskBufferQueue::InitFileSet( FileId fileId, const char* name, uint bucketCount, char* pathBuffer, size_t workDirLength )
{
    char* baseName = pathBuffer + workDirLength;

    FileSet& fileSet = _files[(uint)fileId];
    
    fileSet.name         = name;
    fileSet.files.values = new FileStream[bucketCount];
    fileSet.files.length = bucketCount;

    for( uint i = 0; i < bucketCount; i++ )
    {
        FileStream& file = fileSet.files[i];

        sprintf( baseName, "%s_%u.tmp", name, i );
        if( !file.Open( pathBuffer, FileMode::Create, FileAccess::ReadWrite, FileFlags::NoBuffering | FileFlags::LargeFile ) )
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
void DiskBufferQueue::WriteBuckets( FileId id, const byte* buckets, const uint* sizes )
{
    Command* cmd = GetCommandObject();
    cmd->type            = Command::WriteBuckets;
    cmd->buckets.buffers = buckets;
    cmd->buckets.sizes   = sizes;
    cmd->buckets.fileId  = id;
}

//-----------------------------------------------------------
void DiskBufferQueue::ReleaseBuffer( byte* buffer )
{
    ASSERT( buffer );
    ASSERT( buffer >= _workHeap && buffer < _workHeap + _workHeapSize );

    Command* cmd = GetCommandObject();
    cmd->type                 = Command::ReleaseBuffer;
    cmd->releaseBuffer.buffer = buffer;
}

//-----------------------------------------------------------
byte* DiskBufferQueue::GetBuffer( size_t size )
{
    // First, add any pending released buffers back to the heap
    ConsumeReleasedBuffers();

    ASSERT( size <= _workHeapSize );

    // If we have such a buffer available, grab it, if not, we will
    // have to wait for enough deallocations in order to continue
    for( ;; )
    {
        byte* buffer = nullptr;

        for( size_t i = 0; i < _heapTable.length; i++ )
        {
            if( _heapTable[i].size >= size )
            {
                // Found a big enough slice to use
                _usedHeapSize += size;
                
                buffer = _heapTable[i].address;

                // #TODO: We need to track the size of our buffers for when they are released.

                if( size < _heapTable[i].size )
                {
                    // Split/fragment the current entry
                    _heapTable[i].address += size;
                    _heapTable[i].size    -= size;
                }
                else
                {
                    // We took the whole entry, remove it from the table
                    // and move down any subsequent entries.
                    const size_t remainder = --_heapTable.length - i;
                    if( remainder > 0 )
                        bbmemcpy_t( _heapTable.values + i, _heapTable.values + i + 1, remainder );
                }

                break;
            }
        }

        if( buffer )
            return buffer;

        // No buffer found, we have to wait until buffers are released and then try again
        _releasedBuffers.WaitForProduction();
        ConsumeReleasedBuffers();
    }
}

//-----------------------------------------------------------
void DiskBufferQueue::ConsumeReleasedBuffers()
{
    for( auto i = _releasedBuffers.Begin(); !i.Finished(); i.Next() )
    {
        HeapEntry& entry = _releasedBuffers.Get( i );

        if( _heapTable.length == 0 )
        {
            _heapTable[0] = entry;
            _heapTable.length++;
        }
        else
        {

        }
    }
}

//-----------------------------------------------------------
DiskBufferQueue::Command* DiskBufferQueue::GetCommandObject()
{
    int cmdCount = _cmdCount.load( std::memory_order_acquire );
    cmdCount += _cmdsPending;

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
void DiskBufferQueue::CommitCommands()
{
    ASSERT( _cmdsPending );

    int cmdCount = _cmdCount.load( std::memory_order_acquire );
    ASSERT( cmdCount < BB_DISK_QUEUE_MAX_CMDS );

    while( !_cmdCount.compare_exchange_weak( cmdCount, cmdCount + _cmdsPending,
                                             std::memory_order_release,
                                             std::memory_order_relaxed ) );
    _cmdsPending = 0;
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
                ExecuteCommand( _commands[i] );
            }

            i      = 0;
            cmdEnd = secondPassCount;
        }

        // Release commands
        int curCmdCount = cmdCount;
        while( !_cmdCount.compare_exchange_weak( curCmdCount, curCmdCount - cmdCount,
                                                 std::memory_order_release,
                                                 std::memory_order_relaxed ) );
        _cmdConsumedSignal.Signal();
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
        {
            ASSERT( cmd.releaseBuffer.buffer >= _workHeap && cmd.releaseBuffer.buffer < _workHeap + _workHeapSize );

            int bufferIndex = (int)( (uintptr_t)(cmd.releaseBuffer.buffer - _workHeap) / _chunkSize );
            ASSERT( (size_t)bufferIndex < _workHeapSize / _chunkSize );

            int curNextIndex = _nextBuffer.load( std::memory_order_acquire );
            _bufferList[bufferIndex] = curNextIndex;

            while( !_nextBuffer.compare_exchange_weak( curNextIndex, bufferIndex,
                                                       std::memory_order_release,
                                                       std::memory_order_relaxed ) )
            {
                curNextIndex             = _nextBuffer.load( std::memory_order_acquire );
                _bufferList[bufferIndex] = curNextIndex;
            }
        }
        break;

        default:
        break;
    }
}

//-----------------------------------------------------------
void DiskBufferQueue::CmdWriteBuckets( const Command& cmd )
{
    const FileId fileId  = cmd.buckets.fileId;
    const uint*  sizes   = cmd.buckets.sizes;
    const byte*  buffers = cmd.buckets.buffers;

    FileSet& fileBuckets = _files[(int)fileId];

    const uint bucketCount = (uint)fileBuckets.files.length;

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
                Fatal( "Failed to write to work %s work file.", fileBuckets.name );

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

