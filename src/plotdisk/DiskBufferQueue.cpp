#include "DiskBufferQueue.h"
#include "io/FileStream.h"
#include "io/HybridStream.h"
#include "plotdisk/DiskPlotConfig.h"
#include "plotdisk/IOTransforms.h"
#include "jobs/IOJob.h"
#include "util/Util.h"
#include "util/Log.h"


#define NULL_BUFFER -1


//-----------------------------------------------------------
DiskBufferQueue::DiskBufferQueue( 
    const char* workDir, byte* workBuffer, 
    size_t workBufferSize, uint ioThreadCount,
    bool useDirectIO
)
    : _workDir       ( workDir     )
    , _workHeap      ( workBufferSize, workBuffer )
    , _useDirectIO   ( useDirectIO )
    // , _threadPool    ( ioThreadCount, ThreadPool::Mode::Fixed, true )
    , _dispatchThread()
    , _deleterThread ()
    , _deleteSignal  ()
    , _deleteQueue   ( 128 )
{
    ASSERT( workDir );

    // Initialize Files
    const size_t workDirLen = _workDir.length();

    const size_t PLOT_FILE_LEN = sizeof( "/plot-k32-2021-08-05-18-55-77a011fc20f0003c3adcc739b615041ae56351a22b690fd854ccb6726e5f43b7.plot.tmp" );

    _filePathBuffer = bbmalloc<char>( workDirLen + PLOT_FILE_LEN );  // Should be enough for all our file names

    memcpy( _filePathBuffer, workDir, workDirLen + 1 );

    if( _filePathBuffer[workDirLen-1] != '/' && _filePathBuffer[workDirLen-1] != '\\' )
    {
        _filePathBuffer[workDirLen]   = '/';
        _filePathBuffer[workDirLen+1] = '\0';
        _workDir += "/";
    }

    // Initialize file deleter thread
    _deleterThread.Run( DeleterThreadMain, this );

    // Initialize I/O thread
    _dispatchThread.Run( CommandThreadMain, this );

}

//-----------------------------------------------------------
DiskBufferQueue::~DiskBufferQueue()
{
    _deleterExit = true;
    _deleteSignal.Signal();
    _deleterThread.WaitForExit();

    // #TODO: Wait for command thread

    // #TODO: Delete our file sets
    free( _filePathBuffer );
}

//-----------------------------------------------------------
void DiskBufferQueue::ResetHeap( const size_t heapSize, void* heapBuffer )
{
    _workHeap.ResetHeap( heapSize, heapBuffer );
}

//-----------------------------------------------------------
bool DiskBufferQueue::InitFileSet( FileId fileId, const char* name, uint bucketCount )
{
    return InitFileSet( fileId, name, bucketCount, FileSetOptions::None, nullptr );
}

//-----------------------------------------------------------
bool DiskBufferQueue::InitFileSet( FileId fileId, const char* name, uint bucketCount, const FileSetOptions options, const FileSetInitData* optsData )
{
    char*        pathBuffer    = _filePathBuffer;
    const size_t workDirLength = _workDir.length();

    char* baseName = pathBuffer + workDirLength;

    FileFlags flags = FileFlags::LargeFile;
    if( IsFlagSet( options, FileSetOptions::DirectIO ) )
        flags |= FileFlags::NoBuffering;

    FileSet& fileSet = _files[(uint)fileId];
    ASSERT( !fileSet.files.values );

    fileSet.name         = name;
    fileSet.files.values = new IStream*[bucketCount];
    fileSet.files.length = bucketCount;
    fileSet.blockBuffers = nullptr;
    fileSet.options      = options;

    // #TODO: Try using a single file and opening multiple handles to that file as buckets...
    for( uint i = 0; i < bucketCount; i++ )
    {
        IStream* file = nullptr;
        const bool isCachable = IsFlagSet( options, FileSetOptions::Cachable );
        
        if( isCachable )
            file = new HybridStream();
        else 
            file = new FileStream();

        fileSet.files[i] = file;

        const FileMode fileMode =
        #if _DEBUG && ( BB_DP_DBG_READ_EXISTING_F1 || BB_DP_DBG_SKIP_PHASE_1 )
            fileId != FileId::PLOT ? FileMode::OpenOrCreate : FileMode::Create;
        #else
            FileMode::Create;
        #endif

        if( fileId != FileId::PLOT )
            sprintf( baseName, "%s_%u.tmp", name, i );
        else
            sprintf( baseName, "%s", name );

        bool opened;

        if( isCachable )
        {
            ASSERT( optsData );
            ASSERT( optsData->cache );

            opened = static_cast<HybridStream*>( file )->Open( optsData->cache, optsData->cacheSize, pathBuffer, fileMode, FileAccess::ReadWrite, flags );
        }
        else
            opened = static_cast<FileStream*>( file )->Open( pathBuffer, fileMode, FileAccess::ReadWrite, flags );

        if( !opened )
        {
            // Allow plot file to fail opening
            if( fileId == FileId::PLOT )
            {
                Log::Line( "Failed to open plot file %s with error: %d.", pathBuffer, file->GetError() );
                return false;
            }
            
            Fatal( "Failed to open temp work file @ %s with error: %d.", pathBuffer, file->GetError() );
        }

        if( i == 0 && IsFlagSet( options, FileSetOptions::DirectIO ) )
        {
            const size_t totalBlockSize = file->BlockSize() * bucketCount;
            fileSet.blockBuffers = bbvirtalloc<void>( totalBlockSize );
        }
    }

    return true;
}

//-----------------------------------------------------------
void DiskBufferQueue::SetTransform( FileId fileId, IIOTransform& transform )
{
    _files[(int)fileId].transform = &transform;
}

//-----------------------------------------------------------
void DiskBufferQueue::OpenPlotFile( const char* fileName, const byte* plotId, const byte* plotMemo, uint16 plotMemoSize )
{
    ASSERT( fileName     );
    ASSERT( plotId       );
    ASSERT( plotMemo     );
    ASSERT( plotMemoSize );

    // #TODO: Retry multiple-times.
    const bool didOpen = InitFileSet( FileId::PLOT, fileName, 1 );
    FatalIf( !didOpen, "Failed to open plot file." );

    // Write plot header
    const size_t headerSize =
        ( sizeof( kPOSMagic ) - 1 ) +
        32 +            // plot id
        1  +            // k
        2  +            // kFormatDescription length
        ( sizeof( kFormatDescription ) - 1 ) +
        2  +            // Memo length
        plotMemoSize +  // Memo
        80              // Table pointers
    ;

    _plotHeaderSize = headerSize;

    // #TODO: Support block-aligned like in PlotWriter.cpp
    if( !_plotHeaderbuffer )
        _plotHeaderbuffer = bbvirtalloc<byte>( headerSize );

    byte* header = _plotHeaderbuffer;

    // Encode the headers
    {
        // Magic
        byte* headerWriter = header;
        memcpy( headerWriter, kPOSMagic, sizeof( kPOSMagic ) - 1 );
        headerWriter += sizeof( kPOSMagic ) - 1;

        // Plot Id
        memcpy( headerWriter, plotId, 32 );
        headerWriter += 32;

        // K
        *headerWriter++ = (byte)_K;

        // Format description
        *((uint16*)headerWriter) = Swap16( (uint16)(sizeof( kFormatDescription ) - 1) );
        headerWriter += 2;
        memcpy( headerWriter, kFormatDescription, sizeof( kFormatDescription ) - 1 );
        headerWriter += sizeof( kFormatDescription ) - 1;

        // Memo
        *((uint16*)headerWriter) = Swap16( plotMemoSize );
        headerWriter += 2;
        memcpy( headerWriter, plotMemo, plotMemoSize );
        headerWriter += plotMemoSize;

        // Tables will be copied at the end.
        _plotTablesPointers = (uint64)(headerWriter - header);

        // Write the headers to disk
        WriteFile( FileId::PLOT, 0, header, (size_t)headerSize );
        CommitCommands();
    }
}

//-----------------------------------------------------------
void DiskBufferQueue::WriteBuckets( FileId id, const void* buckets, const uint* sizes )
{
    Command* cmd = GetCommandObject( Command::WriteBuckets );
    cmd->buckets.buffers = (byte*)buckets;
    cmd->buckets.sizes   = sizes;
    cmd->buckets.fileId  = id;
}

//-----------------------------------------------------------
void DiskBufferQueue::WriteFile( FileId id, uint bucket, const void* buffer, size_t size )
{
    Command* cmd = GetCommandObject( Command::WriteFile );
    cmd->file.buffer = (byte*)buffer;
    cmd->file.size   = size;
    cmd->file.fileId = id;
    cmd->file.bucket = bucket;
}

//-----------------------------------------------------------
void DiskBufferQueue::ReadFile( FileId id, uint bucket, void* dstBuffer, size_t readSize )
{
    Command* cmd = GetCommandObject( Command::ReadFile );
    cmd->file.buffer = (byte*)dstBuffer;
    cmd->file.size   = readSize;
    cmd->file.fileId = id;
    cmd->file.bucket = bucket;
}

//-----------------------------------------------------------
void DiskBufferQueue::SeekFile( FileId id, uint bucket, int64 offset, SeekOrigin origin )
{
    Command* cmd = GetCommandObject( Command::SeekFile );
    cmd->seek.fileId = id;
    cmd->seek.bucket = bucket;
    cmd->seek.offset = offset;
    cmd->seek.origin = origin;
}

//-----------------------------------------------------------
void DiskBufferQueue::SeekBucket( FileId id, int64 offset, SeekOrigin origin )
{
    Command* cmd = GetCommandObject( Command::SeekBucket );
    cmd->seek.fileId = id;
    cmd->seek.offset = offset;
    cmd->seek.origin = origin;
}

//-----------------------------------------------------------
void DiskBufferQueue::ReleaseBuffer( void* buffer )
{
    ASSERT( buffer );

    Command* cmd = GetCommandObject( Command::ReleaseBuffer );
    cmd->releaseBuffer.buffer = (byte*)buffer;
}

//-----------------------------------------------------------
void DiskBufferQueue::SignalFence( Fence& fence )
{
    Command* cmd = GetCommandObject( Command::SignalFence );
    cmd->fence.signal = &fence;
    cmd->fence.value  = -1;
}

//-----------------------------------------------------------
void DiskBufferQueue::SignalFence( Fence& fence, uint32 value )
{
    Command* cmd = GetCommandObject( Command::SignalFence );
    cmd->fence.signal = &fence;
    cmd->fence.value  = (int64)value;
}

//-----------------------------------------------------------
void DiskBufferQueue::WaitForFence( Fence& fence )
{
    Command* cmd = GetCommandObject( Command::WaitForFence );
    cmd->fence.signal = &fence;
    cmd->fence.value  = -1;
}

//-----------------------------------------------------------
void DiskBufferQueue::DeleteFile( FileId id, uint bucket )
{
    // Log::Line( "DeleteFile( %u : %u )", id, bucket );
    Command* cmd = GetCommandObject( Command::DeleteFile );
    cmd->deleteFile.fileId = id;
    cmd->deleteFile.bucket = bucket;
}

//-----------------------------------------------------------
void DiskBufferQueue::DeleteBucket( FileId id )
{
    // Log::Line( "DeleteBucket( %u )", id );
    // #TODO: This command must be run in another helper thread in order
    //        to not block while the kernel buffers are being
    //        cleared (when not using direct IO). Otherwise
    //        other commands will get blocked while a command
    //        we don't care about is executing.
    Command* cmd = GetCommandObject( Command::DeleteBucket );
    cmd->deleteFile.fileId = id;
}

//-----------------------------------------------------------
void DiskBufferQueue::CompletePendingReleases()
{
    _workHeap.CompletePendingReleases();
}

//-----------------------------------------------------------
inline DiskBufferQueue::Command* DiskBufferQueue::GetCommandObject( Command::CommandType type )
{
    Command* cmd;
    while( !_commands.Write( cmd ) )
    {
        Log::Line( "[DiskBufferQueue] Command buffer full. Waiting for commands." );
        auto waitTimer = TimerBegin();

        // Block and wait until we have commands free in the buffer
        _cmdConsumedSignal.Wait();
        
        Log::Line( "[DiskBufferQueue] Waited %.6lf seconds for a Command to be available.", TimerEnd( waitTimer ) );
    }

    ZeroMem( cmd );
    cmd->type = type;

    #if DBG_LOG_ENABLE
        Log::Debug( "[DiskBufferQueue] > Snd: %s (%d)", DbgGetCommandName( type ), type );
    #endif

    return cmd;
}

//-----------------------------------------------------------
void DiskBufferQueue::CommitCommands()
{
    //Log::Debug( "Committing %d commands.", _commands._pendingCount );
    _commands.Commit();
    _cmdReadySignal.Signal();
}

//-----------------------------------------------------------
void DiskBufferQueue::CommandThreadMain( DiskBufferQueue* self )
{
    // #TODO: Remove this, testing
    SysHost::SetCurrentThreadAffinityCpuId( SysHost::GetLogicalCPUCount() - 1 );
    self->CommandMain();
}

//-----------------------------------------------------------
byte* DiskBufferQueue::GetBufferForId( const FileId fileId, const uint32 bucket, const size_t size, bool blockUntilFreeBuffer )
{
    if( !IsFlagSet( _files[(int)fileId].options, FileSetOptions::DirectIO ) )
        return GetBuffer( size, blockUntilFreeBuffer );

    const size_t blockOffset = _files[(int)fileId].blockOffsets[bucket];
    const size_t blockSize   = _files[(int)fileId].files[bucket]->BlockSize();

    size_t allocSize = RoundUpToNextBoundaryT( size, blockOffset );
    if( blockOffset > 0 )
        allocSize += blockSize;
    
    byte* buffer = _workHeap.Alloc( allocSize, blockSize, blockUntilFreeBuffer, &_ioBufferWaitTime );
    buffer += blockOffset;
    
    return buffer;
}


///
/// Command Thread
///

//-----------------------------------------------------------
void DiskBufferQueue::CommandMain()
{
    const int CMD_BUF_SIZE = 64;
    Command commands[CMD_BUF_SIZE];

    for( ;; )
    {
        _cmdReadySignal.Wait();

        int cmdCount;
        while( ( ( cmdCount = _commands.Dequeue( commands, CMD_BUF_SIZE ) ) ) )
        {
            _cmdConsumedSignal.Signal();

            for( int i = 0; i < cmdCount; i++ )
                ExecuteCommand( commands[i] );
        }
    }
}

//-----------------------------------------------------------
void DiskBufferQueue::ExecuteCommand( Command& cmd )
{
    //#if DBG_LOG_ENABLE
    //    Log::Debug( "[DiskBufferQueue] ^ Cmd Execute: %s (%d)", DbgGetCommandName( cmd.type ), cmd.type );
    //#endif

    switch( cmd.type )
    {
        case Command::WriteBuckets:
            #if DBG_LOG_ENABLE
                Log::Debug( "[DiskBufferQueue] ^ Cmd WriteBuckets: (%u) addr:0x%p", cmd.buckets.fileId, cmd.buckets.buffers );
            #endif
            CmdWriteBuckets( cmd );
        break;

        case Command::WriteFile:
            #if DBG_LOG_ENABLE
                Log::Debug( "[DiskBufferQueue] ^ Cmd WriteFile: (%u) bucket:%u sz:%llu addr:0x%p", cmd.file.fileId, cmd.file.bucket, cmd.file.size, cmd.file.buffer );
            #endif
            CndWriteFile( cmd );
        break;

        case Command::ReadFile:
            #if DBG_LOG_ENABLE
                Log::Debug( "[DiskBufferQueue] ^ Cmd ReadFile: (%u) bucket:%u sz:%llu addr:0x%p", cmd.file.fileId, cmd.file.bucket, cmd.file.size, cmd.file.buffer );
            #endif
            CmdReadFile( cmd );
        break;

        case Command::SeekFile:
            #if DBG_LOG_ENABLE
                Log::Debug( "[DiskBufferQueue] ^ Cmd SeekFile: (%u) bucket:%u offset:%lld origin:%ld", cmd.seek.fileId, cmd.seek.bucket, cmd.seek.offset, (int)cmd.seek.origin );
            #endif
                if( !_files[(uint)cmd.seek.fileId].files[cmd.seek.bucket]->Seek( cmd.seek.offset, cmd.seek.origin ) )
                {
                    int err = _files[(uint)cmd.seek.fileId].files[cmd.seek.bucket]->GetError();
                    Fatal( "[DiskBufferQueue] Failed to seek file %s.%u with error %d (0x%x)", 
                           _files[(uint)cmd.seek.fileId].name, cmd.seek.bucket, err, err );
                }
        break;

        case Command::SeekBucket:
            #if DBG_LOG_ENABLE
                Log::Debug( "[DiskBufferQueue] ^ Cmd SeekBucket: (%u) offset:%lld origin:%ld", cmd.seek.fileId, cmd.seek.offset, (int)cmd.seek.origin );
            #endif

            CmdSeekBucket( cmd );
        break;

        case Command::ReleaseBuffer:
            #if DBG_LOG_ENABLE
                Log::Debug( "[DiskBufferQueue] ^ Cmd ReleaseBuffer: 0x%p", cmd.releaseBuffer.buffer );
            #endif
            _workHeap.Release( cmd.releaseBuffer.buffer );
        break;

        case Command::SignalFence:
            #if DBG_LOG_ENABLE
                Log::Debug( "[DiskBufferQueue] ^ Cmd MemoryFence" );
            #endif
            ASSERT( cmd.fence.signal );
            if( cmd.fence.value < 0 )
                cmd.fence.signal->Signal();
            else
                cmd.fence.signal->Signal( (uint32)cmd.fence.value );
        break;

        case Command::WaitForFence:
            #if DBG_LOG_ENABLE
                Log::Debug( "[DiskBufferQueue] ^ Cmd WaitForFence" );
            #endif
                ASSERT( cmd.fence.signal );
            cmd.fence.signal->Wait();
        break;

        case Command::DeleteFile:
            #if DBG_LOG_ENABLE
                Log::Debug( "[DiskBufferQueue] ^ Cmd DeleteFile" );
            #endif
             CmdDeleteFile( cmd );  // Dispatch to deleter thread
        break;

        case Command::DeleteBucket:
            #if DBG_LOG_ENABLE
                Log::Debug( "[DiskBufferQueue] ^ Cmd DeleteBucket" );
            #endif
             CmdDeleteBucket( cmd ); // Dispatch to deleter thread
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

    FileSet&     fileSet     = _files[(int)fileId];
    // ASSERT( IsFlagSet( fileBuckets.files[0]->GetFileAccess(), FileAccess::ReadWrite ) );

    const uint32 bucketCount = (uint32)fileSet.files.length;

    Log::Debug( "  >>> Write 0x%p", buffers );

    // Single-threaded for now... We don't have file handles for all the threads yet!
    const size_t blockSize = _blockSize;
    const byte*  buffer    = buffers;

    // if( fileSet.transform )
    // {
    //     IIOTransform::TransformData data = {
    //         .buffer      = (void*)buffers,
    //         .numBuckets  = bucketCount,
    //         .bucketSizes = (uint32*)sizes
    //     };

    //     // #TODO: Profile time elapsed here
    //     fileSet.transform->Write( data );
    // }

    
    for( uint i = 0; i < bucketCount; i++ )
    {
        const size_t bufferSize = sizes[i];
        
        // Only write up-to the block-aligned boundary.
        // The caller is in charge of writing any remainders manually
        // #TODO: Remove the direct IO size adjust
        const size_t writeSize = _useDirectIO == false ? bufferSize :
                                 bufferSize / blockSize * blockSize;

        WriteToFile( *fileSet.files[i], writeSize, buffer, (byte*)fileSet.blockBuffers, fileSet.name, i );
        // ASSERT( IsFlagSet( fileBuckets.files[i].GetFileAccess(), FileAccess::ReadWrite ) );
        // Each bucket buffer must start at the next block-aligned boundary
        const size_t bufferOffset = _useDirectIO == false ? bufferSize :
                                    RoundUpToNextBoundaryT( bufferSize, blockSize );

        buffer += bufferOffset;
    }
}

//-----------------------------------------------------------
void DiskBufferQueue::CndWriteFile( const Command& cmd )
{
    FileSet& fileBuckets = _files[(int)cmd.file.fileId];
    WriteToFile( *fileBuckets.files[cmd.file.bucket], cmd.file.size, cmd.file.buffer, (byte*)fileBuckets.blockBuffers, fileBuckets.name, cmd.file.bucket );
}

//-----------------------------------------------------------
void DiskBufferQueue::CmdReadFile( const Command& cmd )
{
    FileSet& fileBuckets = _files[(int)cmd.file.fileId];
    ReadFromFile( *fileBuckets.files[cmd.file.bucket], cmd.file.size, cmd.file.buffer, (byte*)fileBuckets.blockBuffers, fileBuckets.name, cmd.file.bucket );
}

//-----------------------------------------------------------
void DiskBufferQueue::CmdSeekBucket( const Command& cmd )
{
    FileSet&     fileBuckets = _files[(int)cmd.seek.fileId];
    const uint   bucketCount = (uint)fileBuckets.files.length;

    const int64      seekOffset = cmd.seek.offset;
    const SeekOrigin seekOrigin = cmd.seek.origin;

    for( uint i = 0; i < bucketCount; i++ )
    {
        if( !fileBuckets.files[i]->Seek( seekOffset, seekOrigin ) )
        {
            int err = fileBuckets.files[i]->GetError();
            Fatal( "[DiskBufferQueue] Failed to seek file %s.%u with error %d (0x%x)", fileBuckets.name, i, err, err );
        }
    }
}

//-----------------------------------------------------------
inline void DiskBufferQueue::WriteToFile( IStream& file, size_t size, const byte* buffer, byte* blockBuffer, const char* fileName, uint bucket )
{
    if( !_useDirectIO )
    {
        while( size )
        {
            ssize_t sizeWritten = file.Write( buffer, size );

            if( sizeWritten < 1 )
            {
                const int err = file.GetError();
                Fatal( "Failed to write to '%s.%u' work file with error %d (0x%x).", fileName, bucket, err, err );
            }

            size -= (size_t)sizeWritten;
            buffer += sizeWritten;
        }
    }
    else
    {
        const size_t blockSize   = _blockSize;
        size_t       sizeToWrite = size / blockSize * blockSize;
        const size_t remainder   = size - sizeToWrite;

        while( sizeToWrite )
        {
            ssize_t sizeWritten = file.Write( buffer, sizeToWrite );
            if( sizeWritten < 1 )
            {
                const int err = file.GetError();
                Fatal( "Failed to write to '%s.%u' work file with error %d (0x%x).", fileName, bucket, err, err );
            }

            ASSERT( sizeWritten <= (ssize_t)sizeToWrite );

            sizeToWrite -= (size_t)sizeWritten;
            buffer      += sizeWritten;
        }
        
        if( remainder )
        {
            ASSERT( blockBuffer );
            
            // Unnecessary zeroing of memory, but might be useful for debugging
            memset( blockBuffer, 0, blockSize );
            memcpy( blockBuffer, buffer, remainder );

            ssize_t sizeWritten = file.Write( blockBuffer, blockSize );

            if( sizeWritten < remainder )
            {
                const int err = file.GetError();
                Fatal( "Failed to write block to '%s.%u' work file with error %d (0x%x).", fileName, bucket, err, err );
            }
        }
    }
}

//-----------------------------------------------------------
void DiskBufferQueue::ReadFromFile( IStream& file, size_t size, byte* buffer, byte* blockBuffer, const char* fileName, uint bucket )
{
    if( !_useDirectIO )
    {
        while( size )
        {
            const ssize_t sizeRead = file.Read( buffer, size );
            if( sizeRead < 1 )
            {
                const int err = file.GetError();
                Fatal( "Failed to read from '%s_%u' work file with error %d (0x%x).", fileName, bucket, err, err );
            }

            size  -= (size_t)sizeRead;
            buffer += sizeRead;
        }
    }
    else
    {
        const size_t blockSize  = _blockSize;
        
        /// #NOTE: All our buffers should be block aligned so we should be able to freely read all blocks to them...
        ///       Only implement remainder block reading if we really need to.
    //     size_t       sizeToRead = size / blockSize * blockSize;
    //     const size_t remainder  = size - sizeToRead;

        size_t sizeToRead = CDivT( size, blockSize ) * blockSize;

        while( sizeToRead )
        {
            ssize_t sizeRead = file.Read( buffer, sizeToRead );
            if( sizeRead < 1 )
            {
                const int err = file.GetError();
                Fatal( "Failed to read from '%s_%u' work file with error %d (0x%x).", fileName, bucket, err, err );
            }

            ASSERT( sizeRead <= (ssize_t)sizeToRead );
            
            sizeToRead -= (size_t)sizeRead;
            buffer     += sizeRead;
        }
    }

//     if( remainder )
//     {
//         ASSERT( blockBuffer );
// 
//         file.read
//     }
}

//----------------------------------------------------------
void DiskBufferQueue::CmdDeleteFile( const Command& cmd )
{
    FileDeleteCommand delCmd;
    delCmd.fileId = cmd.deleteFile.fileId;
    delCmd.bucket = (int64)cmd.deleteFile.bucket;
    while( !_deleteQueue.Enqueue( delCmd ) );

    _deleteSignal.Signal();
}

//----------------------------------------------------------
void DiskBufferQueue::CmdDeleteBucket( const Command& cmd )
{
    FileDeleteCommand delCmd;
    delCmd.fileId = cmd.deleteFile.fileId;
    delCmd.bucket = -1;
    while( !_deleteQueue.Enqueue( delCmd ) );
    
    _deleteSignal.Signal();
}

//-----------------------------------------------------------
inline const char* DiskBufferQueue::DbgGetCommandName( Command::CommandType type )
{
    switch( type )
    {
        case DiskBufferQueue::Command::WriteFile:
            return "WriteFile";

        case DiskBufferQueue::Command::WriteBuckets:
            return "WriteBuckets";

        case DiskBufferQueue::Command::ReadFile:
            return "ReadFile";

        case DiskBufferQueue::Command::ReleaseBuffer:
            return "ReleaseBuffer";

        case DiskBufferQueue::Command::SeekFile:
            return "SeekFile";

        case DiskBufferQueue::Command::SeekBucket:
            return "SeekBucket";

        case DiskBufferQueue::Command::SignalFence:
            return "SignalFence";

        case DiskBufferQueue::Command::WaitForFence:
            return "WaitForFence";

        case DiskBufferQueue::Command::DeleteFile:
            return "DeleteFile";

        case DiskBufferQueue::Command::DeleteBucket:
            return "DeleteBucket";

        default:
            ASSERT( 0 );
            return nullptr;
    }
}


///
/// File-Deleter Thread
///
//-----------------------------------------------------------
void DiskBufferQueue::DeleterThreadMain( DiskBufferQueue* self )
{
    self->DeleterMain();
}

//-----------------------------------------------------------
void DiskBufferQueue::DeleterMain()
{
    const int BUFFER_SIZE = 1024;
    FileDeleteCommand commands[BUFFER_SIZE];

    for( ;; )
    {
        _deleteSignal.Wait();

        // Keep grabbing commands until there's none more
        for( ;; )
        {
            const int count = _deleteQueue.Dequeue( commands, BUFFER_SIZE );

            if( count == 0 )
            {
                if( _deleterExit )
                    return;

                break;
            }

            for( int i = 0; i < count; i++ )
            {
                auto& cmd = commands[i];

                if( cmd.bucket < 0 )
                    DeleteBucketNow( cmd.fileId );
                else
                    DeleteFileNow( cmd.fileId, (uint32)cmd.bucket );
            }
        }
    }
}

//-----------------------------------------------------------
void DiskBufferQueue::DeleteFileNow( const FileId fileId, uint32 bucket )
{
    FileSet&    fileSet = _files[(int)fileId];

    const bool isHybridFile = IsFlagSet( fileSet.options, FileSetOptions::DirectIO );
    if( isHybridFile )
    {
        auto* file = static_cast<HybridStream*>( fileSet.files[0] );
        file->Close();
    }
    else
    {
        auto* file = static_cast<FileStream*>( fileSet.files[0] );
        file->Close();
    }

    char* basePath = _filePathBuffer + _workDir.length();
    sprintf( basePath, "%s_%u.tmp", fileSet.name, bucket );
    
    const int r = remove( _filePathBuffer );

    if( r )
        Log::Error( "Error: Failed to delete file %s with errror %d (0x%x).", _filePathBuffer, r, r );
}

//-----------------------------------------------------------
void DiskBufferQueue::DeleteBucketNow( const FileId fileId )
{
    FileSet& fileSet = _files[(int)fileId];

    char* basePath = _filePathBuffer + _workDir.length();

    const bool isHybridFile = IsFlagSet( fileSet.options, FileSetOptions::DirectIO );

    for( size_t i = 0; i < fileSet.files.length; i++ )
    {
        if( isHybridFile )
        {
            auto* file = static_cast<HybridStream*>( fileSet.files[i] );
            file->Close();
        }
        else
        {
            auto* file = static_cast<FileStream*>( fileSet.files[i] );
            file->Close();
        }

        sprintf( basePath, "%s_%u.tmp", fileSet.name, (uint)i );
    
        const int r = remove( _filePathBuffer );

        if( r )
            Log::Error( "Error: Failed to delete file %s with errror %d (0x%x).", _filePathBuffer, r, r );
    }
}

