#include "DiskBufferQueue.h"
#include "io/FileStream.h"
#include "io/HybridStream.h"
#include "plotdisk/DiskPlotConfig.h"
#include "jobs/IOJob.h"
#include "util/Util.h"
#include "util/Log.h"


#define NULL_BUFFER -1

#ifdef _WIN32
    #define PATH_SEPA_STR "\\"
    #define CheckPathSeparator( x ) ((x) == '\\' || (x) == '/')
#else
    #define PATH_SEPA_STR "/"
    #define CheckPathSeparator( x ) ((x) == '/')
#endif

//-----------------------------------------------------------
DiskBufferQueue::DiskBufferQueue( 
    const char* workDir1, const char* workDir2, const char* plotDir, byte* workBuffer, 
    size_t workBufferSize, uint ioThreadCount,
    int32 threadBindId
)
    : _workDir1      ( workDir1 )
    , _workDir2      ( workDir2 )
    , _plotDir       ( plotDir  )
    , _workHeap      ( workBufferSize, workBuffer )
    // , _threadPool    ( ioThreadCount, ThreadPool::Mode::Fixed, true )
    , _dispatchThread()
    , _deleterThread ()
    , _deleteSignal  ()
    , _deleteQueue   ( 128 )
    , _threadBindId  ( threadBindId )
{
    ASSERT( workDir1 );
    ASSERT( plotDir  );
    
    if( !workDir2 )
        workDir2 = workDir1;

    // Initialize path buffers
    FatalIf( _workDir1.length() < 1, "Working directory path 1 is empty." );
    FatalIf( _workDir2.length() < 1, "Working directory path 2 is empty." );
    FatalIf( _plotDir.length()  < 1, "Plot tmp directory is empty." );

    // Add a trailing slash if we don't have one
    if( !CheckPathSeparator( _workDir1.back() ) )
        _workDir1 += PATH_SEPA_STR;
    if( !CheckPathSeparator( _workDir2.back() ) )
        _workDir2 += PATH_SEPA_STR;
    if( !CheckPathSeparator( _plotDir.back() ) )
        _plotDir += PATH_SEPA_STR;

    const size_t workDirLen = std::max( _workDir1.length(), _workDir2.length() );

    const size_t PLOT_FILE_LEN = sizeof( "/plot-k32-2021-08-05-18-55-77a011fc20f0003c3adcc739b615041ae56351a22b690fd854ccb6726e5f43b7.plot.tmp" );

    _filePathBuffer    = bbmalloc<char>( workDirLen + PLOT_FILE_LEN );  // Should be enough for all our file names
    _delFilePathBuffer = bbmalloc<char>( workDirLen + PLOT_FILE_LEN );

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

    free( _filePathBuffer    );
    free( _delFilePathBuffer );
}

//-----------------------------------------------------------
size_t DiskBufferQueue::BlockSize( FileId fileId ) const
{
    ASSERT( _files[(int)fileId].files[0] );
    return _files[(int)fileId].files[0]->BlockSize();
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
    const bool isPlotFile = fileId == FileId::PLOT;
    const bool useTmp2    = IsFlagSet( options, FileSetOptions::UseTemp2 );

    const std::string& wokrDir = isPlotFile ? _plotDir :
                                    useTmp2 ? _workDir2 : _workDir1;
    memcpy( _filePathBuffer, wokrDir.c_str(), wokrDir.length() );

    const char* pathBuffer = _filePathBuffer;
          char* baseName   = _filePathBuffer + wokrDir.length();

    FileFlags flags = FileFlags::LargeFile;
    if( IsFlagSet( options, FileSetOptions::DirectIO ) )
        flags |= FileFlags::NoBuffering;

    FileSet& fileSet = _files[(uint)fileId];

    if( !fileSet.name )
    {
        ASSERT( !fileSet.files.values );

        fileSet.name         = name;
        fileSet.files.values = new IStream*[bucketCount];
        fileSet.files.length = bucketCount;
        fileSet.blockBuffer  = nullptr;
        fileSet.options      = options;

        memset( fileSet.files.values, 0, sizeof( uintptr_t ) * bucketCount );

        if( IsFlagSet( options, FileSetOptions::Interleaved ) || IsFlagSet( options, FileSetOptions::Alternating ) )
        {
            fileSet.readSliceSizes.SetTo( new Span<size_t>[bucketCount], bucketCount );
            fileSet.writeSliceSizes.SetTo( new Span<size_t>[bucketCount], bucketCount );
            for( uint32 i = 0; i < bucketCount; i++ )
            {
                fileSet.readSliceSizes[i].SetTo( new size_t[bucketCount]{}, bucketCount );
                fileSet.writeSliceSizes[i].SetTo( new size_t[bucketCount]{}, bucketCount );
            }
        }

        if( IsFlagSet( options, FileSetOptions::Alternating ) )
        {
            fileSet.maxSliceSize = optsData->maxSliceSize;
            ASSERT( fileSet.maxSliceSize );
        }
    }

    const bool isCachable = IsFlagSet( options, FileSetOptions::Cachable ) && optsData->cacheSize > 0;
    ASSERT( !isCachable || optsData );

    const size_t cacheSize = isCachable ? optsData->cacheSize / bucketCount : 0;
    byte* cache = isCachable ? (byte*)optsData->cache : nullptr;

    // Ensure we don't flag it as a HybridStream if the cache happened to be 0
    if( !isCachable )
        UnSetFlag( fileSet.options, FileSetOptions::Cachable );

    // #TODO: Try using a single file and opening multiple handles to that file as buckets...
    for( uint i = 0; i < bucketCount; i++ )
    {
        IStream* file = fileSet.files[i];

        if( !file )
        {
            if( isCachable )
                file = new HybridStream();
            else 
                file = new FileStream();

            fileSet.files[i] = file;
        }
        

        const FileMode fileMode =
        #if _DEBUG && ( BB_DP_DBG_READ_EXISTING_F1 || BB_DP_DBG_SKIP_PHASE_1 || BB_DP_P1_SKIP_TO_TABLE || BB_DP_DBG_SKIP_TO_C_TABLES )
            !isPlotFile ? FileMode::OpenOrCreate : FileMode::Create;
        #else
            FileMode::Create;
        #endif

        if( !isPlotFile )
            sprintf( baseName, "%s_%u.tmp", name, i );
        else
        {
            sprintf( baseName, "%s", name );

            _plotFullName = pathBuffer;
            _plotFullName.erase( _plotFullName.length() - 4 );
        }

        bool opened;

        if( isCachable )
        {
            opened = static_cast<HybridStream*>( file )->Open( cache, cacheSize, pathBuffer, fileMode, FileAccess::ReadWrite, flags );
            cache += cacheSize;

            ASSERT( cacheSize / file->BlockSize() * file->BlockSize() == cacheSize );
        }
        else
            opened = static_cast<FileStream*>( file )->Open( pathBuffer, fileMode, FileAccess::ReadWrite, flags );

        if( !opened )
        {
            // Allow plot file to fail opening
            if( isPlotFile )
            {
                Log::Line( "Failed to open plot file %s with error: %d.", pathBuffer, file->GetError() );
                return false;
            }
            
            Fatal( "Failed to open temp work file @ %s with error: %d.", pathBuffer, file->GetError() );
        }
        
        // Always align for now.
        if( i == 0 && !fileSet.blockBuffer )//&& IsFlagSet( options, FileSetOptions::DirectIO ) )
        {
            // const size_t totalBlockSize = file->BlockSize() * bucketCount;
            ASSERT( file->BlockSize() );
            fileSet.blockBuffer = bbvirtalloc<void>( file->BlockSize() );   // #TODO: This should be removed, and we should use
                                                                            //        a shared one per temp dir.
        }
    }

    return true;
}

//-----------------------------------------------------------
void DiskBufferQueue::SetTransform( FileId fileId, IIOTransform& transform )
{
    // _files[(int)fileId].transform = &transform;
}

//-----------------------------------------------------------
void DiskBufferQueue::OpenPlotFile( const char* fileName, const byte* plotId, const byte* plotMemo, uint16 plotMemoSize )
{
    // #TODO: fileName should not have .tmp here. 
    //        Change that and then change InitFile to add the .tmp like the rest of the files.
    ASSERT( fileName     );
    ASSERT( plotId       );
    ASSERT( plotMemo     );
    ASSERT( plotMemoSize );
    ASSERT( _plotFullName.length() == 0 );

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

    // Encode the header
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
void DiskBufferQueue::FinishPlot( Fence& fence )
{
    SignalFence( fence );
    CommitCommands();

    char* plotPath = _filePathBuffer;
    char* baseName = plotPath + _plotDir.length();

    const char* plotTmpName = _files[(int)FileId::PLOT].name;

    memcpy( plotPath, _plotDir.c_str(), _plotDir.length() );
    memcpy( baseName, plotTmpName, strlen( plotTmpName ) + 1 );

    fence.Wait();
    CloseFileNow( FileId::PLOT, 0 );

    const uint32 RETRY_COUNT  = 10;
    const long   MS_WAIT_TIME = 1000;

    Log::Line( "Renaming plot to '%s'", _plotFullName.c_str() );

    int32 error = 0;

    for( uint32 i = 0; i < RETRY_COUNT; i++ )
    {
        const bool success = FileStream::Move( plotPath, _plotFullName.c_str(), &error );

        if( success )
            break;
        
        Log::Line( "Error: Could not rename plot file with error: %d.", error );

        if( i+1 == RETRY_COUNT)
        {
            Log::Line( "Error:: Failed to to rename plot file after %u retries. Please rename manually.", RETRY_COUNT );
            break;
        }

        Log::Line( "Retrying in %.2lf seconds...", MS_WAIT_TIME / 1000.0 );
        Thread::Sleep( MS_WAIT_TIME );
    }

    _plotFullName = "";
}

//-----------------------------------------------------------
void DiskBufferQueue::WriteBuckets( FileId id, const void* buckets, const uint* writeSizes, const uint32* sliceSizes )
{
    Command* cmd = GetCommandObject( Command::WriteBuckets );
    cmd->buckets.writeSizes  = writeSizes;
    cmd->buckets.sliceSizes  = sliceSizes != nullptr ? sliceSizes : writeSizes;
    cmd->buckets.buffers     = (byte*)buckets;
    cmd->buckets.elementSize = 1;
    cmd->buckets.fileId      = id;
}

//-----------------------------------------------------------
void DiskBufferQueue::WriteBucketElements( const FileId id, const bool interleaved, const void* buckets, const size_t elementSize, const uint* writeCounts, const uint32* sliceCounts )
{
    ASSERT( buckets );
    ASSERT( elementSize );
    ASSERT( writeCounts );
    ASSERT( elementSize < std::numeric_limits<uint32>::max() );

    Command* cmd = GetCommandObject( Command::WriteBucketElements );
    cmd->buckets.writeSizes  = writeCounts;
    cmd->buckets.sliceSizes  = sliceCounts != nullptr ? sliceCounts : writeCounts;
    cmd->buckets.buffers     = (byte*)buckets;
    cmd->buckets.elementSize = (uint32)elementSize;
    cmd->buckets.fileId      = id;
    cmd->buckets.interleaved = interleaved;
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
void DiskBufferQueue::ReadBucketElements( const FileId id, const bool interleaved, Span<byte>& buffer, const size_t elementSize )
{
    Command* cmd = GetCommandObject( Command::ReadBucket );
    cmd->readBucket.buffer      = &buffer;
    cmd->readBucket.elementSize = (uint32)elementSize;
    cmd->readBucket.fileId      = id;
    cmd->readBucket.interleaved = interleaved;
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
    // #TODO: See DeleteBucket
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
void DiskBufferQueue::TruncateBucket( FileId id, const ssize_t position )
{
    Command* cmd = GetCommandObject( Command::TruncateBucket );
    cmd->truncateBucket.fileId   = id;
    cmd->truncateBucket.position = position;
}

//-----------------------------------------------------------
void DiskBufferQueue::CompletePendingReleases()
{
    _workHeap.CompletePendingReleases();
}

#if _DEBUG
//-----------------------------------------------------------
void DiskBufferQueue::DebugWriteSliceSizes( const TableId table, const FileId fileId )
{
    Command* cmd = GetCommandObject( Command::DBG_WriteSliceSizes );
    cmd->dbgSliceSizes.table  = table;
    cmd->dbgSliceSizes.fileId = fileId;
}

//-----------------------------------------------------------
void DiskBufferQueue::DebugReadSliceSizes( const TableId table, const FileId fileId )
{
    Command* cmd = GetCommandObject( Command::DBG_ReadSliceSizes );
    cmd->dbgSliceSizes.table  = table;
    cmd->dbgSliceSizes.fileId = fileId;
}
#endif

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
    if( self->_threadBindId > -1 )
    {
        const uint32 threadBindId = (uint32)self->_threadBindId;
        ASSERT( threadBindId < SysHost::GetLogicalCPUCount() );
        SysHost::SetCurrentThreadAffinityCpuId( threadBindId );
    }
    self->CommandMain();
}

//-----------------------------------------------------------
// byte* DiskBufferQueue::GetBufferForId( const FileId fileId, const uint32 bucket, const size_t size, bool blockUntilFreeBuffer )
// {
//     if( !IsFlagSet( _files[(int)fileId].options, FileSetOptions::DirectIO ) )
//         return GetBuffer( size, blockUntilFreeBuffer );

//     const size_t blockOffset = _files[(int)fileId].blockOffsets[bucket];
//     const size_t blockSize   = _files[(int)fileId].files[bucket]->BlockSize();

//     size_t allocSize = RoundUpToNextBoundaryT( size, blockOffset );
//     if( blockOffset > 0 )
//         allocSize += blockSize;
    
//     byte* buffer = _workHeap.Alloc( allocSize, blockSize, blockUntilFreeBuffer, &_ioBufferWaitTime );
//     buffer += blockOffset;
    
//     return buffer;
// }


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
            CmdWriteBuckets( cmd, 1 );
        break;

        case Command::WriteBucketElements:
            #if DBG_LOG_ENABLE
                Log::Debug( "[DiskBufferQueue] ^ Cmd WriteBucketElements: (%u) addr:0x%p elementSz: %llu", cmd.buckets.fileId, cmd.buckets.buffers, (llu)cmd.buckets.elementSize );
            #endif
            CmdWriteBuckets( cmd, cmd.buckets.elementSize );
        break;

        case Command::WriteFile:
            #if DBG_LOG_ENABLE
                Log::Debug( "[DiskBufferQueue] ^ Cmd WriteFile: (%u) bucket:%u sz:%llu addr:0x%p", cmd.file.fileId, cmd.file.bucket, cmd.file.size, cmd.file.buffer );
            #endif
            CndWriteFile( cmd );
        break;

        case Command::ReadBucket:
            #if DBG_LOG_ENABLE
                Log::Debug( "[DiskBufferQueue] ^ Cmd ReadBucket: (%u) bucket:%u esz:%llu", cmd.readBucket.fileId, cmd.readBucket.elementSize );
            #endif
            CmdReadBucket( cmd );
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

        case Command::TruncateBucket:
            #if DBG_LOG_ENABLE
                Log::Debug( "[DiskBufferQueue] ^ Cmd TruncateBucket" );
            #endif
            CmdTruncateBucket( cmd );
        break;

    #if _DEBUG
        case Command::DBG_WriteSliceSizes:
            CmdDbgWriteSliceSizes( cmd );
        break;

        case Command::DBG_ReadSliceSizes:
        CmdDbgReadSliceSizes( cmd );
        break;
    #endif

        default:
            ASSERT( 0 );
        break;
    }
}

//-----------------------------------------------------------
void DiskBufferQueue::CmdWriteBuckets( const Command& cmd, const size_t elementSize )
{
    const FileId fileId  = cmd.buckets.fileId;
    const uint*  sizes   = cmd.buckets.writeSizes;
    const byte*  buffers = cmd.buckets.buffers;
    FileSet&     fileSet = _files[(int)fileId];

    // ASSERT( IsFlagSet( fileBuckets.files[0]->GetFileAccess(), FileAccess::ReadWrite ) );

    const uint32 bucketCount = (uint32)fileSet.files.length;

    Log::Debug( "  >>> Write 0x%p", buffers );

    // Single-threaded for now... We don't have file handles for all the threads yet!
    const size_t blockSize = fileSet.files[0]->BlockSize();
    
    const byte* buffer = buffers;

    if( IsFlagSet( fileSet.options, FileSetOptions::Interleaved ) || IsFlagSet( fileSet.options, FileSetOptions::Alternating ) )
    {
        const uint32* sliceSizes = cmd.buckets.sliceSizes;

        // Write in interleaved mode, a whole bucket is written at once
        size_t writeSize = 0;

        for( uint i = 0; i < bucketCount; i++ )
        {
            const size_t sliceWriteSize = sizes[i] * elementSize;
            ASSERT( sliceWriteSize / blockSize * blockSize == sliceWriteSize );
            
            // Save slice sizes for reading-back the bucket
            fileSet.writeSliceSizes[fileSet.writeBucket][i] = sliceSizes[i] * elementSize;    // #TODO: Should we not apply element size here?

            writeSize += sliceWriteSize;
        }

        // #TODO: Do we have to round-up the size to block boundary?
        ASSERT( writeSize / blockSize * blockSize == writeSize );
        ASSERT( fileSet.writeBucket < fileSet.files.Length() );

        if( IsFlagSet( fileSet.options, FileSetOptions::Alternating ) )
        {
            const bool interleaved = cmd.buckets.interleaved;

            // When in alternating mode we have to issue bucketCount writes per bucket as we need t o ensure
            // they are all offset to the bucket slice boundary.
            // #NOTE: We can avoid this on interleaved writes if we add that said offset to the prefix um offset.
            const uint64 maxSliceSize = fileSet.maxSliceSize;

            for( uint slice = 0; slice < bucketCount; slice++ )
            {
                ASSERT( sizes[slice] <= maxSliceSize / elementSize );

                const size_t   sliceWriteSize = sizes[slice] * elementSize;
                const uint32   fileBucketIdx  = interleaved ? fileSet.writeBucket : slice;
                      IStream& file           = *fileSet.files[fileBucketIdx];

                // Seek to the start of the (fixed-size) slice boundary
                {
                    const uint32 sliceSeekIdx = interleaved ? slice : fileSet.writeBucket;
                    const int64  sliceOffset  = (int64)( sliceSeekIdx * maxSliceSize );

                    FatalIf( !file.Seek( sliceOffset, SeekOrigin::Begin ),
                        "Failed to seek file %s.%u.tmp to slice boundary.", fileSet.name, fileBucketIdx );
                }

                WriteToFile( file, sliceWriteSize, buffer, (byte*)fileSet.blockBuffer, fileSet.name, fileBucketIdx );

                buffer += sliceWriteSize;
            }
        }
        else
        {
            WriteToFile( *fileSet.files[fileSet.writeBucket], writeSize, buffer, (byte*)fileSet.blockBuffer, fileSet.name, fileSet.writeBucket );
        }

        if( ++fileSet.writeBucket >= bucketCount )
        {
            ASSERT( fileSet.writeBucket <= fileSet.files.Length() );

            // When the last bucket was written, reset the write bucket and swap file slices
            fileSet.writeBucket = 0;
            std::swap( fileSet.writeSliceSizes, fileSet.readSliceSizes );
        }
    }
    else
    {
        for( uint i = 0; i < bucketCount; i++ )
        {
            const size_t bufferSize = sizes[i] * elementSize;

            // Only write up-to the block-aligned boundary. The caller is in charge of handling unlaigned data.
            ASSERT( bufferSize == bufferSize / blockSize * blockSize );
            WriteToFile( *fileSet.files[i], bufferSize, buffer, (byte*)fileSet.blockBuffer, fileSet.name, i );

            // ASSERT( IsFlagSet( fileBuckets.files[i].GetFileAccess(), FileAccess::ReadWrite ) );
            buffer += bufferSize;
        }
    }
}

//-----------------------------------------------------------
void DiskBufferQueue::CndWriteFile( const Command& cmd )
{
    FileSet& fileBuckets = _files[(int)cmd.file.fileId];
    WriteToFile( *fileBuckets.files[cmd.file.bucket], cmd.file.size, cmd.file.buffer, (byte*)fileBuckets.blockBuffer, fileBuckets.name, cmd.file.bucket );
}

//-----------------------------------------------------------
void DiskBufferQueue::CmdReadBucket( const Command& cmd )
{
    const FileId fileId      = cmd.readBucket.fileId;
    const size_t elementSize = cmd.readBucket.elementSize;
    FileSet&     fileSet     = _files[(int)fileId];
    const uint32 bucketCount = (uint32)fileSet.files.Length();

    const bool alternating               = IsFlagSet( fileSet.options, FileSetOptions::Alternating );
    const bool alternatingNonInterleaved = alternating && !cmd.readBucket.interleaved;

    // #NOTE: Not implemented in the single-read method for code simplicity.
    // if( IsFlagSet( fileSet.options, FileSetOptions::Alternating ) && !cmd.readBucket.interleaved )
    // {
    //     // Read a single file instead
    //     Command readCmd;
    //     readCmd.file.buffer = cmd.readBucket.buffer->Ptr();                               ASSERT( readCmd.file.buffer );
    //     readCmd.file.fileId = fileId;
    //     readCmd.file.bucket = fileSet.readBucket;
    //     readCmd.file.size   = fileSet.sliceSizes[fileSet.readBucket][0] * elementSize;        ASSERT( readCmd.file.size );

    //     CmdReadFile( cmd );
        
    //     // Update user buffer length
    //     auto userBuffer = const_cast<Span<byte>*>( cmd.readBucket.buffer );
    //     userBuffer->length = readCmd.file.size; //readCmd.file.size / elementSize;

    //     fileSet.readBucket = (fileSet.readBucket + 1) % fileSet.files.Length();
    //     return;
    // }
    
    ASSERT( IsFlagSet( fileSet.options, FileSetOptions::Interleaved ) || IsFlagSet( fileSet.options, FileSetOptions::Alternating ) );    // #TODO: Read bucket is just a single file read without this flag
    ASSERT( fileSet.readSliceSizes.Ptr() );
    // ASSERT( fileSet.readBucket == 0 );      // Should be in bucket 0 at this point // #NOTE: Perhaps have the user specify the bucket to read instead?
    
    const bool   directIO    = IsFlagSet( fileSet.options, FileSetOptions::DirectIO );
    const auto   sliceSizes  = fileSet.readSliceSizes;
    const size_t blockSize   = fileSet.files[0]->BlockSize();

    auto readBuffer  = Span<byte>( cmd.readBucket.buffer->Ptr(), cmd.readBucket.buffer->Length() * elementSize );
    auto blockBuffer = Span<byte>( (byte*)fileSet.blockBuffer, blockSize );

    Span<byte> tempBlock;

    const uint64 maxSliceSize = fileSet.maxSliceSize;

    for( uint32 slice = 0; slice < bucketCount; slice++ )
    {
        const size_t   sliceSize     = sliceSizes[slice][fileSet.readBucket];
        const size_t   readSize      = sliceSize + tempBlock.Length();
        const size_t   alignedSize   = CDivT( readSize, blockSize ) * blockSize;   // Sizes are written aligned, and must also be read aligned

        const uint32   fileBucketIdx = alternatingNonInterleaved ? fileSet.readBucket : slice;
              IStream& stream        = *fileSet.files[fileBucketIdx];

        // When alternating, we need to seek to the start of the slice boundary
        if( alternating )
        {
            const uint32 sliceOffsetIdx = alternatingNonInterleaved ? slice : fileSet.readBucket;
            const int64  sliceOffset    = (int64)( sliceOffsetIdx * maxSliceSize );

            FatalIf( !stream.Seek( sliceOffset, SeekOrigin::Begin ), 
                "Failed to seek while reading alternating bucket %s.%u.tmp.", fileSet.name, fileBucketIdx );
        }

        ReadFromFile( stream, alignedSize, readBuffer.Ptr(), nullptr, blockSize, directIO, fileSet.name, fileBucketIdx );

        // Replace the temp block we just overwrote, if we have one
        if( tempBlock.Length() )
            tempBlock.CopyTo( readBuffer );

        // Copy offset temporarily (only if readSize is not block-aligned)
        if( readSize < alignedSize )
        {
            ASSERT( alignedSize - readSize < blockSize );
            
            const auto   lastBlockOffset = alignedSize - blockSize;         ASSERT( readSize > lastBlockOffset );
            const size_t lastBlockSize   = readSize - lastBlockOffset;

            readBuffer = readBuffer.Slice( lastBlockOffset );
            tempBlock  = blockBuffer.Slice( 0, lastBlockSize );
            
            readBuffer.CopyTo( tempBlock, lastBlockSize );
        }
        else
        {
            readBuffer = readBuffer.Slice( alignedSize ); // We just read everything block aligned
            tempBlock  = {};
        }
    }

    size_t elementCount = 0;
    for( uint32 slice = 0; slice < bucketCount; slice++ )
        elementCount += sliceSizes[slice][fileSet.readBucket];

    fileSet.readBucket = (fileSet.readBucket + 1) % fileSet.files.Length();

    // Revert buffer length from bytes to element size
    auto userBuffer = const_cast<Span<byte>*>( cmd.readBucket.buffer );
    userBuffer->length = elementCount / elementSize;
}

//-----------------------------------------------------------
void DiskBufferQueue::CmdReadFile( const Command& cmd )
{
    FileSet& fileSet = _files[(int)cmd.file.fileId];
    const bool   directIO  = IsFlagSet( fileSet.options, FileSetOptions::DirectIO );
    const size_t blockSize = fileSet.files[0]->BlockSize();

    ReadFromFile( *fileSet.files[cmd.file.bucket], cmd.file.size, cmd.file.buffer, (byte*)fileSet.blockBuffer, blockSize, directIO, fileSet.name, cmd.file.bucket );
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
        
        // #TODO: Set the bucket number to 0 on origin == 0?
    }
}

//-----------------------------------------------------------
inline void DiskBufferQueue::WriteToFile( IStream& file, size_t size, const byte* buffer, byte* blockBuffer, const char* fileName, uint bucket )
{
    // if( !_useDirectIO )
    // {
        #if _DEBUG || BB_IO_METRICS_ON
            _writeMetrics.size += size;
            _writeMetrics.count++;
            const auto timer = TimerBegin();
        #endif

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

        #if _DEBUG || BB_IO_METRICS_ON
            _writeMetrics.time += TimerEndTicks( timer );
        #endif
    // }
    // else
    // {
    //     ASSERT()
    //     const size_t blockSize   = _blockSize;
    //     size_t       sizeToWrite = size / blockSize * blockSize;
    //     const size_t remainder   = size - sizeToWrite;

    //     while( sizeToWrite )
    //     {
    //         ssize_t sizeWritten = file.Write( buffer, sizeToWrite );
    //         if( sizeWritten < 1 )
    //         {
    //             const int err = file.GetError();
    //             Fatal( "Failed to write to '%s.%u' work file with error %d (0x%x).", fileName, bucket, err, err );
    //         }

    //         ASSERT( sizeWritten <= (ssize_t)sizeToWrite );

    //         sizeToWrite -= (size_t)sizeWritten;
    //         buffer      += sizeWritten;
    //     }
        
    //     if( remainder )
    //     {
    //         ASSERT( blockBuffer );
            
    //         // Unnecessary zeroing of memory, but might be useful for debugging
    //         memset( blockBuffer, 0, blockSize );
    //         memcpy( blockBuffer, buffer, remainder );

    //         ssize_t sizeWritten = file.Write( blockBuffer, blockSize );

    //         if( sizeWritten < remainder )
    //         {
    //             const int err = file.GetError();
    //             Fatal( "Failed to write block to '%s.%u' work file with error %d (0x%x).", fileName, bucket, err, err );
    //         }
    //     }
    // }
}

//-----------------------------------------------------------
inline void DiskBufferQueue::ReadFromFile( IStream& file, size_t size, byte* buffer, byte* blockBuffer, const size_t blockSize, const bool directIO, const char* fileName, const uint bucket )
{
    #if _DEBUG || BB_IO_METRICS_ON
        _readMetrics.size += size;
        _readMetrics.count++;
        const auto timer = TimerBegin();
    #endif

    if( !directIO )
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
        int err;
        FatalIf( !IOJob::ReadFromFile( file, buffer, size, blockBuffer, blockSize, err ),
            "Failed to read from '%s_%u' work file with error %d (0x%x).", fileName, bucket, err, err );


    //     const size_t blockSize  = _blockSize;
        
    //     /// #NOTE: All our buffers should be block aligned so we should be able to freely read all blocks to them...
    //     ///       Only implement remainder block reading if we really need to.
    // //     size_t       sizeToRead = size / blockSize * blockSize;
    // //     const size_t remainder  = size - sizeToRead;

    //     size_t sizeToRead = CDivT( size, blockSize ) * blockSize;

    //     while( sizeToRead )
    //     {
    //         ssize_t sizeRead = file.Read( buffer, sizeToRead );
    //         if( sizeRead < 1 )
    //         {
    //             const int err = file.GetError();
    //             Fatal( "Failed to read from '%s_%u' work file with error %d (0x%x).", fileName, bucket, err, err );
    //         }

    //         ASSERT( sizeRead <= (ssize_t)sizeToRead );
            
    //         sizeToRead -= (size_t)sizeRead;
    //         buffer     += sizeRead;
    //     }
    }

    #if _DEBUG || BB_IO_METRICS_ON
        _readMetrics.time += TimerEndTicks( timer );
    #endif

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
    DeleteFileNow( cmd.deleteFile.fileId, cmd.deleteFile.bucket );

    // FileDeleteCommand delCmd;
    // delCmd.fileId = cmd.deleteFile.fileId;
    // delCmd.bucket = (int64)cmd.deleteFile.bucket;
    // while( !_deleteQueue.Enqueue( delCmd ) );

    // _deleteSignal.Signal();
}

//----------------------------------------------------------
void DiskBufferQueue::CmdDeleteBucket( const Command& cmd )
{
    DeleteBucketNow( cmd.deleteFile.fileId );

    // FileDeleteCommand delCmd;
    // delCmd.fileId = cmd.deleteFile.fileId;
    // delCmd.bucket = -1;
    // while( !_deleteQueue.Enqueue( delCmd ) );
    
    // _deleteSignal.Signal();
}

//-----------------------------------------------------------
void DiskBufferQueue::CmdTruncateBucket( const Command& cmd )
{
    const auto& tcmd = cmd.truncateBucket;

    FileSet& files = _files[(int)tcmd.fileId];

    for( size_t i = 0; i < files.files.Length(); i++ )
    {
        const bool r = files.files[i]->Truncate( tcmd.position );
        if( !r )
        {
            ASSERT( 0 );
            Log::Line( "Warning: Failed to truncate file %s:%llu", files.name, (uint64)i );
        }
    }
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

        case DiskBufferQueue::Command::TruncateBucket:
            return "TruncateBucket";

        default:
            ASSERT( 0 );
            return nullptr;
    }
}

#if _DEBUG
//-----------------------------------------------------------
void DiskBufferQueue::CmdDbgWriteSliceSizes( const Command& cmd )
{
    const FileId fileId = cmd.dbgSliceSizes.fileId;;
    const char*  fName  = ( fileId == FileId::FX0   || fileId == FileId::FX1   ) ? "y"     :
                          ( fileId == FileId::META0 || fileId == FileId::META1 ) ?  "meta" : "index";

    char path[1024];
    sprintf( path, "%st%d.%s.slices.tmp", BB_DP_DBG_TEST_DIR, (int)cmd.dbgSliceSizes.table+1, fName );

    FileStream file;
    if( !file.Open( path, FileMode::Create, FileAccess::Write ) )
    {
        Log::Error( "Failed to open '%s' for writing.", path );
        return;
    }
    // FatalIf( !file.Open( path, FileMode::Create, FileAccess::Write ), "Failed to open '%s' for writing.", path );

    FileSet& fileSet        = _files[(int)cmd.dbgSliceSizes.fileId];
    const uint32 numBuckets = (uint32)fileSet.files.Length();

    for( uint32 i = 0; i < numBuckets; i++ )
    {
              auto   slices    = fileSet.writeSliceSizes[i];
        const size_t sizeWrite = sizeof( size_t ) * numBuckets;

        if( file.Write( slices.Ptr(), sizeWrite ) != sizeWrite )
        {
            Log::Error( "Failed to write slice size for table %d", (int)cmd.dbgSliceSizes.table+1 );
            return;
        }
        // FatalIf( file.Write( slices.Ptr(), sizeWrite ) != sizeWrite,
        //     "Failed to write slice size for table %d", (int)cmd.dbgSliceSizes.table+1  );
    }
}

//-----------------------------------------------------------
void DiskBufferQueue::CmdDbgReadSliceSizes( const Command& cmd )
{
    const FileId fileId = cmd.dbgSliceSizes.fileId;;
    const char*  fName  = ( fileId == FileId::FX0   || fileId == FileId::FX1   ) ? "y"     :
                          ( fileId == FileId::META0 || fileId == FileId::META1 ) ?  "meta" : "index";

    char path[1024];
    sprintf( path, "%st%d.%s.slices.tmp", BB_DP_DBG_TEST_DIR, (int)cmd.dbgSliceSizes.table+1, fName );

    FileStream file;
    FatalIf( !file.Open( path, FileMode::Open, FileAccess::Read ), "Failed to open '%s' for reading.", path );

    FileSet& fileSet        = _files[(int)cmd.dbgSliceSizes.fileId];
    const uint32 numBuckets = (uint32)fileSet.files.Length();

    for( uint32 i = 0; i < numBuckets; i++ )
    {
              auto   slices   = fileSet.readSliceSizes[i];
        const size_t sizeRead = sizeof( size_t ) * numBuckets;

        FatalIf( file.Read( slices.Ptr(), sizeRead ) != sizeRead,
            "Failed to read slice size for table %d", (int)cmd.dbgSliceSizes.table+1  );
    }
}


#endif

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
inline void DiskBufferQueue::CloseFileNow( const FileId fileId, const uint32 bucket )
{
    FileSet& fileSet = _files[(int)fileId];

    // NOTE: Why are we doing it this way?? Just add Close() to IStream.
    const bool isHybridFile = IsFlagSet( fileSet.options, FileSetOptions::Cachable );
    if( isHybridFile )
    {
        auto* file = static_cast<HybridStream*>( fileSet.files[bucket] );
        file->Close();
    }
    else
    {
        auto* file = static_cast<FileStream*>( fileSet.files[bucket] );
        file->Close();
    }
}

//-----------------------------------------------------------
void DiskBufferQueue::DeleteFileNow( const FileId fileId, const uint32 bucket )
{
    FileSet& fileSet = _files[(int)fileId];

    CloseFileNow( fileId, bucket );

    const bool useTmp2 = IsFlagSet( fileSet.options, FileSetOptions::UseTemp2 );

    const std::string& wokrDir  = useTmp2 ? _workDir2 : _workDir1;
                 char* filePath = _delFilePathBuffer;

    memcpy( filePath, wokrDir.c_str(), wokrDir.length() );
    char* baseName = filePath + wokrDir.length();
    
    sprintf( baseName, "%s_%u.tmp", fileSet.name, bucket );
    
    const int r = remove( filePath );

    if( r )
        Log::Error( "Error: Failed to delete file %s with errror %d (0x%x).", filePath, r, r );
}

//-----------------------------------------------------------
void DiskBufferQueue::DeleteBucketNow( const FileId fileId )
{
    FileSet& fileSet = _files[(int)fileId];

    const bool useTmp2 = IsFlagSet( fileSet.options, FileSetOptions::UseTemp2 );

    const std::string& wokrDir  = useTmp2 ? _workDir2 : _workDir1;
                 char* filePath = _delFilePathBuffer;

    memcpy( filePath, wokrDir.c_str(), wokrDir.length() );
    char* baseName = filePath + wokrDir.length();

    for( size_t i = 0; i < fileSet.files.length; i++ )
    {
        CloseFileNow( fileId, (uint32)i );

        sprintf( baseName, "%s_%u.tmp", fileSet.name, (uint)i );
    
        const int r = remove( filePath );

        if( r )
            Log::Error( "Error: Failed to delete file %s with errror %d (0x%x).", filePath, r, r );
    }
}

