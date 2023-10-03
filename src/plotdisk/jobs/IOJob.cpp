#include "IOJob.h"
#include "threading/ThreadPool.h"
#include "threading/MTJob.h"
#include "util/Util.h"
#include "io/IStream.h"
#include "io/FileStream.h"

//-----------------------------------------------------------
bool IOJob::MTWrite( ThreadPool& pool, uint32 threadCount, 
    IStream**   files, 
    const void* bufferToWrite, const size_t sizeToWrite,
    void**      blockBuffers,  const size_t blockSize,
    int&        error )
{
    return RunIOJob( true, pool, threadCount, files, (byte*)bufferToWrite, sizeToWrite, 
                     (byte**)blockBuffers, blockSize, error );
}

//-----------------------------------------------------------
bool IOJob::MTRead( ThreadPool& pool, uint32 threadCount, 
    IStream** files, 
    void*     dstBuffer,    const size_t sizeToRead,
    void**    blockBuffers, const size_t blockSize,
    int&      error )
{
    
    return RunIOJob( false, pool, threadCount, files, (byte*)dstBuffer, sizeToRead, 
                     (byte**)blockBuffers, blockSize, error );
}

//-----------------------------------------------------------
bool IOJob::RunIOJob( bool write, 
    ThreadPool& pool, 
    uint32      threadCount, 
    IStream**   files, 
    byte*       ioBuffer,     const size_t size,
    byte**      blockBuffers, const size_t blockSize,
    int&        error )
{
    ASSERT( files );
    ASSERT( threadCount <= pool.ThreadCount() );
    ASSERT( ioBuffer     );
    ASSERT( blockBuffers );
    ASSERT( size         );
    ASSERT( blockSize    );


    error       = 0;
    threadCount = std::max( 1u, std::min( threadCount, pool.ThreadCount() ) );

    // For small writes use a single thread
    const size_t minWrite = std::max( blockSize, (size_t)(16 MiB) );

    if( size <= minWrite || threadCount == 1 )
    {
        if( write )
            return WriteToFile( **files, ioBuffer, size, blockBuffers[0], blockSize, error );
        else
            return ReadFromFile( **files, ioBuffer, size, blockBuffers[0], blockSize, error );
    }


    MTJobRunner<IOJob> jobs( pool );

    // Size per thread, aligned to block size
    const size_t sizePerThread = size / threadCount / blockSize * blockSize;
    size_t       sizeRemainder = size - sizePerThread * threadCount;


    size_t fileOffset  = 0;

    for( uint32 i = 0; i < threadCount; i++ )
    {
        ASSERT( blockBuffers[i] );

        auto& job = jobs[i];
        
        job._file        = files[i];
        job._blockSize   = blockSize;
        job._size        = sizePerThread;
        job._buffer      = ioBuffer + fileOffset;
        job._blockBuffer = (byte*)blockBuffers[i];
        job._offset      = (int64)fileOffset;
        job._error       = 0;
        job._isWrite     = write;

        if( fileOffset > 0 )
        {
            if( !job._file->Seek( fileOffset, SeekOrigin::Current ) )
            {
                error = job._file->GetError();
                return false;
            }
        }

        if( sizeRemainder >= blockSize )
        {
            job._size += blockSize;
            sizeRemainder -= blockSize;
        }

        fileOffset += job._size;
    }

    jobs[threadCount-1]._size += sizeRemainder;
    jobs.Run( threadCount );

    // Find the first job with an error
    for( uint32 i = 0; i < threadCount; i++ )
    {
        if( jobs[i]._error != 0 )
        {
            error = jobs[i]._error;
            return false;
        }
    }

    return true;
}

//-----------------------------------------------------------
void IOJob::Run()
{
    if( _isWrite )
        WriteToFile( *_file, _buffer, _size, _blockBuffer, _blockSize, _error );
    else
        ReadFromFile( *_file, _buffer, _size, _blockBuffer, _blockSize, _error );
}

//-----------------------------------------------------------
bool IOJob::WriteToFile( const char* filePath, const void* writeBuffer, const size_t size, int& error )
{
    FileStream file;
    if( !file.Open( filePath, FileMode::Create, FileAccess::Write, FileFlags::NoBuffering ) )
    {
        error = file.GetError();
        return false;
    }

    void* block = bbvirtalloc( file.BlockSize() );
    const bool r = WriteToFile( file, writeBuffer, size, block, file.BlockSize(), error );
    bbvirtfree( block );

    return r;
}

//-----------------------------------------------------------
bool IOJob::WriteToFile( IStream& file, const void* writeBuffer, const size_t size,
                         void* fileBlockBuffer, const size_t blockSize, int& error, size_t* outSizeWritten )                 
{
    error = 0;

    const byte* buffer      = (byte*)writeBuffer;
    byte*       blockBuffer = (byte*)fileBlockBuffer;

    const size_t totalSizeToWrite = size / blockSize * blockSize;           ASSERT( totalSizeToWrite <= size );

    size_t       sizeToWrite = totalSizeToWrite;
    const size_t remainder   = size - sizeToWrite;                          ASSERT( remainder < blockSize );
    ASSERT( !remainder || blockBuffer );

    while( sizeToWrite )
    {
        const ssize_t sizeWritten = file.Write( buffer, sizeToWrite );

        if( sizeWritten < 1 )
        {
            // Output size written thus far
            if( outSizeWritten )
                *outSizeWritten = totalSizeToWrite - sizeToWrite;

            error = file.GetError();
            return false;
        }

        ASSERT( sizeWritten <= (ssize_t)sizeToWrite );

        sizeToWrite -= (size_t)sizeWritten;
        buffer      += sizeWritten;
    }

    // Write unaligned portion, if any
    if( remainder )
    {
        if( !blockBuffer )
        {
            // All aligned data was written (if there was any)
            if( outSizeWritten )
                *outSizeWritten = totalSizeToWrite;

            error = -1;
            return false;
        }

        // Unnecessary zeroing of memory, but might be useful for debugging
        memset( blockBuffer, 0, blockSize );
        memcpy( blockBuffer, buffer, remainder );

        const ssize_t sizeWritten = file.Write( blockBuffer, blockSize );

        if( sizeWritten < 1 )
        {
            // All aligned data was written (if there was any)
            if( outSizeWritten )
                *outSizeWritten = totalSizeToWrite;

            error = file.GetError();
            return false;
        }

        // Expect to always write a full block.
        if( (size_t)sizeWritten != blockSize )
        {
            if( outSizeWritten )
                *outSizeWritten = totalSizeToWrite + (size_t)sizeWritten;

            error = -2;
            return false;
        }
    }

    if( outSizeWritten )
        *outSizeWritten = size;

    return true;
}

//-----------------------------------------------------------
bool IOJob::WriteToFileUnaligned( const char* filePath, const void* writeBuffer, const size_t size, int& error )
{
    FileStream file;
    if( !file.Open( filePath, FileMode::Create, FileAccess::Write, FileFlags::None ) )
    {
        error = file.GetError();
        return false;
    }

    return WriteToFileUnaligned( file, writeBuffer, size, error );
}

//-----------------------------------------------------------
bool IOJob::WriteToFileUnaligned( IStream& file, const void* writeBuffer, const size_t size, int& error )
{
    error = 0;

    const byte* buffer      = (byte*)writeBuffer;
    size_t      sizeToWrite = size;

    while( sizeToWrite )
    {
        ssize_t sizeWritten = file.Write( buffer, sizeToWrite );
        if( sizeWritten < 1 )
        {
            error = file.GetError();
            return false;
        }

        ASSERT( sizeWritten <= (ssize_t)sizeToWrite );

        sizeToWrite -= (size_t)sizeWritten;
        buffer      += sizeWritten;
    }

    return true;
}

//-----------------------------------------------------------
void* IOJob::ReadAllBytesDirect( const char* path, int& error )
{
    size_t byteCount = 0;
    return ReadAllBytesDirect( path, error, byteCount );
}

//-----------------------------------------------------------
void* IOJob::ReadAllBytesDirect( const char* path, int& error, size_t& byteCount )
{
    byteCount = 0;

    FileStream file;
    if( !file.Open( path, FileMode::Open, FileAccess::Read, FileFlags::NoBuffering ) )
        return nullptr;

    const size_t blockSize = file.BlockSize();
    const size_t readSize  = file.Size();
    const size_t allocSize = RoundUpToNextBoundaryT( readSize, blockSize );

    void* block  = bbvirtalloc( blockSize );
    void* buffer = bbvirtalloc( allocSize );

    const bool r = ReadFromFile( file, buffer, readSize, block, blockSize, error );

    bbvirtfree( block );
    if( !r )
    {
        bbvirtfree( buffer );
        return nullptr;
    }

    byteCount = readSize;
    return buffer;
}


//-----------------------------------------------------------
bool IOJob::ReadFromFile( const char* path, void* buffer, const size_t size,
                          void* blockBuffer, const size_t blockSize, int& error )
{
    FileStream file;
    if( !file.Open( path, FileMode::Open, FileAccess::Read ) )
    {
        error = file.GetError();
        return false;
    }

    return ReadFromFile( file, buffer, size, blockBuffer, blockSize, error );
}

//-----------------------------------------------------------
bool IOJob::ReadFromFile( const char* path, void* buffer, const size_t size, int& error )
{
    ASSERT( path   );
    ASSERT( buffer );

    FileStream file;
    if( !file.Open( path, FileMode::Open, FileAccess::Read, FileFlags::NoBuffering ) )
        return false;

    const size_t blockSize = file.BlockSize();

    void* block  = bbvirtalloc( blockSize );
    const bool r = ReadFromFile( file, buffer, size, block, blockSize, error );

    bbvirtfree( block );

    return r;
}

//-----------------------------------------------------------
bool IOJob::ReadFromFile( IStream& file, void* readBuffer, const size_t size,
                          void* fileBlockBuffer, const size_t blockSize, int& error )
{
    error = 0;

    byte* buffer      = (byte*)readBuffer;
    byte* blockBuffer = (byte*)fileBlockBuffer;

    size_t       sizeToRead = size / blockSize * blockSize;
    const size_t remainder  = size - sizeToRead;

    // size_t sizeToRead = CDivT( size, blockSize ) * blockSize;

    while( sizeToRead )
    {
        ssize_t sizeRead = file.Read( buffer, sizeToRead );
        if( sizeRead < 1 )
        {
            error = file.GetError();
            return false;
        }

        ASSERT( sizeRead <= (ssize_t)sizeToRead );
        
        sizeToRead -= (size_t)sizeRead;
        buffer     += sizeRead;
    }

    if( remainder )
    {
        if( blockBuffer == nullptr )
        {
            error = -1;
            return false;
        }

        ssize_t sizeRead = file.Read( blockBuffer, blockSize );

        if( sizeRead < (ssize_t)remainder )
        {
            error = file.GetError();
            return false;
        }

        memcpy( buffer, blockBuffer, remainder );
    }

    return true;
}

//-----------------------------------------------------------
bool IOJob::ReadFromFileUnaligned( const char* path, void* buffer, const size_t size, int& error )
{
    FileStream file;
    if( !file.Open( path, FileMode::Open, FileAccess::Read ) )
    {
        error = file.GetError();
        return false;
    }

    return ReadFromFileUnaligned( file, buffer, size, error );
}

//-----------------------------------------------------------
bool IOJob::ReadFromFileUnaligned( IStream& file, void* buffer, const size_t size, int& error )
{
    return ReadFromFile( file, buffer, size, nullptr, 1, error );
}

