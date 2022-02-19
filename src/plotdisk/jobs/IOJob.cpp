#include "IOJob.h"
#include "threading/ThreadPool.h"
#include "threading/MTJob.h"
#include "io/FileStream.h"
#include "util/Util.h"

//-----------------------------------------------------------
double IOWriteJob::WriteWithThreads( 
        uint threadCount, ThreadPool& pool, FileStream* files, 
        const byte* bufferToWrite, size_t sizeToWrite,
        byte** blockBuffers, const size_t blockSize,
        int& error )
{
    error = 0;

    ASSERT( threadCount <= pool.ThreadCount() );
    ASSERT( bufferToWrite );
    ASSERT( blockBuffers  );
    ASSERT( sizeToWrite   );
    ASSERT( blockSize     );

    const size_t minWrite = std::max( blockSize, (size_t)16 MB );

    // For small writes or block-sized writes, use a single thread thread
    if( sizeToWrite <= minWrite || threadCount == 1 )
    {
        auto timer = TimerBegin();
        WriteToFile( *files, bufferToWrite, sizeToWrite, blockBuffers[0], blockSize, error );
        return TimerEnd( timer );
    }

    MTJobRunner<IOWriteJob> jobs( pool );

    // Size per thread, aligned to block size
    const size_t sizePerThread = sizeToWrite / threadCount / blockSize * blockSize;
    const size_t sizeRemainder = sizeToWrite - sizePerThread * threadCount;
    // #TODO: Worth spreading left over blocks between threads?

    for( uint64 i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];
        
        job._file        = files+i;
        job._blockSize   = blockSize;
        job._size        = sizePerThread;
        job._buffer      = bufferToWrite + sizePerThread * i;
        job._blockBuffer = blockBuffers[i];
        job._offset      = (int64)(sizePerThread * i);
        job._error       = &error;

        if( job._offset > 0 )
        {
            if( !job._file->Seek( job._offset, SeekOrigin::Current ) )
            {
                error = job._file->GetError();
                return 0;
            }
        }

        ASSERT( job._blockBuffer );
    }

    jobs[threadCount-1]._size += sizeRemainder;

    const double elapsed = jobs.Run();

    return elapsed;
}

//-----------------------------------------------------------
void IOWriteJob::Run()
{
    int error;
    WriteToFile( *_file, _buffer, _size, _blockBuffer, _blockSize, error );
    
    // More than one thread can override this, but that's fine.
    if( error )
        *_error = error;
}

//-----------------------------------------------------------
void IOWriteJob::WriteToFile( FileStream& file, const byte* buffer, const size_t size,
                              byte* blockBuffer, const size_t blockSize, int& error )                 
{
    error = 0;

    size_t       sizeToWrite = size / blockSize * blockSize;
    const size_t remainder   = size - sizeToWrite;

    while( sizeToWrite )
    {
        ssize_t sizeWritten = file.Write( buffer, sizeToWrite );
        if( sizeWritten < 1 )
        {
            error = file.GetError();
            return;
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

        if( sizeWritten < 1 )
        {
            error = file.GetError();
        }
    }
}

//-----------------------------------------------------------
bool IOWriteJob::ReadFromFile( FileStream& file, byte* buffer, const size_t size,
                               byte* blockBuffer, const size_t blockSize, int& error )
{
    /// #NOTE: All our buffers should be block aligned so we should be able to freely read all blocks to them...
    ///       Only implement remainder block reading if we really need to.
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
