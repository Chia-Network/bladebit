#pragma once

#include "threading/MTJob.h"

class IStream;
class ThreadPool;

struct IOJob : MTJob<IOJob>
{
    IStream*    _file;
    size_t      _blockSize;
    size_t      _size;
    byte*       _buffer;
    byte*       _blockBuffer;
    int64       _offset;        // If not zero, the file will be seeked to this relative offset before writing.
    int         _error;
    bool        _isWrite;       // True if the job is a write job

    static bool WriteTest( const char* testPath, size_t testSize, uint threadCount );

    static bool MTWrite( ThreadPool& pool, uint32 threadCount, 
        IStream** files, 
        const void* bufferToWrite, const size_t sizeToWrite,
        void** blockBuffers, const size_t blockSize,
        int& error );

    static bool MTRead( ThreadPool& pool, uint32 const threadCount, 
        IStream** files, 
        void*  dstBuffer, const size_t sizeToRead,
        void** blockBuffers, const size_t blockSize,
        int& error );

    void Run() override;

    static bool WriteToFile( IStream& file, const void* writeBuffer, const size_t size,
                             void* fileBlockBuffer, const size_t blockSize, int& error );

    static bool ReadFromFile( IStream& file, void* buffer, const size_t size,
                              void* blockBuffer, const size_t blockSize, int& error );

    static bool ReadFromFile( const char* path, void* buffer, const size_t size,
                              void* blockBuffer, const size_t blockSize, int& error );

    static void* ReadAllBytesDirect( const char* path, int& error );

private:
    static bool RunIOJob( bool write, ThreadPool& pool, uint32 threadCount, 
        IStream** files, 
        byte* ioBuffer, const size_t size,
        byte** blockBuffers, const size_t blockSize,
        int& error );
};