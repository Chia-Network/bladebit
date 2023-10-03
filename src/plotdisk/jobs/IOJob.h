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

    static bool WriteToFile( const char* filePath, const void* writeBuffer, const size_t size, int& error );

    // Aligned write.
    // Guaranteed to write all data in the buffer, if not it returns false and sets the error.
    // Negative error values are non-OS errors.
    static bool WriteToFile( IStream& file, const void* writeBuffer, const size_t size,
                             void* fileBlockBuffer, const size_t blockSize, int& error, size_t* outSizeWritten = nullptr );
    
    static bool WriteToFileUnaligned( const char* filePath, const void* writeBuffer, const size_t size, int& error );
    static bool WriteToFileUnaligned( IStream& file, const void* writeBuffer, const size_t size, int& error );

    static bool ReadFromFile( IStream& file, void* buffer, const size_t size,
                              void* blockBuffer, const size_t blockSize, int& error );

    static bool ReadFromFile( const char* path, void* buffer, const size_t size,
                              void* blockBuffer, const size_t blockSize, int& error );

    static bool ReadFromFile( const char* path, void* buffer, const size_t size, int& error );


    static bool ReadFromFileUnaligned( const char* path, void* buffer, const size_t size, int& error );
    static bool ReadFromFileUnaligned( IStream& file, void* buffer, const size_t size, int& error );

    static void* ReadAllBytesDirect( const char* path, int& error );
    
    static void* ReadAllBytesDirect( const char* path, int& error, size_t& byteCount );

private:

    static bool RunIOJob( bool write, ThreadPool& pool, uint32 threadCount, 
        IStream** files, 
        byte* ioBuffer, const size_t size,
        byte** blockBuffers, const size_t blockSize,
        int& error );
};