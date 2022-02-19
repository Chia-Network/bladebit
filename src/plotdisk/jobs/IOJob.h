#pragma once

#include "threading/MTJob.h"

class FileStream;
class ThreadPool;

struct IOWriteJob : MTJob<IOWriteJob>
{
    FileStream* _file;
    size_t      _blockSize;
    size_t      _size;
    const byte* _buffer;
    byte*       _blockBuffer;
    int64       _offset;        // If not zero, the file will be seeked to
    int*        _error;         //  this relative offset before writing.

    static bool WriteTest( const char* testPath, size_t testSize, uint threadCount );

    static double WriteWithThreads( 
        uint threadCount, ThreadPool& pool, FileStream* files, 
        const byte* bufferToWrite,size_t sizeToWrite,
        byte** blockBuffers, const size_t blockSize,
        int& error );

    void Run() override;

    static void WriteToFile( FileStream& file, const byte* buffer, const size_t size,
                             byte* blockBuffer, const size_t blockSize, int& error );

    static bool ReadFromFile( FileStream& file, byte* buffer, const size_t size,
                              byte* blockBuffer, const size_t blockSize, int& error );
};