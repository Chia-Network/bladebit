#pragma once
#include "FileStream.h"

class HybridStream : public IStream
{

public:
    inline HybridStream() {}
    inline ~HybridStream() {}

    bool Open( void* memory, ssize_t memorySize, const char* path, FileMode mode, FileAccess access, FileFlags flags = FileFlags::None );
    // bool Open( ssize_t memorySize, const char* path, FileMode mode, FileAccess access, FileFlags flags = FileFlags::None );
    void Close();

    ssize_t Read( void* buffer, size_t size ) override;

    ssize_t Write( const void* buffer, size_t size ) override;

    bool Seek( int64 offset, SeekOrigin origin ) override;

    bool Flush() override;

    size_t BlockSize() const override;

    ssize_t Size() override;
    
    bool Truncate( const ssize_t length ) override;

    int GetError() override;

    inline bool IsOpen() const { return _file.IsOpen(); }

    FileStream& File() { return _file; }

private:
    FileStream _file;                       // Backing file
    byte*      _memory        = nullptr;    // Memory buffer
    size_t     _memSize       = 0;
    size_t     _position      = 0;    
    int        _error         = 0;
};