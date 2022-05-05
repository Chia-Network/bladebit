#pragma once

enum class SeekOrigin : int32
{
    Begin  = 0,
    Current,
    End
};


// Stream Interface
class IStream
{
public:
    inline virtual ~IStream() {}

    virtual ssize_t Read( void* buffer, size_t size ) = 0;

    virtual ssize_t Write( const void* buffer, size_t size ) = 0;

    virtual bool Seek( int64 offset, SeekOrigin origin ) = 0;

    virtual bool Flush() = 0;

    virtual size_t BlockSize() const = 0;

    virtual ssize_t Size() = 0;

    virtual bool Truncate( const ssize_t length ) = 0;

    virtual int GetError() = 0;

};