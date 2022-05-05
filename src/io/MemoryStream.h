#pragma once
#include "IStream.h"

class MemoryStream : public IStream
{
public:

    ssize_t Read( void* buffer, size_t size ) override;

    ssize_t Write( const void* buffer, size_t size ) override;

    bool Seek( int64 offset, SeekOrigin origin ) override;

    bool Flush() override;

    size_t BlockSize() const override;

    ssize_t Size() override;

    int GetError() override;
};