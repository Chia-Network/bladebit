#pragma once
#include "IAllocator.h"

class IStackAllocator : public IAllocator
{
public:

    virtual size_t Size() const = 0;
    virtual void Pop( const size_t size ) = 0;
    virtual void PopToMarker( const size_t sizeMarker ) = 0;
};


class DummyAllocator : public IStackAllocator
{
public:
    //-----------------------------------------------------------
    void* Alloc( const size_t size, const size_t alignment ) override
    {
        const size_t paddedSize = RoundUpToNextBoundaryT( _size, alignment );
        _size = paddedSize + size;
        return nullptr;
    }

    //-----------------------------------------------------------
    inline size_t Size() const override { return _size; }

     //-----------------------------------------------------------
    inline void Pop( const size_t size ) override
    {
        ASSERT( size >= _size );
        _size -= size;
    }

    //-----------------------------------------------------------
    inline void PopToMarker( const size_t sizeMarker ) override
    {
        ASSERT( sizeMarker <= _size );
        _size = sizeMarker;
    }

private:
    size_t _size = 0;
};

// Fixed-capacity stack allocator
class StackAllocator : public IStackAllocator
{
public:

    //-----------------------------------------------------------
    inline StackAllocator( void* buffer, const size_t capacity )
        : _buffer  ( (byte*)buffer )
        , _capacity( capacity )
    {}

    //-----------------------------------------------------------
    inline void* Alloc( const size_t size, const size_t alignment ) override
    {
        // Start address must be aligned to the specified alignment
        const size_t paddedSize = RoundUpToNextBoundaryT( _size, alignment );

        ASSERT( size > 0 );
        ASSERT( _size < _capacity ); 
        ASSERT( paddedSize <= _capacity );
        FatalIf( !(_capacity - paddedSize >= size), "Allocation buffer overrun." );

        void* ptr = reinterpret_cast<void*>( _buffer + paddedSize );
        _size = paddedSize + size;

        return ptr;
    }

    //-----------------------------------------------------------
    inline byte* Buffer() const
    {
        return _buffer;
    }

    //-----------------------------------------------------------
    inline size_t Capacity() const
    {
        return _capacity;
    }

    //-----------------------------------------------------------
    inline size_t Size() const override
    {
        return _size;
    }

    //-----------------------------------------------------------
    inline size_t Remainder() const
    {
        return _capacity - _size;
    }

    //-----------------------------------------------------------
    inline byte* Ptr() const
    {
        return _buffer;
    }

    //-----------------------------------------------------------
    inline byte* Top() const
    {
        return _buffer + _size;
    }

    //-----------------------------------------------------------
    inline void Pop( const size_t size ) override
    {
        ASSERT( size >= _size );
        _size -= size;
    }

    //-----------------------------------------------------------
    inline void PopToMarker( const size_t sizeMarker ) override
    {
        ASSERT( sizeMarker <= _size );
        _size = sizeMarker;
    }

private:
    byte*  _buffer;
    size_t _capacity;   // Stack capacity
    size_t _size = 0;   // Current allocated size/stack size
};