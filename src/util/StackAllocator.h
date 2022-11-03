#pragma once

class IAllocator
{
public:
    virtual ~IAllocator() {}

    virtual void* Alloc( const size_t size, const size_t alignment ) = 0;

    //-----------------------------------------------------------
    template<typename T>
    inline T* AllocT( const size_t size, size_t alignment = alignof( T ) )
    {
        return reinterpret_cast<T*>( Alloc( size, alignment ) );
    }

    //-----------------------------------------------------------
    template<typename T>
    inline T* CAlloc( const size_t count, size_t alignment = alignof( T ) )
    {
        const size_t allocSize = sizeof( T ) * count;
        ASSERT( allocSize >= count );
        
        return AllocT<T>( allocSize, alignment );
    }

    //-----------------------------------------------------------
    template<typename T>
    inline Span<T> CAllocSpan( const size_t count, size_t alignment = alignof( T ) )
    {
        return Span<T>( this->CAlloc<T>( count, alignment ), count );
    }

    //-----------------------------------------------------------
    inline void* CAlloc( const size_t count, const size_t size, const size_t alignment )
    {
        const size_t paddedSize = RoundUpToNextBoundaryT( size, alignment );
        
        return Alloc( paddedSize * count, alignment );
    }
};

class DummyAllocator : public IAllocator
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
    inline size_t Size() const { return _size; }

private:
    size_t _size = 0;
};


// Fixed-capacity stack allocator
class StackAllocator : public IAllocator
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
    inline size_t Size() const
    {
        return _size;
    }

    //-----------------------------------------------------------
    inline size_t Remainder() const
    {
        return _capacity - _size;
    }

    //-----------------------------------------------------------
    inline byte* Top() const
    {
        return _buffer + _size;
    }

    //-----------------------------------------------------------
    inline void Pop( const size_t size )
    {
        ASSERT( size >= _size );
        _size -= size;
    }

    //-----------------------------------------------------------
    inline void PopToMarker( const size_t sizeMarker )
    {
        ASSERT( sizeMarker <= _size );
        _size = sizeMarker;
    }

private:
    byte*  _buffer;
    size_t _capacity;   // Stack capacity
    size_t _size = 0;   // Current allocated size/stack size
};