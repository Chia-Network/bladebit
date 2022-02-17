#pragma once

// Fixed-capacity stack allocator
class StackAllocator
{
public:

    //-----------------------------------------------------------
    inline StackAllocator( void* buffer, size_t capacity )
        : _buffer  ( (byte*)buffer )
        , _capacity( capacity )
    {}

    //-----------------------------------------------------------
    inline void* Alloc( size_t size, size_t alignment )
    {
        // Start address must be aligned to the specified alignment
        const size_t padding = RoundUpToNextBoundaryT( _size, alignment );
        
        ASSERT( size > 0 );
        ASSERT( _capacity - (_size + padding) >= size );

        void* ptr = reinterpret_cast<void*>( _buffer + padding );
        _size += padding + size;

        return ptr;
    }

    //-----------------------------------------------------------
    template<typename T>
    inline T* AllocT( size_t size, size_t alignment = alignof( T ) )
    {
        return reinterpret_cast<T*>( Alloc( size, alignement ) );
    }

    //-----------------------------------------------------------
    template<typename T>
    inline T* CAlloc( size_t count, size_t alignment = alignof( T ) )
    {
        const size_t allocSize = sizeof( T ) * count;
        ASSERT( allocSize > count );
        
        return AllocT<T>( allocSize, alignment );
    }


private:
    byte*  _buffer;
    size_t _capacity;
    size_t _size = 0;
};