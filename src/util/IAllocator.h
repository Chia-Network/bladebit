#pragma once
#include "util/Util.h"

class IAllocator
{
public:
    virtual ~IAllocator() {}

    virtual void* Alloc( const size_t size, const size_t alignment ) = 0;

    inline virtual void Free( void* ptr ) { (void)ptr; }

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

    //-----------------------------------------------------------
    inline void TryFree( void* ptr )
    {
        if( ptr )
            Free( ptr );
    }

    //-----------------------------------------------------------
    template<typename T>
    inline void SafeFree( T*& ptr )
    {
        if( ptr )
        {
            Free( ptr );
            ptr = nullptr;
        }
    }
};

class GlobalAllocator : public IAllocator
{
public:
    inline void* Alloc( const size_t size, const size_t alignment ) override
    {
        // Ignore alignment
        (void)alignment;
        return malloc( size );
    }
};

// class ProxyAllocator : public IAllocator
// {
//     IAllocator& _allocator;

// public:
//     ProxyAllocator() = delete;
//     inline ProxyAllocator( IAllocator& allocator )
//         , _allocator( allocator )
//     {}

//     inline ProxyAllocator( const ProxyAllocator& other )
//         : _allocator( other._allocator )
//     {}

//     inline void* Alloc( const size_t size, const size_t alignment ) override
//     {
//         return _allocator.Alloc( size, alignment );
//     }
// };
