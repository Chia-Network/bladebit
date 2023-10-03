#pragma once

#include "StackAllocator.h"
#include "util/Util.h"


class VirtualAllocator : public IAllocator
{
public:
    inline void* Alloc( const size_t size, const size_t alignment ) override
    {
        (void)alignment;

        const size_t allocSize = PageAlign( size );
        _size += allocSize;
        
        return bbvirtallocbounded<byte>( allocSize );
    }

    inline void* TryAlloc( const size_t size )
    {
        const size_t allocSize = PageAlign( size );

        void* ptr = bb_try_virt_alloc( allocSize );
        if( !ptr )
            _failureCount++;
        else
            _size += allocSize;

        return ptr;
    }

    template<typename T>
    inline T* TryCAlloc( const size_t count )
    {
        return TryAlloc( sizeof( T ) * count );
    }

    template<typename T>
    inline Span<T> TryCAllocSpan( const size_t count )
    {
        T* ptr = TryCAlloc<T>( count );
        return Span<T>( ptr, ptr ? count : 0 );
    }


    inline void* AllocBounded( const size_t size )
    {
        const size_t allocSize = PageAlign( size );
        _size += allocSize + GetPageSize() * 2;

        return bbvirtallocbounded( size );
    }

    inline void* TryAllocBounded( const size_t size )
    {
        const size_t allocSize = PageAlign( size );

        void* ptr = bb_try_virt_alloc_bounded( size );
        if( !ptr )
            _failureCount++;
        else
            _size += allocSize + GetPageSize() * 2;

        return ptr;
    }

    template<typename T>
    inline T* TryCAllocBounded( const size_t count )
    {
        return (T*)TryAllocBounded( sizeof( T ) * count );
    }

    template<typename T>
    inline Span<T> TryCAllocBoundedSpan( const size_t count )
    {
        T* ptr = TryCAllocBounded<T>( count );
        return Span<T>( ptr, ptr ? count : 0 );
    }


    inline size_t GetPageSize() const { return SysHost::GetPageSize(); }

    inline size_t PageAlign( const size_t size ) const
    {
        return RoundUpToNextBoundaryT<size_t>( size, GetPageSize() );
    }

    inline size_t AllocSize() const { return _size; }

    inline size_t FailCount() const { return _failureCount; }

private:
    size_t _size         = 0;
    size_t _failureCount = 0;
};

