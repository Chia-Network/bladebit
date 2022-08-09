#pragma once
#include <string.h>

template<typename T>
struct Span
{
    // #TODO: Make these immutable
    T*     values;
    size_t length;

    inline Span()
        : values( nullptr )
        , length( 0 )
    {}

    inline Span( T* values, size_t length )
        : values( values )
        , length( length )
    {}

    inline void SetTo( T* values, const size_t length )
    {
        this->values = values;
        this->length = length;
    }

    inline size_t Length() const { return length; }

    inline T* Ptr() const { return values; }

    inline T& operator[]( unsigned int index ) const
    { 
        ASSERT( index < length );
        return this->values[index]; 
    }

    inline T& operator[]( size_t index ) const
    { 
        ASSERT( index < length );
        return this->values[index]; 
    }

    inline T& operator[]( int64 index ) const
    {
        ASSERT( index < (int64)length );
        return this->values[index];
    }

// size_t not the same as uint64 on clang.
#ifdef __clang__
    inline T& operator[]( uint64 index ) const
    {
        ASSERT( index < length );
        return this->values[index];
    }
#endif

    inline T& operator[]( int index ) const
    { 
        ASSERT( index > -1 && (size_t)index < length );
        return this->values[index];
    }

    inline Span<T> Slice( const size_t index, const size_t length ) const
    {
        ASSERT( index <= this->length );
        ASSERT( index + length >= index );
        ASSERT( index + length <= this->length );

        return Span<T>( this->values + index, length );
    }

    inline Span<T> Slice( const size_t index ) const
    {
        ASSERT( index <= length );
        return Slice( index, length - index );
    }

    inline Span<T> Slice() const
    {
        return Slice( 0, length );
    }

    inline Span<T> SliceSize( const size_t size ) const
    {
        ASSERT( size <= length );
        return Slice( 0, size );
    }

#ifdef __clang__
    inline Span<T> Slice( const uint64 index ) const { return Slice( (size_t)index ); }
#endif
    inline Span<T> Slice( const uint32 index ) const { return Slice( (size_t)index ); }
    inline Span<T> Slice( const int64 index ) const { ASSERT( index >= 0); return Slice( (size_t)index ); }
    inline Span<T> Slice( const int32 index ) const { ASSERT( index >= 0); return Slice( (size_t)index ); }

    inline void CopyTo( Span<T> other, const size_t size ) const
    {
        ASSERT( length >= size );
        ASSERT( other.length >= size );

        memcpy( other.values, values, size * sizeof( T ) );
    }

    inline void CopyTo( Span<T> other ) const
    {
        CopyTo( other, length );
    }

    inline void ZeroOutElements()
    {
        if( length < 1 )
            return;

        memset( values, 0, sizeof( T ) * length );
    }

    inline bool EqualElements( const Span<T>& other, const size_t count ) const
    {
        ASSERT( count <= other.length );
        ASSERT( length >= count );
        
        return memcmp( values, other.values, count * sizeof( T ) ) == 0;
    }

    inline bool EqualElements( const Span<T>& other ) const
    {
        if( other.length != length )
            return false;

        return EqualElements( other, length );
    }

    template<typename TCast>
    inline Span<TCast> As() const
    {
        const size_t size         = sizeof( T ) * length;
        const size_t targetLength = size / sizeof( TCast );

        return Span<TCast>( reinterpret_cast<TCast*>( values ), targetLength );
    }
};

typedef Span<uint8_t> ByteSpan;
