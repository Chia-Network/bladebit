#pragma once


template<typename T>
struct Span
{
    // #TODO: Make these immutable
    T*     values;
    size_t length;

    inline Span(){}

    inline Span( T* values, size_t length )
        : values( values )
        , length( length )
    {}

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

    inline Span<T> Slice( const uint32 index ) const { return Slice( (size_t)index ); }
    inline Span<T> Slice( const int64 index ) const { ASSERT( index >= 0); return Slice( (size_t)index ); }
    inline Span<T> Slice( const int32 index ) const { ASSERT( index >= 0); return Slice( (size_t)index ); }
};

typedef Span<uint8_t> ByteSpan;
