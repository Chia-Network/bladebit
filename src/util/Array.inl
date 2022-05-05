#pragma once
#include "Array.h"
#include "util/Util.h"


template<typename T>
inline Array<T>::Array() : Array( size_t( 0 ) )
{
}

template<typename T>
inline Array<T>::Array( size_t capacity )
    : _elements( nullptr  )
    , _capacity( capacity )
    , _length  ( 0        )
{
    if( _capacity )
        _elements = bbcalloc<T>( capacity );
}

template<typename T>
inline Array<T>::~Array()
{
    if( _elements )
        free( _elements );
}

template<typename T>
inline T& Array<T>::Push()
{
    CheckCapacity( 1u );

    T* value = &_elements[_length++];
    
    value = new ( (void*)value ) T();
    return *value;
}

template<typename T>
inline T& Array<T>::Push( const T& value )
{
    CheckCapacity( 1u );

    T* e = &_elements[_length++];
    
    *e = value;
    return *e;
}

template<typename T>
inline void Array<T>::Pop()
{
    ASSERT( _length );
    
    _elements[--_length].~T();
}

template<typename T>
inline T& Array<T>::Insert( const T& value, size_t index )
{
    ASSERT( index <= _length );
    
    CheckCapacity( 1u );

    // Append to the end of the array?
    if( index == _length )
        return Push( value );

    // Move-up remainder elements
    const size_t remainder = _length - index;
    _length++;

    bbmemcpy_t( _elements + index + 1, _elements + index, remainder );

    T* e = &_elements[index];
    *e = value;

    return *e;
}

template<typename T>
inline T& Array<T>::Insert( size_t index )
{
    ASSERT( index <= _length );
    
    CheckCapacity( 1u );

    // Append to the end of the array?
    if( index == _length )
        return Push();

    const size_t remainder = _length - index;
    _length++;

    bbmemcpy_t(  _elements + index + 1, _elements + index, remainder );

    T* value = new ( (void*)&_elements[index] ) T();

    return *value;
}

template<typename T>
inline void Array<T>::Remove( size_t index )
{
    ASSERT( index < _length );

    if( index == _length - 1 )
    {
        Pop();
        return;
    }
    
    _elements[index].~T();

    --_length;
    const size_t remainder = _length - index;

    bbmemcpy_t( _elements + index, _elements + index + 1, remainder );
}

template<typename T>
inline void Array<T>::UnorderedRemove( size_t index )
{
    ASSERT( index < _length );

    if( index == _length - 1 )
    {
        Pop();
        return;
    }

    _elements[index].~T();
    _elements[index] = _elements[--_length];
}

template<typename T>
inline size_t Array<T>::Length() const
{
    return _length;
}

template<typename T>
inline T* Array<T>::Ptr() const
{
    return _elements;
}

template<typename T>
inline void Array<T>::CheckCapacity( size_t count )
{
    ASSERT( _length < 9223372036854775807 );
    
    const size_t newLength = _length + count;
    
    // Do we need to resize our buffer?
    if( newLength > _capacity )
    {
        size_t capacityGrow = 8;

        if( _capacity )
            capacityGrow = _capacity < 1024u ? _capacity : 1024u;
        
        size_t newCapacity = _capacity + capacityGrow;
        if( newCapacity < _capacity )
        {
            if( _capacity == 9223372036854775807 )
                Fatal( "Array exceeded maximum size." );

            newCapacity = 9223372036854775807;
        }

        _elements = bbcrealloc( _elements, newCapacity );
        _capacity = newCapacity;
    }
}
