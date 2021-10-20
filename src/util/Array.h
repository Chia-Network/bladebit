#pragma once

// Simply dynamic array without the stl overhead
template<typename T>
class Array
{
public:
    Array();
    Array( size_t capacity );
    ~Array();

    T& Push();
    T& Push( const T& value );

    void Pop();

    T& Insert( const T& value, size_t index );
    T& Insert( size_t index );

    void Remove( size_t index );

    // Remove an element, and if it was not the last one in
    // the array, the last element is placed in its index.
    void UnorderedRemove( size_t index );

    size_t Length() const;

    T* Ptr() const;

    inline T& operator[]( unsigned int index ) const
    {
        ASSERT( index < _length );
        return this->_elements[index];
    }

    inline T& operator[]( size_t index ) const
    {
        ASSERT( index < _length );
        return this->_elements[index];
    }
    inline T& operator[]( int index ) const
    {
        ASSERT( index > -1 && (size_t)index < _length );
        return this->_elements[index];
    }

private:
    inline void CheckCapacity( size_t count );

private:
    T*     _elements;
    size_t _capacity;
    size_t _length;
};

#include "Array.inl"

