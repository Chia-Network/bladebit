#pragma once

template<typename T>
struct View
{
    T*     _ptr;
    size_t _count;

    //-----------------------------------------------------------
    inline View( T* ptr, size_t count )
        : _ptr  ( ptr   )
        , _count( count )
    {}

    //-----------------------------------------------------------
    inline View( T* ptr )
        : _ptr  ( ptr )
        , _count( 0   )
    {}

    
    //-----------------------------------------------------------
    inline T* operator-> () const { return this->_ptr; }

    //-----------------------------------------------------------
    inline operator T* ( ) const { return this->_ptr; }

    // Returns true if the object ptr is not null
    inline operator bool() const { return this->_ptr != nullptr; }


};