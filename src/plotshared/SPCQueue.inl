#pragma once

//-----------------------------------------------------------
template<typename T, int Capacity>
SPCQueue<T, Capacity>::SPCQueue()
{}

//-----------------------------------------------------------
template<typename T, int Capacity>
SPCQueue<T, Capacity>::~SPCQueue()
{}

//-----------------------------------------------------------
template<typename T, int Capacity>
int SPCQueue<T, Capacity>::Count() const
{
    return _count.load( std::memory_order_relaxed );
}

//-----------------------------------------------------------
template<typename T, int Capacity>
bool SPCQueue<T, Capacity>::Enqueue( const T& value )
{
    T* entry;
    if( Write( entry ) )
    {
        *entry = value;
        Commit();
        return true;
    }

    return false;
}

//-----------------------------------------------------------
template<typename T, int Capacity>
bool SPCQueue<T, Capacity>::Write( T*& outValue )
{
    int count = _pendingCount + _committedCount.load( std::memory_order_acquire );
    ASSERT( count <= Capacity );

    if( count < Capacity )
    {
        outValue = &_buffer[_writePosition];

        ++_writePosition %= Capacity;
        _pendingCount++;

        return true;
    }

    return false;
}

//-----------------------------------------------------------
template<typename T, int Capacity>
void SPCQueue<T, Capacity>::Commit()
{
    ASSERT( _pendingCount );
//     if( _pendingCount < 1 )
//         return;

 
    int commited = _committedCount.load( std::memory_order_acquire );
    ASSERT( commited < Capacity );

    // Publish entries to the consumer thread
    while( !_committedCount.compare_exchange_weak( commited, commited + _pendingCount,
                                                   std::memory_order_release,
                                                   std::memory_order_relaxed ) );

    _pendingCount = 0;
}

//-----------------------------------------------------------
template<typename T, int Capacity>
int SPCQueue<T, Capacity>::Dequeue( T* values, int capacity )
{
    ASSERT( values );
    ASSERT( capacity > 0 );

    int curCount = _committedCount.load( std::memory_order_acquire );
    int count    = std::min( capacity, curCount );

    if( count < 1 )
        return 0;
    
    const int readPos = _readPosition;
    _readPosition = ( readPos + count ) % Capacity;

    const int copyCount = std::min( count, Capacity - readPos );
    bbmemcpy_t( values, _buffer + readPos, (size_t)count );

    // We might have to do 2 copies if we're wrapping around the end of the buffer
    const int remainder = count - copyCount;
    if( remainder > 0 )
    {
        Log::Debug( "[0x%p] Wrap", this );
        bbmemcpy_t( values + copyCount, _buffer, (size_t)remainder );
    }

    // Publish to the producer thread that we've consumed entries
    while( !_committedCount.compare_exchange_weak( curCount, curCount - count,
                                                   std::memory_order_release,
                                                   std::memory_order_relaxed ) );

    return count;
}


