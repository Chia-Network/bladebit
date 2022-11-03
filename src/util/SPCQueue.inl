#pragma once
#include "util/Log.h"
#include "util/Util.h"


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
    return _committedCount.load( std::memory_order_relaxed );
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
    //ASSERT( _pendingCount );
    if( _pendingCount < 1 )
        return;

 
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
    bbmemcpy_t( values, _buffer + readPos, (size_t)copyCount );

    // We might have to do 2 copies if we're wrapping around the end of the buffer
    const int remainder = count - copyCount;
    if( remainder > 0 )
    {
        #if _DEBUG
        Log::Debug( "[0x%p] Wrap", this );
        #endif
        bbmemcpy_t( values + copyCount, _buffer, (size_t)remainder );
    }

    // Publish to the producer thread that we've consumed entries
    while( !_committedCount.compare_exchange_weak( curCount, curCount - count,
                                                   std::memory_order_release,
                                                   std::memory_order_relaxed ) );

    return count;
}




///
/// Dynamic
///

//-----------------------------------------------------------
template<typename T, size_t _growSize>
GrowableSPCQueue<T,_growSize>::GrowableSPCQueue(size_t capacity )
        : _producerState( _states )
        , _consumerState( _states )
{
    FatalIf( capacity > (size_t)std::numeric_limits<int>::max(),
             "GrowableSPCQueue capacity cannot exceed %d.",
             std::numeric_limits<int>::max() );

    memset( _states, 0, sizeof( _states ) );

    if( capacity > 0 )
    {
        _producerState->buffer = bbcalloc<T>( capacity );
    }

    _producerState->capacity = (int)capacity;
}

//-----------------------------------------------------------
template<typename T, size_t _growSize>
GrowableSPCQueue<T,_growSize>::GrowableSPCQueue()
        : GrowableSPCQueue(0 )
{}

//-----------------------------------------------------------
template<typename T, size_t _growSize>
GrowableSPCQueue<T,_growSize>::~GrowableSPCQueue()
{
    if( _producerState->buffer )
        free( _producerState->buffer );

    if( _consumerState->buffer != _producerState->buffer )
        free( _consumerState->buffer );
}

//-----------------------------------------------------------
template<typename T, size_t _growSize>
bool GrowableSPCQueue<T,_growSize>::Write(T*& outValue )
{
    const int count = _pendingCount + _producerState->committedCount.load( std::memory_order_acquire );
    ASSERT( count <= _producerState->capacity );

    if( count < _producerState->capacity )
    {
        outValue = &_producerState->buffer[_writePosition];

        ++_writePosition %= _producerState->capacity;
        _pendingCount++;

        return true;
    }

    // We ran out of buffer space, we'll have to create a new one,
    // adding a signal to the consumer thread to discard the current buffer
    // and use the new one once it has finished reading from it.

    if( _growSize == 0 )
        return false;

    // We only support double-buffering, that is one temporary buffer being
    // used by the consumer thread while it switches to the new one, so if
    // the consumer is still using an old buffer, we cannot allocate a new one.
    ConsumerState* newState = _newState.load( std::memory_order_acquire );
    if( newState != nullptr )
    {
        // #NOTE: Although we could technically still resize in this case if we manage to
        //        swap with null and acquire the _newState, we ought not to because it
        //        can cause deadlocks. The reason being that the consumer thread might be
        //        accompanied by a signal mechanism whenever the producer writes to the queue.
        //        In the case where the producer acquires the new state and swaps it for null
        //        during this resize step, and the consumer is signalled at the same time,
        //        the consumer thread may find no elements to consume with the consumer state.
        //        Which could be unexpected behavior for the user if they had signalled after committing.
        return false;
    }

    const size_t currentCapacity = _producerState->capacity;

    size_t newCapacity = currentCapacity + _growSize;

    const size_t MAX_CAPACITY = std::numeric_limits<int>::max();

    // Overflowed?
    if( newCapacity < currentCapacity )
        newCapacity = MAX_CAPACITY;

    // Were we already maxed-out?
    if( newCapacity == currentCapacity )
        return false;

    // Re-allocate
    T* newBuffer = bbcalloc<T>( newCapacity );

    // Set our new state
    _producerState = _states + _nextState;
    _nextState     = (++_nextState) % 2;

    _producerState->buffer         = newBuffer;
    _producerState->capacity       = (int)newCapacity;
    _producerState->committedCount = 0;

    _oldPendingCount = _pendingCount;
    _pendingCount    = 0;

    // The new state will not be published to the consumer thread until
    // the next commit call. This is because we may have pending
    // elements still not committed. If we change the state now, those
    // elements may end up as being pending in the new state, which is false.
    _pendingState = true;

    // Actually grab the new entry buffer now
    return Write( outValue );
}

//-----------------------------------------------------------
template<typename T, size_t _growSize>
void GrowableSPCQueue<T,_growSize>::Commit()
{
    if( _pendingCount < 1 && _oldPendingCount < 1 )
        return;

    // If we resized to a new state, we may have pending counts from that old state
    if( _pendingState )
    {
        if( _oldPendingCount > 0 )
        {
            int committed = _consumerState->committedCount.load( std::memory_order_acquire );
            ASSERT( committed < _consumerState->capacity );

            // Publish entries to the consumer thread
            while( !_consumerState->committedCount.compare_exchange_weak( committed, committed + _oldPendingCount,
                                                                          std::memory_order_release,
                                                                          std::memory_order_relaxed ) );

            _oldPendingCount = 0;
        }

        ASSERT( _newState == nullptr );
        _newState.store( _producerState, std::memory_order_release );
        _pendingState = false;
    }

    int committed = _producerState->committedCount.load( std::memory_order_acquire );
    ASSERT( committed < _producerState->capacity );

    // Publish entries to the consumer thread
    while( !_producerState->committedCount.compare_exchange_weak( committed, committed + _pendingCount,
                                                                  std::memory_order_release,
                                                                  std::memory_order_relaxed ) );

    _pendingCount = 0;
}

//-----------------------------------------------------------
template<typename T, size_t _growSize>
bool GrowableSPCQueue<T, _growSize>::Enqueue(const T& value )
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
template<typename T, size_t _growSize>
int GrowableSPCQueue<T, _growSize>::Dequeue( T* values, int capacity )
{
    ASSERT( values );
    ASSERT( capacity > 0 );
    ASSERT( _consumerState );

    int curCount = _consumerState->committedCount.load( std::memory_order_acquire );
    int count    = std::min( capacity, curCount );

    if( count > 0 )
    {
        const size_t bufferCapacity = _consumerState->capacity;
        T* consumerBuffer = _consumerState->buffer;

        const int readPos = _readPosition;
        _readPosition = ( readPos + count ) % (int)bufferCapacity;

        const int copyCount = std::min( count, (int)bufferCapacity - readPos );
        bbmemcpy_t( values, consumerBuffer + readPos, (size_t)count );

        // We might have to do 2 copies if we're wrapping around the end of the buffer
        const int remainder = count - copyCount;
        if( remainder > 0 )
        {
            #if _DEBUG
                Log::Debug( "[0x%p] Wrap", this );
            #endif

            bbmemcpy_t( values + copyCount, consumerBuffer, (size_t)remainder );
        }

        // Publish to the producer thread that we've consumed entries
        while( !_consumerState->committedCount.compare_exchange_weak( curCount, curCount - count,
                                                                      std::memory_order_release,
                                                                      std::memory_order_relaxed ) );
    }

    // Check if we have a pending state change, if so apply it
    // if we've consumed all of our current state's pending entries.
    ConsumerState* newState = nullptr;
    const int committedRemaining = curCount - count;
    if( committedRemaining == 0 && ( newState = _newState.load( std::memory_order_acquire ) ) != nullptr )
    {
        // Delete the current state's buffer
        free( _consumerState->buffer );
        _consumerState->buffer = nullptr;

        // Publish to the producer that we've changed states
        _newState.store( nullptr, std::memory_order_release );

        // Change to the new state and check if we can consume some more entries
        _consumerState = newState;
        ASSERT( newState );

        if( capacity > count )
        {
            return count + this->Dequeue( values + count, capacity - count );
        }
    }

    ASSERT( count >= 0 );
    return count;
}

