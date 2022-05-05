#include "Fence.h"

//-----------------------------------------------------------
Fence::Fence()
    : _value ( 0 )
    , _signal()
{
}

//-----------------------------------------------------------
Fence::~Fence()
{
}

//-----------------------------------------------------------
void Fence::Signal()
{
    _value ++;
    _signal.Signal();
}

//-----------------------------------------------------------
void Fence::Signal( uint32 value )
{
    // _value.store( value, std::memory_order_release );
    _value = value;
    _signal.Signal();
}

//-----------------------------------------------------------
void Fence::Reset( uint32 value )
{
    // _value.store( value, std::memory_order_release );
    _value = value;
    _signal.Reset();
}

//-----------------------------------------------------------
void Fence::Wait()
{
    _signal.Wait();
}

//-----------------------------------------------------------
void Fence::Wait( Duration& accumulator )
{
    const auto startTime = TimerBegin();
    _signal.Wait();
    accumulator += TimerEndTicks( startTime );
}

//-----------------------------------------------------------
void Fence::Wait( uint32 value )
{
    // while( _value.load( std::memory_order_relaxed ) < value )
    while( _value < value )
        _signal.Wait();
}

//-----------------------------------------------------------
void Fence::Wait( uint32 value, Duration& accumulator )
{
    while( _value < value )
    {
        const auto startTime = TimerBegin();
        _signal.Wait();
        accumulator += TimerEndTicks( startTime );
    }
}

