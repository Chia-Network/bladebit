#pragma once
#include "threading/AutoResetSignal.h"

class Fence
{
public:
    Fence();
    ~Fence();

    // inline uint32 Value() const { return _value.load( std::memory_order_acquire ); }
    inline uint32 Value() const { return _value; }

    // Should be only called by a single producer (ie. only 1 thread).
    void Signal();

    void Signal( uint32 value );

    // Should only be used when you know that the producer thread will not call Signal anymore.
    void Reset( uint32 value = 0 );

    // Wait until the fence is signalled with any value
    void Wait();

    // Wait until the fence is signalled with any value
    void Wait( Duration& accumulator );

    // Wait until the fence reaches or passes the specified value
    void Wait( uint32 value );

    // Wait until the fence reaches or passes the specified value
    void Wait( uint32 value, Duration& accumulator );

private:
    // std::atomic<uint32> _value;
    // #NOTE: Don't think we need atomic, since memory ordering ought to be enforced by the mutex.
    volatile uint32     _value = 0;
    AutoResetSignal     _signal;

};

