#include "threading/AutoResetSignal.h"

class Fence
{
public:
    Fence();
    ~Fence();

    inline uint32 Value() const { return _value.load( std::memory_order_acquire ); }

    // Should be only called by a single producer (ie. only 1 thread).
    void Signal();

    void Signal( uint32 value );

    // Should only be used when you know that the producer thread will not call Signal anymore.
    void Reset( uint32 value = 0 );

    void WaitForAnyValue();

    void WaitForValue( uint32 value );

private:
    std::atomic<uint32> _value;
    AutoResetSignal     _signal;

};

