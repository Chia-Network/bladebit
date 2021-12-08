#pragma once
#include "threading/AutoResetSignal.h"

struct Pairs
{
    uint32* left ;
    uint16* right;
};


// Represents a double-buffered blob of data of the size of
// the block size of the device we're writing to * 2 (one for y one for x)
struct DoubleBuffer
{
    byte*           front;
    byte*           back;
    AutoResetSignal fence;

    inline DoubleBuffer()
    {
        // Has to be initially signaled, since the first swap doesn't need to wait.
        fence.Signal();
    }
    inline ~DoubleBuffer() {}

    inline void Flip()
    {
        fence.Wait();
        std::swap( front, back );
    }
};