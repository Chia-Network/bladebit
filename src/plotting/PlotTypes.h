#pragma once
#include "threading/AutoResetSignal.h"

struct Pairs
{
    uint32* left ;
    uint16* right;
};

struct Pair
{
    uint32 left;
    uint32 right;

    inline Pair& AddOffset( const uint32 offset )
    {
        left  += offset;
        right += offset;

        return *this;
    }

    inline Pair& SubstractOffset( const uint32 offset )
    {
        left  -= offset;
        right -= offset;

        return *this;
    }
};
static_assert( sizeof( Pair ) == 8, "Invalid Pair struct." );