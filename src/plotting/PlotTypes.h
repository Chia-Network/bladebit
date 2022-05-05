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
};
static_assert( sizeof( Pair ) == 8, "Invalid Pair struct." );