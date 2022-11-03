#pragma once

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


enum class PlotTable
{
    Table1 = 0,
    Table2,
    Table3,
    Table4,
    Table5,
    Table6,
    Table7,
    C1,
    C2,
    C3,
}; ImplementArithmeticOps( PlotTable );

