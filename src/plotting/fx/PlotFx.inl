#pragma once

#include "plotting/Tables.h"
#include "plotting/PlotTypes.h"

#define B3Round( intputByteSize ) \
    uint32 state[16] = {                      \
        0x6A09E667UL,   /*IV full*/           \
        0xBB67AE85UL,                         \
        0x3C6EF372UL,                         \
        0xA54FF53AUL,                         \
        0x510E527FUL,                         \
        0x9B05688CUL,                         \
        0x1F83D9ABUL,                         \
        0x5BE0CD19UL,                         \
        0x6A09E667UL,   /*IV 0-4*/            \
        0xBB67AE85UL,                         \
        0x3C6EF372UL,                         \
        0xA54FF53AUL,                         \
        0,               /*count lo*/         \
        0,               /*count hi*/         \
        (intputByteSize),/*buffer length*/    \
        11              /*flags. Always 11*/  \
    };                                        \
                                              \
    round_fn( state, (uint32*)&input[0], 0 ); \
    round_fn( state, (uint32*)&input[0], 1 ); \
    round_fn( state, (uint32*)&input[0], 2 ); \
    round_fn( state, (uint32*)&input[0], 3 ); \
    round_fn( state, (uint32*)&input[0], 4 ); \
    round_fn( state, (uint32*)&input[0], 5 ); \
    round_fn( state, (uint32*)&input[0], 6 ); 

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"

FORCE_INLINE uint32_t rotr32( uint32_t w, uint32_t c )
{
    return ( w >> c ) | ( w << ( 32 - c ) );
}

FORCE_INLINE void g( uint32_t* state, size_t a, size_t b, size_t c, size_t d,
                     uint32_t x, uint32_t y )
{
    state[a] = state[a] + state[b] + x;
    state[d] = rotr32( state[d] ^ state[a], 16 );
    state[c] = state[c] + state[d];
    state[b] = rotr32( state[b] ^ state[c], 12 );
    state[a] = state[a] + state[b] + y;
    state[d] = rotr32( state[d] ^ state[a], 8 );
    state[c] = state[c] + state[d];
    state[b] = rotr32( state[b] ^ state[c], 7 );
}

FORCE_INLINE void round_fn( uint32_t state[16], const uint32_t* msg, size_t round )
{
    static const uint8_t MSG_SCHEDULE[7][16] = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
        {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
        {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
        {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
        {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
        {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
    };
    // Select the message schedule based on the round.
    const uint8_t* schedule = MSG_SCHEDULE[round];

    // Mix the columns.
    g( state, 0, 4, 8 , 12, msg[schedule[0]], msg[schedule[1]] );
    g( state, 1, 5, 9 , 13, msg[schedule[2]], msg[schedule[3]] );
    g( state, 2, 6, 10, 14, msg[schedule[4]], msg[schedule[5]] );
    g( state, 3, 7, 11, 15, msg[schedule[6]], msg[schedule[7]] );

    // Mix the rows.
    g( state, 0, 5, 10, 15, msg[schedule[8]] , msg[schedule[9]] );
    g( state, 1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]] );
    g( state, 2, 7, 8 , 13, msg[schedule[12]], msg[schedule[13]] );
    g( state, 3, 4, 9 , 14, msg[schedule[14]], msg[schedule[15]] );
}


//-----------------------------------------------------------
template<TableId rTable>
FORCE_INLINE typename K32TYOut<rTable>::Type FxK32HashOne( 
    const uint64                           y,
    const typename K32MetaType<rTable>::In metaL,
    const typename K32MetaType<rTable>::In metaR,
    typename K32MetaType<rTable>::Out&     outMeta
    )
{
    constexpr size_t MetaInMulti  = TableMetaIn <rTable>::Multiplier;
    constexpr size_t MetaOutMulti = TableMetaOut<rTable>::Multiplier;

    using TYOut = typename K32TYOut<rTable>::Type;

    const uint32 k           = 32;
    const uint32 metaSize    = k * MetaInMulti;
    const uint32 metaSizeLR  = metaSize * 2;
    const uint32 inputSize   = CDiv( 38 + metaSizeLR, 8 );
    const uint32 shiftBits   = MetaOutMulti == 0 ? 0 : kExtraBits;      // Table 7 does not output with kExtraBits
    const uint32 ySize       = k + kExtraBits;                          // = 38
    const uint32 yShift      = 64 - (k + shiftBits);                    // = 26 or 32

    uint64 input [8];
    uint64 output[4];
        
    if constexpr( MetaInMulti == 1 )
    {
        const uint64 l = metaL;
        const uint64 r = metaR;
        
        const uint64 i0 = y << 26 | l >> 6;
        const uint64 i1 = l << 58 | r << 26;

        input[0] = Swap64( i0 );
        input[1] = Swap64( i1 );
        input[2] = 0;
        input[3] = 0;
        input[4] = 0;
        input[5] = 0;
        input[6] = 0;
        input[7] = 0;

        if constexpr( MetaOutMulti == 2 )
            outMeta = l << 32 | r;
    }
    else if constexpr ( MetaInMulti == 2 )
    {
        input[0] = Swap64( y     << 26 | metaL >> 38 );
        input[1] = Swap64( metaL << 26 | metaR >> 38 );
        input[2] = Swap64( metaR << 26 );
        input[3] = 0;
        input[4] = 0;
        input[5] = 0;
        input[6] = 0;
        input[7] = 0;

        if constexpr ( MetaOutMulti == 4 )
        {
            outMeta.m0 = metaL;
            outMeta.m1 = metaR;
        }
    }
    else if constexpr ( MetaInMulti == 3 )
    {
        const uint64 l0 = metaL.m0;
        const uint64 l1 = metaL.m1 & 0xFFFFFFFF;
        const uint64 r0 = metaR.m0;
        const uint64 r1 = metaR.m1 & 0xFFFFFFFF;
        
        input[0] = Swap64( y  << 26 | l0 >> 38 );
        input[1] = Swap64( l0 << 26 | l1 >> 6  );
        input[2] = Swap64( l1 << 58 | r0 >> 6  );
        input[3] = Swap64( r0 << 58 | r1 << 26 );
        input[4] = 0;
        input[5] = 0;
        input[6] = 0;
        input[7] = 0;
    }
    else if constexpr ( MetaInMulti == 4 )
    {
        input[0] = Swap64( y        << 26 | metaL.m0 >> 38 );
        input[1] = Swap64( metaL.m0 << 26 | metaL.m1 >> 38 );
        input[2] = Swap64( metaL.m1 << 26 | metaR.m0 >> 38 );
        input[3] = Swap64( metaR.m0 << 26 | metaR.m1 >> 38 );
        input[4] = Swap64( metaR.m1 << 26 );
        input[5] = 0;
        input[6] = 0;
        input[7] = 0;
    }

    B3Round( inputSize );

    uint32* out = (uint32*)output;
    out[0] = state[0] ^ state[8];
    out[1] = state[1] ^ state[9];

    const uint64 oy = Swap64( *output ) >> yShift;

    // Save output metadata
    if constexpr ( MetaOutMulti == 2 && MetaInMulti == 3 )
    {
        out[2] = state[2] ^ state[10];
        out[3] = state[3] ^ state[11];

        const uint64 h0 = Swap64( output[0] );
        const uint64 h1 = Swap64( output[1] );

        outMeta = h0 << ySize | h1 >> 26;
    }
    else if constexpr ( MetaOutMulti == 3 )
    {
        out[2] = state[2] ^ state[10];
        out[3] = state[3] ^ state[11];
        out[4] = state[4] ^ state[12];
        out[5] = state[5] ^ state[13];

        const uint64 h0 = Swap64( output[0] );
        const uint64 h1 = Swap64( output[1] );
        const uint64 h2 = Swap64( output[2] );

        outMeta.m0 = h0 << ySize | h1 >> 26;
        outMeta.m1 = ((h1 << 6) & 0xFFFFFFC0) | h2 >> 58;
    }
    else if constexpr ( MetaOutMulti == 4 && MetaInMulti != 2 )
    {
        out[2] = state[2] ^ state[10];
        out[3] = state[3] ^ state[11];
        out[4] = state[4] ^ state[12];
        out[5] = state[5] ^ state[13];

        const uint64 h0 = Swap64( output[0] );
        const uint64 h1 = Swap64( output[1] );
        const uint64 h2 = Swap64( output[2] );

        outMeta.m0 = h0 << ySize | h1 >> 26;
        outMeta.m1 = h1 << 38    | h2 >> 26;
    }

    return (TYOut)oy;
}

#pragma GCC diagnostic pop

#undef B3Round

