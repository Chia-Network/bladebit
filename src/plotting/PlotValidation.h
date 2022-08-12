#pragma once
#include "util/BitView.h"
#include "ChiaConsts.h"
#include "plotting/PlotTools.h"

#define PROOF_X_COUNT       64
#define MAX_K_SIZE          48
#define MAX_META_MULTIPLIER 4
#define MAX_Y_BIT_SIZE      ( MAX_K_SIZE + kExtraBits )
#define MAX_META_BIT_SIZE   ( MAX_K_SIZE * MAX_META_MULTIPLIER )
#define MAX_FX_BIT_SIZE     ( MAX_Y_BIT_SIZE + MAX_META_BIT_SIZE + MAX_META_BIT_SIZE )

typedef Bits<MAX_Y_BIT_SIZE>    YBits;
typedef Bits<MAX_META_BIT_SIZE> MetaBits;
typedef Bits<MAX_FX_BIT_SIZE>   FxBits;


class PlotValidation
{
    static bool ValidateFullProof( const uint32 k, const byte plotId[BB_PLOT_ID_LEN], uint64 fullProofXs[PROOF_X_COUNT], uint64& outF7 );
    
    static uint64 BytesToUInt64( const byte bytes[8] );
    static uint64 SliceUInt64FromBits( const byte* bytes, uint32 bitOffset, uint32 bitCount );

    static bool FxMatch( uint64 yL, uint64 yR );

    static void FxGen( const TableId table, const uint32 k, 
                       const uint64 y, const MetaBits& metaL, const MetaBits& metaR,
                       uint64& outY, MetaBits& outMeta );
};