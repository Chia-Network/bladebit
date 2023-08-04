#pragma once

#include "plotting/PlotTools.h"

// #NOTE: For now it is implemented in PlotValidator.cpp
bool ValidateFullProof( const uint32 k, const byte plotId[BB_PLOT_ID_LEN], uint64 fullProofXs[PROOF_X_COUNT], uint64& outF7 );

namespace PlotValidation
{
    inline static bool ValidateFullProof( const uint32 k, const byte plotId[BB_PLOT_ID_LEN], uint64 fullProofXs[PROOF_X_COUNT], uint64& outF7 )
    {
        return ::ValidateFullProof( k, plotId, fullProofXs, outF7 );
    }
    
    static uint64 BytesToUInt64( const byte bytes[8] );
    static uint64 SliceUInt64FromBits( const byte* bytes, uint32 bitOffset, uint32 bitCount );

    static bool FxMatch( uint64 yL, uint64 yR );

    static void FxGen( const TableId table, const uint32 k, 
                       const uint64 y, const MetaBits& metaL, const MetaBits& metaR,
                       uint64& outY, MetaBits& outMeta );
};