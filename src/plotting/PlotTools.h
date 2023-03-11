#pragma once
#include "ChiaConsts.h"
#include "util/KeyTools.h"
#include "util/BitView.h"

#define PROOF_X_COUNT       64
#define MAX_K_SIZE          50
#define MAX_META_MULTIPLIER 4
#define MAX_Y_BIT_SIZE      ( MAX_K_SIZE + kExtraBits )
#define MAX_META_BIT_SIZE   ( MAX_K_SIZE * MAX_META_MULTIPLIER )
#define MAX_FX_BIT_SIZE     ( MAX_Y_BIT_SIZE + MAX_META_BIT_SIZE + MAX_META_BIT_SIZE )

typedef Bits<MAX_Y_BIT_SIZE>    YBits;
typedef Bits<MAX_META_BIT_SIZE> MetaBits;
typedef Bits<MAX_FX_BIT_SIZE>   FxBits;

typedef unsigned FSE_CTable;
typedef unsigned FSE_DTable;

struct PlotTools
{
    static void GenPlotFileName( const byte plotId[BB_PLOT_ID_LEN], char outPlotFileName[BB_COMPRESSED_PLOT_FILE_LEN_TMP], uint32 compressionLevel );
    static void PlotIdToString( const byte plotId[BB_PLOT_ID_LEN], char plotIdString[BB_PLOT_ID_HEX_LEN+1] );

    static bool PlotStringToId( const char plotIdString[BB_PLOT_ID_HEX_LEN+1], byte plotId[BB_PLOT_ID_LEN] );

    static bls::G1Element GeneratePlotPublicKey( const bls::G1Element& localPk, bls::G1Element& farmerPk, const bool includeTaproot );

    static void GeneratePlotIdAndMemo( 
        byte            plotId  [BB_PLOT_ID_LEN], 
        byte            plotMemo[BB_PLOT_MEMO_MAX_SIZE], 
        uint16&         outMemoSize,
        bls::G1Element& farmerPK,
        bls::G1Element* poolPK,
        PuzzleHash*     contractPuzzleHash
    );

    static FSE_CTable* GenFSECompressionTable( double rValue, size_t* outTableSize = nullptr );
    static FSE_DTable* GenFSEDecompressionTable( double rValue, size_t* outTableSize = nullptr );

    // static void PlotIdToStringTmp( const byte* plotId, const byte plotIdString[BB_PLOT_FILE_LEN_TMP] );

    // //-----------------------------------------------------------
    // static uint32_t CalculateLinePointSize(uint8_t k) { return Util::ByteAlign(2 * k) / 8; }

    // // This is the full size of the deltas section in a park. However, it will not be fully filled
    // static uint32_t CalculateMaxDeltasSize(uint8_t k, uint8_t table_index)
    // {
    //     if (table_index == 1) {
    //         return Util::ByteAlign((kEntriesPerPark - 1) * kMaxAverageDeltaTable1) / 8;
    //     }
    //     return Util::ByteAlign((kEntriesPerPark - 1) * kMaxAverageDelta) / 8;
    // }

    // static uint32_t CalculateStubsSize(uint32_t k)
    // {
    //     return Util::ByteAlign((kEntriesPerPark - 1) * (k - kStubMinusBits)) / 8;
    // }

    // static uint32_t CalculateParkSize(uint8_t k, uint8_t table_index)
    // {
    //     return CalculateLinePointSize(k) + CalculateStubsSize(k) +
    //            CalculateMaxDeltasSize(k, table_index);
    // }


    
};

