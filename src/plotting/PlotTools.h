#pragma once
#include "ChiaConsts.h"
#include "util/KeyTools.h"

#define BB_PLOT_PROOF_X_COUNT 32

#define BB_PLOT_ID_LEN 32
#define BB_PLOT_ID_HEX_LEN (BB_PLOT_ID_LEN * 2)

#define BB_PLOT_MEMO_MAX_SIZE (48+48+32)

#define BB_PLOT_FILE_LEN_TMP (sizeof( "plot-k32-2021-08-05-18-55-77a011fc20f0003c3adcc739b615041ae56351a22b690fd854ccb6726e5f43b7.plot.tmp" ) - 1)
#define BB_PLOT_FILE_LEN (BB_PLOT_FILE_LEN_TMP - 4)

struct PlotTools
{
    static void GenPlotFileName( const byte plotId[BB_PLOT_ID_LEN], char outPlotFileName[BB_PLOT_FILE_LEN] );
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

