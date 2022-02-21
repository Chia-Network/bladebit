#pragma once
#include "plotting/PlotTools.h"
#include "util/KeyTools.h"

struct GlobalPlotConfig
{
    uint32 threadCount   = 0;
    uint32 plotCount     = 1;

    const char* plotIdStr    = nullptr;
    const char* plotMemoStr  = nullptr;
    // byte*       plotId       = new byte[BB_PLOT_ID_LEN];
    // byte*       plotMemo     = new byte[BB_PLOT_MEMO_MAX_SIZE];
    // uint16      plotMemoSize = 0;

    // #TODO: Allow multiple output paths
    const char* outputFolder = nullptr;

    bool showMemo           = false;
    bool warmStart          = false;
    bool disableNuma        = false;
    bool disableCpuAffinity = false;

    bls::G1Element  farmerPublicKey;
    bls::G1Element* poolPublicKey          = nullptr;   // Either poolPublicKey or poolContractPuzzleHash must be set.
    PuzzleHash*     poolContractPuzzleHash = nullptr;   // If both are set, poolContractPuzzleHash will be used over
                                                        // the poolPublicKey.
};

