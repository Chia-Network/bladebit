#pragma once
#include "plotting/Compression.h"
#include <string>

struct PuzzleHash;
namespace bls
{
    class G1Element;
}

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

    uint32       outputFolderCount         = 1;
    std::string* outputFolders             = nullptr;

    bool            benchmarkMode          = false;            // For testing plot timings -- Does not write the plot file to disk
    bool            showMemo               = false;
    bool            warmStart              = false;
    bool            disableNuma            = false;
    bool            disableCpuAffinity     = false;
    bool            disableOutputDirectIO  = false;            // Do not use direct I/O when writing the plot files
    bool            verbose                = false;            // Allow some verbose output
    uint32          compressionLevel       = 0;                // 0 == no compression. 1 = 16 bits. 2 = 15 bits, ..., 6 = 11 bits
    uint32          compressedEntryBits    = 32;               // Bit size of table 1 entries. If compressed, then it is set to <= 16.
    FSE_CTable*     ctable                 = nullptr;          // Compression table if making compressed plots
    size_t          cTableSize             = 0;
    uint32          numDroppedTables       = 0;                // When compression level > 0 : 1. > 8 : 2. Otherwise 0 (no compression)
    CompressionInfo compressionInfo        = {};

    bls::G1Element* farmerPublicKey        = nullptr;
    bls::G1Element* poolPublicKey          = nullptr;   // Either poolPublicKey or poolContractPuzzleHash must be set.
    PuzzleHash*     poolContractPuzzleHash = nullptr;   // If both are set, poolContractPuzzleHash will be used over
                                                        // the poolPublicKey.

};

