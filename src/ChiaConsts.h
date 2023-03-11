#pragma once
#include "util/Util.h"
#include "plotting/Tables.h"

// Hard-coded k to 32
#define _K 32
#define ENTRIES_PER_TABLE ( 1ull << _K )

///
/// These are extracted from chiapos.
/// Comments are taken directly from chiapos unless otherwise specified.
///

// F1 evaluations are done in batches of 2^kBatchSizes
#define kBatchSizes 8

// ChaCha8 block size
#define kF1BlockSizeBits 512

#define kF1BlockSize (kF1BlockSizeBits / 8u)

// Extra bits of output from the f functions. Instead of being a function from k -> k bits,
// it's a function from k -> k + kExtraBits bits. This allows less collisions in matches.
// Refer to the paper for mathematical motivations.
#define kExtraBits 6

// Convenience variable
#define kExtraBitsPow (1 << kExtraBits)

// B and C groups which constitute a bucket, or BC group. These groups determine how
// elements match with each other. Two elements must be in adjacent buckets to match.
#define kB  119ull
#define kC  127ull
#define kBC (kB * kC)

// This (times k) is the length of the metadata that must be kept for each entry. For example,
// for a table 4 entry, we must keep 4k additional bits for each entry, which is used to compute f5.
// @Harold: Not that we changed this from 1-indexing in the chiapos implementation
//          to use 0-indexing. So index 0 == table 1.
const byte kVectorLens[] = { 0, 1, 2, 4, 4, 3, 2 };

// Defined in PlotContext.cpp
extern uint16_t L_targets[2][kBC][kExtraBitsPow];


// How many f7s per C1 entry, and how many C1 entries per C2 entry
#define kCheckpoint1Interval 10000
#define kCheckpoint2Interval 10000

// EPP for the final file, the higher this is, the less variability, and lower delta
// Note: if this is increased, ParkVector size must increase
#define kEntriesPerPark      2048

// To store deltas for EPP entries, the average delta must be less than this number of bits
#define kMaxAverageDeltaTable1 5.6
#define kMaxAverageDelta       3.5

// C3 entries contain deltas for f7 values, the max average size is the following
#define kC3BitsPerEntry 2.4

// The number of bits in the stub is k minus this value
#define kStubMinusBits  3

// The ANS encoding R values for the 7 final plot tables
// Tweaking the R values might allow lowering of the max average deltas,
// and reducing final plot size
const double kRValues[7] = { 4.7, 2.75, 2.75, 2.7, 2.6, 2.45 };

// The ANS encoding R value for the C3 checkpoint table
#define kC3R 1.0

#define kPOSMagic          "Proof of Space Plot"
#define kFormatDescription "v1.0"

#define CHIA_PLOT_V2_MAGIC  0x544F4C50ul   // "PLOT"
#define CHIA_PLOT_VERSION_2 2

#define BB_PLOT_ID_LEN 32
#define BB_PLOT_ID_HEX_LEN (BB_PLOT_ID_LEN * 2)

#define BB_PLOT_MEMO_MAX_SIZE (48+48+32)

#define BB_PLOT_FILE_LEN_TMP (sizeof( "plot-k32-2021-08-05-18-55-77a011fc20f0003c3adcc739b615041ae56351a22b690fd854ccb6726e5f43b7.plot.tmp" ) - 1)
#define BB_PLOT_FILE_LEN (BB_PLOT_FILE_LEN_TMP - 4)

#define BB_COMPRESSED_PLOT_FILE_LEN_TMP (sizeof( "plot-k32-c01-2021-08-05-18-55-77a011fc20f0003c3adcc739b615041ae56351a22b690fd854ccb6726e5f43b7.plot.tmp" ) - 1)
#define BB_COMPRESSED_PLOT_FILE_LEN (BB_PLOT_FILE_LEN_TMP - 4)

#define BB_PLOT_PROOF_X_COUNT 64

#define BB_CHIA_CHALLENGE_SIZE 32
#define BB_CHIA_QUALITY_SIZE   32

#define BB_CHIA_K_MAX_VALUE    50


// Initializes L_targets table
//-----------------------------------------------------------
inline void LoadLTargets()
{
    // #TODO: Fix this as it is not thread-safe
    static bool _initialized = false;

    if( _initialized )
        return;

    _initialized = true;
    
    for( byte parity = 0; parity < 2; parity++ ) 
    {
        for( uint16 i = 0; i < kBC; i++ )
        {
            uint16_t indJ = i / kC;

            for( uint16 m = 0; m < kExtraBitsPow; m++ )
            {
                const uint16 yr = ((indJ + m) % kB) * kC + (((2 * m + parity) * (2 * m + parity) + i) % kC);
                
                L_targets[parity][i][m] = yr;
            }
        }
    }
}


// This is the full size of the deltas section in a park. However, it will not be fully filled
//-----------------------------------------------------------
inline constexpr size_t CalculateMaxDeltasSize( const TableId tableId )
{
    ASSERT( tableId < TableId::Table7 );

    if( tableId == TableId::Table1 )
        return CDiv( (size_t)((kEntriesPerPark - 1) * kMaxAverageDeltaTable1), 8 );
    
    return CDiv( (size_t)( (kEntriesPerPark - 1) * kMaxAverageDelta ), 8 );
}

/// Fixed size for parks
//-----------------------------------------------------------
inline constexpr size_t CalculateParkSize( TableId tableId )
{
    // if( tableId == TableId::Table1 ) return 8960;
    return 
        CDiv( _K * 2, 8 ) +                                         // LinePoint size
        CDiv( (kEntriesPerPark - 1) * (_K - kStubMinusBits), 8 ) +  // Stub Size 
        CalculateMaxDeltasSize( tableId );                          // Max delta

    // return CalculateLinePointSize(k) + CalculateStubsSize(k) +
    //         CalculateMaxDeltasSize(k, table_index);
}

//-----------------------------------------------------------
inline constexpr size_t CalculateParkSize( const TableId tableId, const uint32 k )
{
    ASSERT( k >= 20 );

    return 
        CDiv( k * 2, 8 ) +                                         // LinePoint size
        CDiv( (kEntriesPerPark - 1) * (k - kStubMinusBits), 8 ) +  // Stub Size 
        CalculateMaxDeltasSize( tableId );                         // Max delta

    // return CalculateLinePointSize(k) + CalculateStubsSize(k) +
    //         CalculateMaxDeltasSize(k, table_index);
}

// Calculates the size of one C3 park. This will store bits for each f7 between
// two C1 checkpoints, depending on how many times that f7 is present.
//-----------------------------------------------------------
constexpr inline static size_t CalculateC3Size()
{
    return (size_t)CDiv( kC3BitsPerEntry * kCheckpoint1Interval, 8 );
}

//-----------------------------------------------------------
inline constexpr static size_t CalculatePark7Size( const uint k )
{
    return CDiv( (k + 1) * (uint64)kEntriesPerPark, 8 );
}

//-----------------------------------------------------------
constexpr inline static size_t LinePointSizeBits( uint32 k )
{
    return (size_t)k * 2;
}

//-----------------------------------------------------------
constexpr inline static size_t LinePointSizeBytes( uint32 k )
{
    return CDiv( LinePointSizeBits( k ), 8 );
}
