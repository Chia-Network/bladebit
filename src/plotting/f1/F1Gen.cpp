#include "F1Gen.h"
#include "pos/chacha8.h"
#include "plotting/PlotTools.h"
#include "util/BitView.h"

//-----------------------------------------------------------
uint64 F1GenSingleForK( const uint32 k, const byte plotId[BB_PLOT_ID_LEN], uint64 x )
{
    ASSERT( k > 0 );
    ASSERT( k <= BB_CHIA_K_MAX_VALUE );

    const uint32 xShift = k - kExtraBits;

    // Prepare ChaCha key
    byte key[32] = { 1 };
    memcpy( key + 1, plotId, 31 );

    chacha8_ctx chacha;
    chacha8_keysetup( &chacha, key, 256, NULL );

    // Enough to hold 2 cha-cha blocks since a value my span over 2 blocks
    byte AlignAs(8) blocks[kF1BlockSize*2];

    const uint64 blockIdx    = x * k / kF1BlockSizeBits; 
    const uint64 blockEndIdx = (x * k + k - 1) / kF1BlockSizeBits; 
    
    const uint32 nBlocks = (uint32)(blockEndIdx - blockIdx + 1);

    chacha8_get_keystream( &chacha, blockIdx, nBlocks, blocks );

    // Get the starting and end locations of y in bits relative to our block
    const uint64 bitStart = x * k - blockIdx * kF1BlockSizeBits;

    CPBitReader hashBits( blocks, sizeof( blocks ) * 8 );
    
    uint64 y = hashBits.Read64At( bitStart, k );
    y = ( y << kExtraBits ) | ( x >> xShift );

    return y;
}