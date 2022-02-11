#include "memplot/CTables.h"
#include "ChiaConsts.h"
#include "util/Log.h"
#include "util/BitView.h"
#include "io/FileStream.h"
#include "PlotReader.h"
#include "plotshared/PlotTools.h"
#include "memplot/LPGen.h"
#include "pos/chacha8.h"
#include "b3/blake3.h"

#define PROOF_X_COUNT 64
#define MAX_K_SIZE          50
#define MAX_META_MULTIPLIER 4
#define MAX_Y_BIT_SIZE      ( MAX_K_SIZE + kExtraBits )
#define MAX_META_BIT_SIZE   ( MAX_K_SIZE * MAX_META_MULTIPLIER )
#define MAX_FX_BIT_SIZE     ( MAX_Y_BIT_SIZE + MAX_META_BIT_SIZE )

typedef Bits<MAX_Y_BIT_SIZE>    YBits;
typedef Bits<MAX_META_BIT_SIZE> MetaBits;
typedef Bits<MAX_FX_BIT_SIZE>   FxBits;


std::string plotPath = "/mnt/p5510a/plots/plot-k32-2022-02-07-03-50-c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835.plot";

bool FetchProof( PlotReader& plot, uint64 t6LPIndex, uint64 fullProofXs[32] );
bool ValidateFullProof( uint64 fullProofXs[32] );

uint64 BytesToUInt64( const byte bytes[8] );
uint64 SliceUInt64FromBits( const byte* bytes, uint32 bitOffset, uint32 bitCount );

bool FxMatch( uint64 yL, uint64 yR );

void FxGen( const TableId table, const uint32 k, 
            const uint64 y, const MetaBits& metaL, const MetaBits& metaR,
            uint64& outY, MetaBits& outMeta );

//-----------------------------------------------------------
void TestPlotValidate()
{
    LoadLTargets();

    MemoryPlot plotFile;
    FatalIf( !plotFile.Open( plotPath.c_str() ), 
        "Failed to open plot at path '%s'.", plotPath.c_str() );

    PlotReader plot( plotFile );

    // uint64 numLinePoint = 0;
    // uint128* linePoints = bbcalloc<uint128>( kEntriesPerPark );
    // plot.ReadLPPark( PlotTable::Table6, 0, linePoints, numLinePoint );

    // Test read C3 park
    uint64* f7Entries = bbcalloc<uint64>( kCheckpoint1Interval );
    memset( f7Entries, 0, kCheckpoint1Interval * sizeof( uint64 ) );

    // Check how many C3 parks we have
    const uint64 c3ParkCount = plotFile.TableSize( PlotTable::C1 ) / sizeof( uint32 ) - 1;

    // Read the C3 parks
    // FatalIf( plot.ReadC3Park( 0, f7Entries ) < 0, "Could not read C3 park." );

    // Test read p7
    uint64* p7Entries = bbcalloc<uint64>( kEntriesPerPark );
    // memset( p7Entries, 0, sizeof( kEntriesPerPark ) * sizeof( uint64 ) );



    uint64 curPark7 = 0;
    FatalIf( !plot.ReadP7Entries( 0, p7Entries ), "Failed to read P7 0." );
    // ASSERT( p7Entries[0] == 3208650999 );
    
    uint64 proofFailCount = 0;
    uint64 fullProofXs[32];

    for( uint64 i = 0; i < c3ParkCount; i++ )
    {
        FatalIf( plot.ReadC3Park( i, f7Entries ) < 0, "Could not read C3 park %llu.", i );

        const uint64 f7IdxBase = i * kCheckpoint1Interval;

        for( uint32 e = 0; e < kCheckpoint1Interval; i++ )
        {
            const uint64 f7Idx       = f7IdxBase + e;
            const uint64 p7ParkIndex = f7Idx / kEntriesPerPark;
            const uint64 f7          = f7Entries[e];

            if( p7ParkIndex != curPark7 )
            {
                ASSERT( p7ParkIndex == curPark7+1 );
                curPark7 = p7ParkIndex;

                FatalIf( !plot.ReadP7Entries( p7ParkIndex, p7Entries ), "Failed to read P7 %llu.", p7ParkIndex );
            }

            const uint64 p7LocalIdx = f7Idx - p7ParkIndex * kEntriesPerPark;

            const uint64 t6Index = p7Entries[p7LocalIdx];
            
            if( FetchProof( plot, t6Index, fullProofXs ) )
            {
                // Now we can validate the proof
            }
            else
            {
                proofFailCount++;
                Log::Line( "Proof fetch failed for f7[%llu] = %llu ( 0x%16llx ) ", f7Idx, f7, f7 );
            }
            
        }
    }
}

//-----------------------------------------------------------
bool FetchProof( PlotReader& plot, uint64 t6LPIndex, uint64 fullProofXs[32] )
{
    uint64 lpIndices[2][PROOF_X_COUNT];
    memset( lpIndices, 0, sizeof( lpIndices ) );

    uint64* lpIdxSrc = lpIndices[0];
    uint64* lpIdxDst = lpIndices[1];

    *lpIdxSrc = t6LPIndex;

    // Fetch line points to back pointers going through all our tables
    // from 6 to 1, grabbing all of the x's that make up a proof.
    uint32 lookupCount = 1;

    for( TableId table = TableId::Table6; table >= TableId::Table1; table-- )
    {
        ASSERT( lookupCount <= 32 );

        for( uint32 i = 0, dst = 0; i < lookupCount; i++, dst += 2 )
        {
            const uint64 idx = lpIdxSrc[i];

            uint128 lp = 0;
            if( !plot.ReadLP( table, idx, lp ) )
                return false;

            BackPtr ptr = LinePointToSquare( lp );

            lpIdxDst[dst+0] = ptr.x;
            lpIdxDst[dst+1] = ptr.y;
        }

        lookupCount <<= 1;

        std::swap( lpIdxSrc, lpIdxDst );
        memset( lpIdxDst, 0, sizeof( uint64 ) * PROOF_X_COUNT );
    }

    // Full proof x's will be at the src ptr
    memcpy( fullProofXs, lpIdxSrc, sizeof( uint64 ) * 32 );
    return true;
}

//-----------------------------------------------------------
bool ValidateFullProof( PlotReader& plot, uint64 fullProofXs[32] )
{
    const uint32 k        = plot.PlotFile().K();
    const uint32 yBitSize = k + kExtraBits;

    uint64   fx  [PROOF_X_COUNT];
    MetaBits meta[PROOF_X_COUNT];

    // Convert these x's to f1 values
    uint64 f1[PROOF_X_COUNT] = {0};
    {
        const uint32 xShift = k - kExtraBits;
        
        // Prepare ChaCha key
        byte key[32] = { 1 };
        memcpy( key + 1, plot.PlotFile().PlotId(), 31 );

        chacha8_ctx chacha;
        chacha8_keysetup( &chacha, key, 256, NULL );

        // Enough to hold 2 cha-cha blocks since a value my span over 2 blocks
        byte blocks[kF1BlockSize*2];

        for( uint32 i = 0; i < PROOF_X_COUNT; i++ )
        {
            const uint64 x        = fullProofXs[i];
            const uint64 blockIdx = x * k / kF1BlockSizeBits; 

            chacha8_get_keystream( &chacha, blockIdx, 2, blocks );

            // Get the starting and end locations of y in bits relative to our block
            const uint64 bitStart = x * k - blockIdx * kF1BlockSizeBits;

            const uint64 y = SliceUInt64FromBits( blocks, bitStart, k );

            fx  [i] = ( y << kExtraBits ) | ( x >> xShift );
            meta[i] = MetaBits( x, k );
        }
    }

    // Forward propagate f1 values to get the final f7
    uint32 iterCount = PROOF_X_COUNT;
    for( TableId table = TableId::Table2; table <= TableId::Table7; table++ )
    {
        for( uint32 i = 0, dst = 0; i < iterCount; i+= 2, dst++ )
        {
            uint64 y0 = fx[i+0];
            uint64 y1 = fx[i+1];

            if( y0 > y0 ) 
                std::swap( y0, y1 );

            // Must be on the same group
            if( !FxMatch( y0, y1 ) )
                return false;


            // FxGen
            const MetaBits& lMeta = meta[i+0];
            const MetaBits& rMeta = meta[i+1];

            uint64 outY;
            MetaBits outMeta;
            FxGen( table, k, y0, lMeta, rMeta, outY, outMeta );

            fx  [dst] = outY;
            meta[dst] = outMeta;
        }
    }

    return true;
}

//-----------------------------------------------------------
bool FxMatch( uint64 yL, uint64 yR )
{
    uint8  rMapCounts [kBC];
    uint16 rMapIndices[kBC];

    const uint64 groupL = yL / kBC;
    const uint64 groupR = yR / kBC;

    if( groupR - groupL != 1 )
        return false;

    // Groups are adjacent, check if the y values actually match
    const uint16 parity = groupL & 1;

    const uint64 groupLRangeStart = groupL * kBC;
    const uint64 groupRRangeStart = groupR * kBC;

    const uint64 localLY = yL - groupLRangeStart;
    const uint64 localRY = yR - groupRRangeStart;
    
    for( int iK = 0; iK < kExtraBitsPow; iK++ )
    {
        const uint64 targetR = L_targets[parity][localLY][iK];
        
        if( targetR == localRY )
            return true;
    } 

    return false;
}

//-----------------------------------------------------------
void FxGen( const TableId table, const uint32 k, 
            const uint64 y, const MetaBits& metaL, const MetaBits& metaR,
            uint64& outY, MetaBits& outMeta )
{
    FxBits input( y, k + kExtraBits );

    if( table < TableId::Table4 )
    {
        outMeta = metaL + metaR;
        input += outMeta;
    }
    else
    {
        input += metaL;
        input += metaR;
    }

    byte inputBytes[64];
    byte hashBytes [32];

    input.ToBytes( inputBytes );

    blake3_hasher hasher;
    blake3_hasher_init    ( &hasher );
    blake3_hasher_update  ( &hasher, inputBytes, input.LengthBytes() );
    blake3_hasher_finalize( &hasher, hashBytes, sizeof( hashBytes ) );

    outY = SliceUInt64FromBits( hashBytes, 0, sizeof( hashBytes ) * 8 )
            >> ( 64 - (k + kExtraBits) );

    if( table >= TableId::Table4 && table < TableId::Table7 )
    {
        size_t multiplier = 0;
        switch( table )
        {
            case TableId::Table4: multiplier = TableMetaOut<TableId::Table4>::Multiplier; break;
            case TableId::Table5: multiplier = TableMetaOut<TableId::Table5>::Multiplier; break;
            case TableId::Table6: multiplier = TableMetaOut<TableId::Table6>::Multiplier; break;
            default: 
                ASSERT( 0 );
                break;
        }

        outMeta = MetaBits( hashBytes, k * multiplier, k + kExtraBits );
    }
}

//-----------------------------------------------------------
/// Convertes 8 bytes to uint64 and endian-swaps it.
/// This takes any byte alignment, so that bytes does
/// not have to be aligned to 64-bit boundary.
/// This is for compatibility for how chiapos extracts
/// bytes into integers.
//-----------------------------------------------------------
inline uint64 BytesToUInt64( const byte bytes[8] )
{
    uint64 tmp;
    memcpy( &tmp, bytes, sizeof( uint64 ) );
    return Swap64( tmp );
}

//-----------------------------------------------------------
// Treats bytes as a set of 64-bit big-endian fields,
// from which it will extract a whole 64-bit value
// at the given bit offset. 
// The result may be truncated if the requested number of 
// bits + the number of bits overflows the 64-bit field.
// That is, if the local bit offset in the target bit field
// + the bitCount is greater than 64.
// This function is for compatibility with the way chiapos
// slices bits off of binary byte blobs.
//-----------------------------------------------------------
inline uint64 SliceUInt64FromBits( const byte* bytes, uint32 bitOffset, uint32 bitCount )
{
    ASSERT( bitCount <= 64 );

    const uint64 startByte = bitOffset / 8;
    bytes += startByte;

    // Convert bit offset to be local to the uint64 field
    bitOffset -= bitOffset / 64;

    uint64 field = BytesToUInt64( bytes );
    
    field <<= bitOffset;     // Start bits from the MSBits
    field >>= 64 - bitCount; // Take the MSbits

    return field;
}

