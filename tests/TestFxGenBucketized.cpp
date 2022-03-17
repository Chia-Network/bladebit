#include "catch2/catch_test_macros.hpp"
#include "util/jobs/MemJobs.h"
#include "threading/ThreadPool.h"
#include "plotting/PlotTypes.h"
#include "util/Util.h"
#include "util/Log.h"

// struct FxGen
// {
//     template<typename TMeta>
//     static void CalculateFx( const int64 entryCount, const uint64 bucket, const Pair* pairs );
// };

// //-----------------------------------------------------------
// TEST_CASE( "FxGenBucketized", "[fx_tests]" )
// {
//     const uint32 threadCount = SysHost::GetLogicalCPUCount();
//     ThreadPool pool( threadCount );


// }


// //-----------------------------------------------------------
// template<typename TYOut, typename TMetaIn, typename TMetaOut>
// void ComputeFxJob( FpFxJob<TYOut, TMetaIn, TMetaOut>* job )
// {
//     const size_t metaKMultiplierIn  = SizeForMeta<TMetaIn >::Value;
//     const size_t metaKMultiplierOut = SizeForMeta<TMetaOut>::Value;

//     // Table 7 (identified by 0 metadata output) we don't have k + kExtraBits sized y's.
//     // so we need to shift by 32 bits, instead of 26.
//     constexpr size_t extraBitsShift = metaKMultiplierOut == 0 ? 0 : kExtraBits; 

//     const uint64   entryCount    = job->entryCount;
//     const Pair*    lrPairs       = job->lrPairs;
//     const TMetaIn* inMetaBuffer  = job->inMetaBuffer;
//     const uint64*  inYBuffer     = job->inYBuffer;
//     TMetaOut*      outMetaBuffer = job->outMetaBuffer;
//     TYOut*         outYBuffer    = job->outYBuffer;

//     #if _DEBUG
//         uint64 lastLeft = 0;
//     #endif

//     // Intermediate metadata holder
//     uint64 lrMetadata[4];

//     for( uint64 i = 0; i < entryCount; i++ )
//     {
//         const Pair& pair = lrPairs[i];

//         #if _DEBUG
//             ASSERT( pair.left >= lastLeft );
//             lastLeft = pair.left;
//         #endif

//         // Read y
//         const uint64 y = inYBuffer[pair.left];

//         // Read metadata
//         if constexpr( metaKMultiplierIn == 1 )
//         {
//             uint32* meta32 = (uint32*)lrMetadata;

//             meta32[0] = inMetaBuffer[pair.left ];    // Metadata( l and r x's)
//             meta32[1] = inMetaBuffer[pair.right];
//         }
//         else if constexpr( metaKMultiplierIn == 2 )
//         {
//             lrMetadata[0] = inMetaBuffer[pair.left ];
//             lrMetadata[1] = inMetaBuffer[pair.right];
//         }
//         else
//         {
//             // For 3 and 4 we just use 16 bytes (2 64-bit entries)
//             const Meta4* inMeta4 = static_cast<const Meta4*>( inMetaBuffer );
//             const Meta4& meta4L  = inMeta4[pair.left ];
//             const Meta4& meta4R  = inMeta4[pair.right];

//             lrMetadata[0] = meta4L.m0;
//             lrMetadata[1] = meta4L.m1;
//             lrMetadata[2] = meta4R.m0;
//             lrMetadata[3] = meta4R.m1;
//         }

//         TYOut f = (TYOut)ComputeFx<metaKMultiplierIn, metaKMultiplierOut, extraBitsShift>( y, lrMetadata, (uint64*)outMetaBuffer );

//         outYBuffer[i] = f;

//         if constexpr( metaKMultiplierOut != 0 )
//             outMetaBuffer ++;
//     }
// }

// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored "-Wattributes"

// //-----------------------------------------------------------
// template<size_t metaKMultiplierIn, size_t metaKMultiplierOut, uint ShiftBits>
// FORCE_INLINE uint64 ComputeFx( uint64 y, uint64* metaData, uint64* metaOut )
// {
//     static_assert( metaKMultiplierIn != 0, "Invalid metaKMultiplier" );

//     // Helper consts
//     const uint   k           = _K;
//     const uint32 ySize       = k + kExtraBits;         // = 38
//     const uint32 yShift      = 64 - (k + ShiftBits);   // = 26 or 32
//     const size_t metaSize    = k * metaKMultiplierIn;
//     const size_t metaSizeLR  = metaSize * 2;

//     const size_t bufferSize  = CDiv( ySize + metaSizeLR, 8 );


//     // Hashing input and output buffers
//     uint64 input [5];       // y + L + R
//     uint64 output[4];       // blake3 hashed output

//     blake3_hasher hasher;


//     // Prepare the input buffer depending on the metadata size
//     if constexpr( metaKMultiplierIn == 1 )
//     {
//         /**
//          * 32-bit per metadata (1*32) L == 4 R == 4
//          * Metadata: L: [32] R: [32]
//          * 
//          * Serialized:
//          *  y  L     L  R -
//          * [38|26]  [6|32|-]
//          *    0        1
//          */

//         const uint64 l = reinterpret_cast<uint32*>( metaData )[0];
//         const uint64 r = reinterpret_cast<uint32*>( metaData )[1];

//         input[0] = Swap64( y << 26 | l >> 6  );
//         input[1] = Swap64( l << 58 | r << 26 );

//         // Metadata is just L + R of 8 bytes
//         if constexpr( metaKMultiplierOut == 2 )
//             metaOut[0] = l << 32 | r;
//     }
//     else if constexpr( metaKMultiplierIn == 2 )
//     {
//         /**
//          * 64-bit per metadata (2*32) L == 8 R == 8
//          * Metadata: L: [64] R: [64]
//          * 
//          * Serialized:
//          *  y   L    L  R     R  -
//          * [38|26]  [38|26]  [38|-]
//          *    0        1       2
//          */
//         const uint64 l = metaData[0];
//         const uint64 r = metaData[1];

//         input[0] = Swap64( y << 26 | l >> 38 );
//         input[1] = Swap64( l << 26 | r >> 38 );
//         input[2] = Swap64( r << 26 );

//         // Metadata is just L + R again of 16 bytes
//         if constexpr( metaKMultiplierOut == 4 )
//         {
//             metaOut[0] = l;
//             metaOut[1] = r;
//         }
//     }
//     else if constexpr( metaKMultiplierIn == 3 )
//     {
//         /**
//         * 96-bit per metadata (3*32) L == 12 bytes R == 12 bytes
//         * Metadata: L: [64][32] R: [64][32]
//         *               L0  L1      R0  R1 
//         * Serialized:
//         *  y  L0    L0 L1   L1 R0   R0 R1 -
//         * [38|26]  [38|26]  [6|58]  [6|32|-]
//         *    0        1       2        3
//         */

//         const uint64 l0 = metaData[0];
//         const uint64 l1 = metaData[1] & 0xFFFFFFFF;
//         const uint64 r0 = metaData[2];
//         const uint64 r1 = metaData[3] & 0xFFFFFFFF;
        
//         input[0] = Swap64( y  << 26 | l0 >> 38 );
//         input[1] = Swap64( l0 << 26 | l1 >> 6  );
//         input[2] = Swap64( l1 << 58 | r0 >> 6  );
//         input[3] = Swap64( r0 << 58 | r1 << 26 );
//     }
//     else if constexpr( metaKMultiplierIn == 4 )
//     {
//         /**
//         * 128-bit per metadata (4*32) L == 16 bytes R == 16 bytes
//         * Metadata  : L [64][64] R: [64][64]
//         *                L0  L1      R0  R1
//         * Serialized: 
//         *  y  L0    L0 L1    L1 R0    R0 R1    R1 -
//         * [38|26]  [38|26]  [38|26]  [38|26]  [38|-]
//         *    0        1        2        3        4
//         */
        
//         const uint64 l0 = metaData[0];
//         const uint64 l1 = metaData[1];
//         const uint64 r0 = metaData[2];
//         const uint64 r1 = metaData[3];

//         input[0] = Swap64( y  << 26 | l0 >> 38 );
//         input[1] = Swap64( l0 << 26 | l1 >> 38 );
//         input[2] = Swap64( l1 << 26 | r0 >> 38 );
//         input[3] = Swap64( r0 << 26 | r1 >> 38 );
//         input[4] = Swap64( r1 << 26 );
//     }


//     // Hash the input
//     blake3_hasher_init( &hasher );
//     blake3_hasher_update( &hasher, input, bufferSize );
//     blake3_hasher_finalize( &hasher, (uint8_t*)output, sizeof( output ) );

//     uint64 f = Swap64( *output ) >> yShift;


//     ///
//     /// Calculate metadata for tables >= 4
//     ///
//     // Only table 6 do we output size 2 with an input of size 3.
//     // Otherwise for output == 2 we calculate the output above
//     // as it is just L + R, and it is not taken from the output
//     // of the blake3 hash.
//     if constexpr ( metaKMultiplierOut == 2 && metaKMultiplierIn == 3 )
//     {
//         const uint64 h0 = Swap64( output[0] );
//         const uint64 h1 = Swap64( output[1] );

//         metaOut[0] = h0 << ySize | h1 >> 26;
//     }
//     else if constexpr ( metaKMultiplierOut == 3 )
//     {
//         const uint64 h0 = Swap64( output[0] );
//         const uint64 h1 = Swap64( output[1] );
//         const uint64 h2 = Swap64( output[2] );

//         metaOut[0] = h0 << ySize | h1 >> 26;
//         metaOut[1] = ((h1 << 6) & 0xFFFFFFC0) | h2 >> 58;
//     }
//     else if constexpr ( metaKMultiplierOut == 4 && metaKMultiplierIn != 2 ) // In = 2 is calculated above with L + R
//     {
//         const uint64 h0 = Swap64( output[0] );
//         const uint64 h1 = Swap64( output[1] );
//         const uint64 h2 = Swap64( output[2] );

//         metaOut[0] = h0 << ySize | h1 >> 26;
//         metaOut[1] = h1 << 38    | h2 >> 26;
//     }
    
//     return f;
// }

// #pragma GCC diagnostic pop