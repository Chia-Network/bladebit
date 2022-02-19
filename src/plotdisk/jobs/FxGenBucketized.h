#pragma once

#include "threading/AutoResetSignal.h"
#include "../DiskBufferQueue.h"
#include "plotting/Tables.h"
#include "JobShared.h"
#include "plotting/PlotTypes.h"
#include "b3/blake3.h"


// //-----------------------------------------------------------
// template<TableId tableId>
// static void ComputeFxForTable( 
//     const uint64  bucket, 
//     uint32        entryCount, 
//     const Pairs   pairs,
//     const uint32* yIn, 
//     const uint64* metaInA, 
//     const uint64* metaInB,
//     uint32*       yOut,
//     byte*         bucketOut, 
//     uint64*       metaOutA, 
//     uint64*       metaOutB );

// struct FxBucket
// {
//     // IO buffers
//     uint32* yFront    ;
//     uint32* yBack     ;
//     uint32* sortKey   ;
//     uint64* metaAFront; 
//     uint64* metaBFront;
//     uint64* metaABack ;
//     uint64* metaBBack ;
//     Pairs   pairs     ;

//     // Back/front buffer fences
//     AutoResetSignal frontFence;
//     AutoResetSignal backFence ;

//     // Fixed bufers
//     uint32* groupBoundaries;
//     byte*   bucketId       ;
//     uint32* yTmp           ;
//     uint64* metaATmp       ;
//     uint64* metaBTmp       ;

//     // Current file ids
//     FileId  yFileId    ;
//     FileId  metaAFileId;
//     FileId  metaBFileId;

//     // Cross-bucket data (for groups that overlap between 2 bucket boundaries)
//     uint32* yCross    ;
//     uint64* metaACross;
//     uint64* metaBCross;

//     // Used for overflow entries when using Direct IO (entries that not align to file block boundaries)
//     // OverflowBuffer yOverflow;
//     // OverflowBuffer metaAOverflow;
//     // OverflowBuffer metaBOverflow;
// };

template<TableId table>
struct FxGenBucketized
{
    // void GenerateFxBucket(
    //     ThreadPool&      pool, 
    //     uint             threadCount,
    //     const size_t     chunkSize
    // );

    static void GenerateFxBucketizedToDisk(
        DiskBufferQueue& diskQueue,
        size_t           writeInterval,
        ThreadPool&      pool, 
        uint32           threadCount,

        const uint32     bucketIdx,
        const uint32     entryCount,
        const uint32     sortKeyOffset,
        Pairs            pairs,
        byte*            bucketIndices,

        const uint32*    yIn,
        const uint64*    metaAIn,
        const uint64*    metaBIn,

        uint32*          yTmp,
        uint64*          metaATmp,
        uint64*          metaBTmp,

        uint32           bucketCounts[BB_DP_BUCKET_COUNT]
    );

    static void GenerateFxBucketizedInMemory(
        ThreadPool&      pool, 
        uint32           threadCount,
        uint32           bucketIdx,
        const uint32     entryCount,
        Pairs            pairs,
        byte*            bucketIndices,

        const uint32*    yIn,
        const uint64*    metaAIn,
        const uint64*    metaBIn,

        uint32*          yTmp,
        uint32*          metaATmp,
        uint32*          metaBTmp,

        uint32*          yOut,
        uint32*          metaAOut,
        uint32*          metaBOut,

        uint32           bucketCounts[BB_DP_BUCKET_COUNT]
    );
};

// Instantiate a version for each table
template struct FxGenBucketized<TableId::Table1>;
template struct FxGenBucketized<TableId::Table2>;
template struct FxGenBucketized<TableId::Table3>;
template struct FxGenBucketized<TableId::Table4>;
template struct FxGenBucketized<TableId::Table5>;
template struct FxGenBucketized<TableId::Table6>;
template struct FxGenBucketized<TableId::Table7>;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"


//-----------------------------------------------------------
template<TableId tableId>
FORCE_INLINE 
void ComputeFxForTable( const uint64 bucket, uint32 entryCount, const Pairs pairs, 
                        const uint32* yIn, const uint64* metaInA, const uint64* metaInB, 
                        uint32* yOut, byte* bucketOut, uint64* metaOutA, uint64* metaOutB, uint32 jobId )
{
    constexpr size_t metaKMultiplierIn  = TableMetaIn <tableId>::Multiplier;
    constexpr size_t metaKMultiplierOut = TableMetaOut<tableId>::Multiplier;

    // Helper consts
    // Table 7 (identified by 0 metadata output) we don't have k + kExtraBits sized y's.
    // so we need to shift by 32 bits, instead of 26.
    // constexpr uint extraBitsShift = tableId == TableId::Table7 ? 0 : kExtraBits;

    constexpr uint   shiftBits   = metaKMultiplierOut == 0 ? 0 : kExtraBits;
    constexpr uint   k           = _K;
    constexpr uint32 ySize       = k + kExtraBits;         // = 38
    constexpr uint32 yShift      = 64u - (k + shiftBits);  // = 26 or 32
    constexpr size_t metaSize    = k * metaKMultiplierIn;
    constexpr size_t metaSizeLR  = metaSize * 2;
    constexpr size_t bufferSize  = CDiv( ySize + metaSizeLR, 8u );

    // Bucket for extending y
    // const uint64 bucket = ( (uint64)bucketIdx ) << 32;

    // Meta extraction
    uint64 l0, l1, r0, r1;

    // Hashing
    uint64 input [5];       // y + L + R
    uint64 output[4];       // blake3 hashed output

    static_assert( bufferSize <= sizeof( input ), "Invalid fx input buffer size." );

    blake3_hasher hasher;

    #if _DEBUG
        uint64 prevY    = bucket | yIn[pairs.left[0]];
        uint64 prevLeft = 0;
    #endif

    for( uint i = 0; i < entryCount; i++ )
    {
        const uint32 left  = pairs.left[i];
        const uint32 right = left + pairs.right[i];
        ASSERT( left < right );

        const uint64 y     = bucket | yIn[left];
        
        #if _DEBUG
            ASSERT( y >= prevY );
            ASSERT( left >= prevLeft );
            prevY    = y;
            prevLeft = left;
        #endif

        // Extract metadata
        if constexpr( metaKMultiplierIn == 1 )
        {
            l0 = reinterpret_cast<const uint32*>( metaInA )[left ];
            r0 = reinterpret_cast<const uint32*>( metaInA )[right];

            input[0] = Swap64( y  << 26 | l0 >> 6  );
            input[1] = Swap64( l0 << 58 | r0 << 26 );
        }
        else if constexpr( metaKMultiplierIn == 2 )
        {
            l0 = metaInA[left ];
            r0 = metaInA[right];

            input[0] = Swap64( y  << 26 | l0 >> 38 );
            input[1] = Swap64( l0 << 26 | r0 >> 38 );
            input[2] = Swap64( r0 << 26 );
        }
        else if constexpr( metaKMultiplierIn == 3 )
        {
            l0 = metaInA[left];
            l1 = reinterpret_cast<const uint32*>( metaInB )[left ];
            r0 = metaInA[right];
            r1 = reinterpret_cast<const uint32*>( metaInB )[right];
        
            input[0] = Swap64( y  << 26 | l0 >> 38 );
            input[1] = Swap64( l0 << 26 | l1 >> 6  );
            input[2] = Swap64( l1 << 58 | r0 >> 6  );
            input[3] = Swap64( r0 << 58 | r1 << 26 );
        }
        else if constexpr( metaKMultiplierIn == 4 )
        {
            l0 = metaInA[left ];
            l1 = metaInB[left ];
            r0 = metaInA[right];
            r1 = metaInB[right];

            input[0] = Swap64( y  << 26 | l0 >> 38 );
            input[1] = Swap64( l0 << 26 | l1 >> 38 );
            input[2] = Swap64( l1 << 26 | r0 >> 38 );
            input[3] = Swap64( r0 << 26 | r1 >> 38 );
            input[4] = Swap64( r1 << 26 );
        }

        // Hash input
        blake3_hasher_init( &hasher );
        blake3_hasher_update( &hasher, input, bufferSize );
        blake3_hasher_finalize( &hasher, (uint8_t*)output, sizeof( output ) );

        uint64 fx = Swap64( *output ) >> yShift;
        yOut[i] = (uint32)fx;

        if constexpr( tableId != TableId::Table7 )
        {
            // Store the bucket id for this y value
            bucketOut[i] = (byte)( fx >> 32 );
        }
        else
        {
            // For table 7 we don't have extra bits,
            // but we do want to be able to store per bucket,
            // in order to sort. So let's just use the high 
            // bits of the 32 bit values itself
            bucketOut[i] = (byte)( ( fx >> 26 ) & 0b111111 );
        }

        // Calculate output metadata
        if constexpr( metaKMultiplierOut == 2 && metaKMultiplierIn == 1 )
        {
            metaOutA[i] = l0 << 32 | r0;
        }
        else if constexpr ( metaKMultiplierOut == 2 && metaKMultiplierIn == 3 )
        {
            const uint64 h0 = Swap64( output[0] );
            const uint64 h1 = Swap64( output[1] );

            metaOutA[i] = h0 << ySize | h1 >> 26;
        }
        else if constexpr ( metaKMultiplierOut == 3 )
        {
            const uint64 h0 = Swap64( output[0] );
            const uint64 h1 = Swap64( output[1] );
            const uint64 h2 = Swap64( output[2] );

            metaOutA[i] = h0 << ySize | h1 >> 26;
            reinterpret_cast<uint32*>( metaOutB )[i] = (uint32)( ((h1 << 6) & 0xFFFFFFC0) | h2 >> 58 );
        }
        else if constexpr( metaKMultiplierOut == 4 && metaKMultiplierIn == 2 )
        {
            metaOutA[i] = l0;
            metaOutB[i] = r0;
        }
        else if constexpr ( metaKMultiplierOut == 4 && metaKMultiplierIn != 2 )
        {
            const uint64 h0 = Swap64( output[0] );
            const uint64 h1 = Swap64( output[1] );
            const uint64 h2 = Swap64( output[2] );

            metaOutA[i] = h0 << ySize | h1 >> 26;
            metaOutB[i] = h1 << 38    | h2 >> 26;
        }
    }
}

#pragma GCC diagnostic pop

