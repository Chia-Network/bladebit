#include "CudaPlotContext.h"
#include "CudaFx.h"

#define CU_FX_THREADS_PER_BLOCK 256

#define B3Round( intputByteSize ) \
    uint32 state[16] = {                      \
        0x6A09E667UL,   /*IV full*/           \
        0xBB67AE85UL,                         \
        0x3C6EF372UL,                         \
        0xA54FF53AUL,                         \
        0x510E527FUL,                         \
        0x9B05688CUL,                         \
        0x1F83D9ABUL,                         \
        0x5BE0CD19UL,                         \
        0x6A09E667UL,   /*IV 0-4*/            \
        0xBB67AE85UL,                         \
        0x3C6EF372UL,                         \
        0xA54FF53AUL,                         \
        0,               /*count lo*/         \
        0,               /*count hi*/         \
        (intputByteSize),/*buffer length*/    \
        11              /*flags. Always 11*/  \
    };                                        \
                                              \
    round_fn( state, (uint32*)&input[0], 0 ); \
    round_fn( state, (uint32*)&input[0], 1 ); \
    round_fn( state, (uint32*)&input[0], 2 ); \
    round_fn( state, (uint32*)&input[0], 3 ); \
    round_fn( state, (uint32*)&input[0], 4 ); \
    round_fn( state, (uint32*)&input[0], 5 ); \
    round_fn( state, (uint32*)&input[0], 6 ); 

__forceinline__ __device__ uint32_t rotr32( uint32_t w, uint32_t c )
{
    return ( w >> c ) | ( w << ( 32 - c ) );
}

__forceinline__ __device__ void g( uint32_t* state, size_t a, size_t b, size_t c, size_t d, uint32_t x, uint32_t y )
{
    state[a] = state[a] + state[b] + x;
    state[d] = rotr32( state[d] ^ state[a], 16 );
    state[c] = state[c] + state[d];
    state[b] = rotr32( state[b] ^ state[c], 12 );
    state[a] = state[a] + state[b] + y;
    state[d] = rotr32( state[d] ^ state[a], 8 );
    state[c] = state[c] + state[d];
    state[b] = rotr32( state[b] ^ state[c], 7 );
}

__forceinline__ __device__ void round_fn( uint32_t state[16], const uint32_t* msg, size_t round )
{
    static const uint8_t MSG_SCHEDULE[7][16] = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
        {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
        {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
        {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
        {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
        {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
    };
    // Select the message schedule based on the round.
    const uint8_t* schedule = MSG_SCHEDULE[round];

    // Mix the columns.
    g( state, 0, 4, 8 , 12, msg[schedule[0]], msg[schedule[1]] );
    g( state, 1, 5, 9 , 13, msg[schedule[2]], msg[schedule[3]] );
    g( state, 2, 6, 10, 14, msg[schedule[4]], msg[schedule[5]] );
    g( state, 3, 7, 11, 15, msg[schedule[6]], msg[schedule[7]] );

    // Mix the rows.
    g( state, 0, 5, 10, 15, msg[schedule[8]] , msg[schedule[9]] );
    g( state, 1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]] );
    g( state, 2, 7, 8 , 13, msg[schedule[12]], msg[schedule[13]] );
    g( state, 3, 4, 9 , 14, msg[schedule[14]], msg[schedule[15]] );
}


//-----------------------------------------------------------
__forceinline__ __device__ void Blake3RunRounds( uint64 input[8], const uint32 intputByteSize )
{
    uint32 state[16] = {                     
        0x6A09E667UL,   // IV full           
        0xBB67AE85UL,                        
        0x3C6EF372UL,                        
        0xA54FF53AUL,                        
        0x510E527FUL,                        
        0x9B05688CUL,                        
        0x1F83D9ABUL,                        
        0x5BE0CD19UL,                        
        0x6A09E667UL,   // IV 0-4            
        0xBB67AE85UL,                        
        0x3C6EF372UL,                        
        0xA54FF53AUL,                        
        0,              // count lo          
        0,              // count hi          
        intputByteSize, // buffer length     
        11              // flags. Always 11  
    };                                       
                                             
    round_fn( state, (uint32*)&input[0], 0 );
    round_fn( state, (uint32*)&input[0], 1 );
    round_fn( state, (uint32*)&input[0], 2 );
    round_fn( state, (uint32*)&input[0], 3 );
    round_fn( state, (uint32*)&input[0], 4 );
    round_fn( state, (uint32*)&input[0], 5 );
    round_fn( state, (uint32*)&input[0], 6 );
}

//-----------------------------------------------------------
__global__ void ValidatePairs( const uint32 matchCount, const uint32 entryCount, const Pair* pairs )
{
    const uint32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if( gid >= matchCount)
        return;

    const Pair p = pairs[gid];
    CUDA_ASSERT( p.left  < entryCount );
    CUDA_ASSERT( p.right < entryCount );
}

enum class FxVariant
{
    Regular = 0,
    InlineTable1,
    Compressed
};

//-----------------------------------------------------------
template<TableId rTable>
__global__ void HarvestFxK32Kernel( 
    uint64*       yOut, 
    void*         metaOutVoid,
    const uint32  matchCount, 
    const Pair*   pairsIn, 
    const uint64* yIn,
    const void*   metaInVoid
)
{
    const uint32 id  = threadIdx.x;
    const uint32 gid = (uint32)(blockIdx.x * blockDim.x + id);

    if( gid >= matchCount )
        return;

    using TMetaIn  = typename K32MetaType<rTable>::In;
    using TMetaOut = typename K32MetaType<rTable>::Out;

    constexpr size_t MetaInMulti  = TableMetaIn <rTable>::Multiplier;
    constexpr size_t MetaOutMulti = TableMetaOut<rTable>::Multiplier;


    const uint32 k           = BBCU_K;
    const uint32 ySize       = k + kExtraBits;
    const uint32 yExtraBits  = MetaOutMulti == 0 ? 0 : kExtraBits;
    const uint32 yShift      = 64 - (k + yExtraBits);
    
    const uint32 metaSize    = k * MetaInMulti;
    const uint32 metaSizeLR  = metaSize * 2;
    const uint32 inputSize   = CuCDiv( ySize + metaSizeLR, 8 );


    //  uint64 yMask   = (1ull << (k+yExtraBits)) - 1;

    // const uint32 matchCount = *pMatchCount;

    const TMetaIn*  metaIn  = (TMetaIn*)metaInVoid;
          TMetaOut* metaOut = (TMetaOut*)metaOutVoid;

    // Gen fx and meta
    uint64   oy;
    TMetaOut ometa;
    
    {
        uint64 input [8];
        uint64 output[4];

        const Pair pair = pairsIn[gid];
        
        // CUDA_ASSERT( pair.left  < entryCount );
        // CUDA_ASSERT( pair.right < entryCount );

        const uint64 y = yIn[pair.left];

        if constexpr( MetaInMulti == 1 )
        {
            const uint64 l = metaIn[pair.left ];
            const uint64 r = metaIn[pair.right];

            const uint64 i0 = y << 26 | l >> 6;
            const uint64 i1 = l << 58 | r << 26;

            input[0] = CuBSwap64( i0 );
            input[1] = CuBSwap64( i1 );
            input[2] = 0;
            input[3] = 0;
            input[4] = 0;
            input[5] = 0;
            input[6] = 0;
            input[7] = 0;

            if constexpr( MetaOutMulti == 2 )
                ometa = l << 32 | r;
        }
        else if constexpr ( MetaInMulti == 2 )
        {
            const uint64 l = metaIn[pair.left ];
            const uint64 r = metaIn[pair.right];

            input[0] = CuBSwap64( y << 26 | l >> 38 );
            input[1] = CuBSwap64( l << 26 | r >> 38 );
            input[2] = CuBSwap64( r << 26 );
            input[3] = 0;
            input[4] = 0;
            input[5] = 0;
            input[6] = 0;
            input[7] = 0;

            if constexpr ( MetaOutMulti == 4 )
            {
                ometa.m0 = l;
                ometa.m1 = r;
            }
        }
        else if constexpr ( MetaInMulti == 3 )
        {
            const uint64 l0 = metaIn[pair.left ].m0;
            const uint64 l1 = metaIn[pair.left ].m1 & 0xFFFFFFFF;
            const uint64 r0 = metaIn[pair.right].m0;
            const uint64 r1 = metaIn[pair.right].m1 & 0xFFFFFFFF;
            
            input[0] = CuBSwap64( y  << 26 | l0 >> 38 );
            input[1] = CuBSwap64( l0 << 26 | l1 >> 6  );
            input[2] = CuBSwap64( l1 << 58 | r0 >> 6  );
            input[3] = CuBSwap64( r0 << 58 | r1 << 26 );
            input[4] = 0;
            input[5] = 0;
            input[6] = 0;
            input[7] = 0;
        }
        else if constexpr ( MetaInMulti == 4 )
        {
            const K32Meta4 l = metaIn[pair.left ];
            const K32Meta4 r = metaIn[pair.right];

            input[0] = CuBSwap64( y    << 26 | l.m0 >> 38 );
            input[1] = CuBSwap64( l.m0 << 26 | l.m1 >> 38 );
            input[2] = CuBSwap64( l.m1 << 26 | r.m0 >> 38 );
            input[3] = CuBSwap64( r.m0 << 26 | r.m1 >> 38 );
            input[4] = CuBSwap64( r.m1 << 26 );
            input[5] = 0;
            input[6] = 0;
            input[7] = 0;
        }

        B3Round( inputSize );

        uint32* out = (uint32*)output;
        out[0] = state[0] ^ state[8];
        out[1] = state[1] ^ state[9];

        oy = CuBSwap64( *output ) >> yShift;

        // Save output metadata
        if constexpr ( MetaOutMulti == 2 && MetaInMulti == 3 )
        {
            out[2] = state[2] ^ state[10];
            out[3] = state[3] ^ state[11];

            const uint64 h0 = CuBSwap64( output[0] );
            const uint64 h1 = CuBSwap64( output[1] );

            ometa = h0 << ySize | h1 >> 26;
        }
        else if constexpr ( MetaOutMulti == 3 )
        {
            out[2] = state[2] ^ state[10];
            out[3] = state[3] ^ state[11];
            out[4] = state[4] ^ state[12];
            out[5] = state[5] ^ state[13];

            const uint64 h0 = CuBSwap64( output[0] );
            const uint64 h1 = CuBSwap64( output[1] );
            const uint64 h2 = CuBSwap64( output[2] );

            ometa.m0 = h0 << ySize | h1 >> 26;
            ometa.m1 = ((h1 << 6) & 0xFFFFFFC0) | h2 >> 58;
        }
        else if constexpr ( MetaOutMulti == 4 && MetaInMulti != 2 )
        {
            out[2] = state[2] ^ state[10];
            out[3] = state[3] ^ state[11];
            out[4] = state[4] ^ state[12];
            out[5] = state[5] ^ state[13];

            const uint64 h0 = CuBSwap64( output[0] );
            const uint64 h1 = CuBSwap64( output[1] );
            const uint64 h2 = CuBSwap64( output[2] );

            ometa.m0 = h0 << ySize | h1 >> 26;
            ometa.m1 = h1 << 38    | h2 >> 26;
        }
    }

    // OK to store the value now
    yOut[gid] = oy;

    if constexpr ( MetaOutMulti > 0 )
        metaOut[gid] = ometa;
}

//-----------------------------------------------------------
template<FxVariant Variant, TableId rTable>
__global__ void GenFxCuda( const uint32* pMatchCount, const uint64 bucketMask, const Pair* pairs, const uint32* yIn, const void* metaInVoid,
                           uint32* yOut, void* metaOutVoid, const uint32 pairsOffset, uint32* pairsOutL, uint16* pairsOutR, uint32* globalBucketCounts,
                           const Pair* inlinedXPairs
#if _DEBUG
, const uint32 entryCount
#endif
)
{
    using TMetaIn     = typename K32MetaType<rTable>::In;
    using TMetaOut    = typename K32MetaType<rTable>::Out;

    constexpr size_t MetaInMulti  = TableMetaIn <rTable>::Multiplier;
    constexpr size_t MetaOutMulti = TableMetaOut<rTable>::Multiplier;

    constexpr uint32 yMask   = MetaOutMulti == 0 ? BBC_Y_MASK_T7 : BBC_Y_MASK;

    const uint32 k           = BBCU_K;
    const uint32 metaSize    = k * MetaInMulti;
    const uint32 metaSizeLR  = metaSize * 2;
    const uint32 inputSize   = CuCDiv( 38 + metaSizeLR, 8 );

    const uint32 yShiftBits  = MetaOutMulti == 0 ? 0 : kExtraBits;
    const uint32 ySize       = k + kExtraBits;
    const uint32 yShift      = 64 - (k + yShiftBits);
    const uint32 bucketShift = MetaOutMulti == 0 ? BBC_BUCKET_SHIFT_T7 : BBC_BUCKET_SHIFT;

    const uint32 id  = threadIdx.x;
    const uint32 gid = (uint32)(blockIdx.x * blockDim.x + id);

    const uint32 matchCount = *pMatchCount;

    const TMetaIn*  metaIn  = (TMetaIn*)metaInVoid;
          TMetaOut* metaOut = (TMetaOut*)metaOutVoid;

    CUDA_ASSERT( BBCU_BUCKET_COUNT <= CU_FX_THREADS_PER_BLOCK );
    __shared__ uint32 sharedBucketCounts[BBCU_BUCKET_COUNT];
    if( id < BBCU_BUCKET_COUNT )
        sharedBucketCounts[id] = 0;

    __syncthreads();

    // Gen fx and meta
    uint64   oy;
    TMetaOut ometa;
    uint32   offset;
    uint32   bucket;
    Pair     pair;

    if( gid < matchCount )
    {
        uint64 input [8];
        uint64 output[4];

        pair = pairs[gid];
        
        CUDA_ASSERT( pair.left  < entryCount );
        CUDA_ASSERT( pair.right < entryCount );

        const uint64 y = bucketMask | yIn[pair.left];
            
        if constexpr( MetaInMulti == 1 )
        {
            const uint64 l = metaIn[pair.left ];
            const uint64 r = metaIn[pair.right];

            const uint64 i0 = y << 26 | l >> 6;
            const uint64 i1 = l << 58 | r << 26;

            input[0] = CuBSwap64( i0 );
            input[1] = CuBSwap64( i1 );
            input[2] = 0;
            input[3] = 0;
            input[4] = 0;
            input[5] = 0;
            input[6] = 0;
            input[7] = 0;

            if constexpr( MetaOutMulti == 2 )
                ometa = l << 32 | r;
        }
        else if constexpr ( MetaInMulti == 2 )
        {
            const uint64 l = metaIn[pair.left ];
            const uint64 r = metaIn[pair.right];

            input[0] = CuBSwap64( y << 26 | l >> 38 );
            input[1] = CuBSwap64( l << 26 | r >> 38 );
            input[2] = CuBSwap64( r << 26 );
            input[3] = 0;
            input[4] = 0;
            input[5] = 0;
            input[6] = 0;
            input[7] = 0;

            if constexpr ( MetaOutMulti == 4 )
            {
                ometa.m0 = l;
                ometa.m1 = r;
            }
        }
        else if constexpr ( MetaInMulti == 3 )
        {
            const uint64 l0 = metaIn[pair.left ].m0;
            const uint64 l1 = metaIn[pair.left ].m1 & 0xFFFFFFFF;
            const uint64 r0 = metaIn[pair.right].m0;
            const uint64 r1 = metaIn[pair.right].m1 & 0xFFFFFFFF;
            
            input[0] = CuBSwap64( y  << 26 | l0 >> 38 );
            input[1] = CuBSwap64( l0 << 26 | l1 >> 6  );
            input[2] = CuBSwap64( l1 << 58 | r0 >> 6  );
            input[3] = CuBSwap64( r0 << 58 | r1 << 26 );
            input[4] = 0;
            input[5] = 0;
            input[6] = 0;
            input[7] = 0;
        }
        else if constexpr ( MetaInMulti == 4 )
        {
            const K32Meta4 l = metaIn[pair.left ];
            const K32Meta4 r = metaIn[pair.right];

            input[0] = CuBSwap64( y    << 26 | l.m0 >> 38 );
            input[1] = CuBSwap64( l.m0 << 26 | l.m1 >> 38 );
            input[2] = CuBSwap64( l.m1 << 26 | r.m0 >> 38 );
            input[3] = CuBSwap64( r.m0 << 26 | r.m1 >> 38 );
            input[4] = CuBSwap64( r.m1 << 26 );
            input[5] = 0;
            input[6] = 0;
            input[7] = 0;
        }

        B3Round( inputSize );

        uint32* out = (uint32*)output;
        out[0] = state[0] ^ state[8];
        out[1] = state[1] ^ state[9];

        oy = CuBSwap64( *output ) >> yShift;
        
        // Save output metadata
        if constexpr ( MetaOutMulti == 2 && MetaInMulti == 3 )
        {
            out[2] = state[2] ^ state[10];
            out[3] = state[3] ^ state[11];

            const uint64 h0 = CuBSwap64( output[0] );
            const uint64 h1 = CuBSwap64( output[1] );

            ometa = h0 << ySize | h1 >> 26;
        }
        else if constexpr ( MetaOutMulti == 3 )
        {
            out[2] = state[2] ^ state[10];
            out[3] = state[3] ^ state[11];
            out[4] = state[4] ^ state[12];
            out[5] = state[5] ^ state[13];

            const uint64 h0 = CuBSwap64( output[0] );
            const uint64 h1 = CuBSwap64( output[1] );
            const uint64 h2 = CuBSwap64( output[2] );

            ometa.m0 = h0 << ySize | h1 >> 26;
            ometa.m1 = ((h1 << 6) & 0xFFFFFFC0) | h2 >> 58;
        }
        else if constexpr ( MetaOutMulti == 4 && MetaInMulti != 2 )
        {
            out[2] = state[2] ^ state[10];
            out[3] = state[3] ^ state[11];
            out[4] = state[4] ^ state[12];
            out[5] = state[5] ^ state[13];

            const uint64 h0 = CuBSwap64( output[0] );
            const uint64 h1 = CuBSwap64( output[1] );
            const uint64 h2 = CuBSwap64( output[2] );

            ometa.m0 = h0 << ySize | h1 >> 26;
            ometa.m1 = h1 << 38    | h2 >> 26;
        }

        // Save local offset in the target bucket
        bucket = oy >> bucketShift;
        CUDA_ASSERT( bucket < BBCU_BUCKET_COUNT );
        
        // Record local offset to the shared bucket count
        offset = atomicAdd( &sharedBucketCounts[bucket], 1 );
        CUDA_ASSERT( offset < CU_FX_THREADS_PER_BLOCK );
    }

    // Store this block's bucket offset into the global bucket counts,
    // and get our global offset for that particular bucket
    __syncthreads();

    if( id < BBCU_BUCKET_COUNT )
        sharedBucketCounts[id] = atomicAdd( &globalBucketCounts[id], sharedBucketCounts[id] );

    __syncthreads();

    if( gid >= matchCount )
        return;

    /// Distribute
    // Now we have our global offset within a bucket.
    // Since all bucket slices are fixed-size, we don't need to calculate a prefix sum.
    // We can simply store directly in the slice address
    // #TODO: Perhaps we just cap the entries per slices here, instead of allocating more memory...
    const uint32 offsetInSlice   = sharedBucketCounts[bucket] + offset;
    const uint32 sliceOffsetY    = bucket * (uint32)BBCU_MAX_SLICE_ENTRY_COUNT;
    // const uint32 sliceOffsetMeta = sliceOffsetY * ( BBCU_HOST_META_MULTIPLIER / ( sizeof( TMetaOut ) / sizeof( uint32 ) ) );
    const uint32 dstY            = sliceOffsetY    + offsetInSlice;
    // const uint32 dstMeta         = sliceOffsetMeta + offsetInSlice;


#if _DEBUG
    if( offsetInSlice >= BBCU_MAX_SLICE_ENTRY_COUNT )
    {
        printf( "[%u] (bucket %u) Y Offset %u (local: %u) (in-slice: %u/%u) is out of range %u.\n", gid, bucket, dstY, offset, offsetInSlice, (uint32)BBCU_MAX_SLICE_ENTRY_COUNT, (uint32)BBCU_BUCKET_ALLOC_ENTRY_COUNT );
    }
#endif

    // OK to store the value now
    yOut[dstY] = (uint32)oy & yMask;

    if constexpr ( Variant == FxVariant::Regular )
    {
        pairsOutL[dstY] = pairsOffset + pair.left;
        pairsOutR[dstY] = (uint16)(pair.right - pair.left);
    }
    else if constexpr( Variant == FxVariant::InlineTable1 )
    {
        // Inlined x's
        ((Pair*)pairsOutL)[dstY] = inlinedXPairs[gid];
    }
    else if constexpr ( Variant == FxVariant::Compressed )
    {
        // Compressed x's
        pairsOutL[dstY] = ((uint32*)inlinedXPairs)[gid];
    }

    if constexpr ( MetaOutMulti > 0 )
        metaOut[dstY] = ometa;
}

//-----------------------------------------------------------
template<FxVariant Variant>
inline void GenFxForTable( CudaK32PlotContext& cx, const uint32* devYIn, const uint32* devMetaIn, cudaStream_t stream )
{
    const uint32 cudaBlockCount = CDiv( BBCU_BUCKET_ALLOC_ENTRY_COUNT, CU_FX_THREADS_PER_BLOCK );
    const uint64 bucketMask     = BBC_BUCKET_MASK( cx.bucket );

    const bool isCompressed = (uint32)cx.table <= cx.gCfg->numDroppedTables;
    const bool isPairs      = !isCompressed && (uint32)cx.table == cx.gCfg->numDroppedTables+1;
    

#if _DEBUG
    #define DBG_FX_INPUT_ENTRY_COUNT ,cx.bucketCounts[(int)cx.table-1][cx.bucket]
#else 
    #define DBG_FX_INPUT_ENTRY_COUNT
#endif

    // Get next download buffer
    uint32* devYOut      = (uint32*)cx.yOut.LockDeviceBuffer( stream );

    uint32* devPairsLOut = nullptr;
    uint16* devPairsROut = nullptr;

    if( isPairs )
    {
        devPairsLOut = (uint32*)cx.xPairsOut.LockDeviceBuffer( stream );
        devPairsROut = nullptr;
    }
    else
    {
        devPairsLOut = (uint32*)cx.pairsLOut.LockDeviceBuffer( stream );

        if( !isCompressed )
            devPairsROut = (uint16*)cx.pairsROut.LockDeviceBuffer( stream );
    }

    void* devMetaOut = cx.table < TableId::Table7 ? cx.metaOut.LockDeviceBuffer( stream ) : nullptr;

    uint32* devBucketCounts = cx.devSliceCounts + cx.bucket * BBCU_BUCKET_COUNT;

    #define FX_CUDA_ARGS cx.devMatchCount, bucketMask, cx.devMatches, devYIn, devMetaIn, \
         devYOut, devMetaOut, cx.prevTablePairOffset, devPairsLOut, devPairsROut, \
         devBucketCounts, cx.devInlinedXs DBG_FX_INPUT_ENTRY_COUNT
// return;
    switch( cx.table )
    {
        case TableId::Table2: GenFxCuda<Variant, TableId::Table2><<<cudaBlockCount, CU_FX_THREADS_PER_BLOCK, 0, stream>>>( FX_CUDA_ARGS ); break;
        case TableId::Table3: GenFxCuda<Variant, TableId::Table3><<<cudaBlockCount, CU_FX_THREADS_PER_BLOCK, 0, stream>>>( FX_CUDA_ARGS ); break;
        case TableId::Table4: GenFxCuda<Variant, TableId::Table4><<<cudaBlockCount, CU_FX_THREADS_PER_BLOCK, 0, stream>>>( FX_CUDA_ARGS ); break;
        case TableId::Table5: GenFxCuda<Variant, TableId::Table5><<<cudaBlockCount, CU_FX_THREADS_PER_BLOCK, 0, stream>>>( FX_CUDA_ARGS ); break;
        case TableId::Table6: GenFxCuda<Variant, TableId::Table6><<<cudaBlockCount, CU_FX_THREADS_PER_BLOCK, 0, stream>>>( FX_CUDA_ARGS ); break;
        case TableId::Table7: GenFxCuda<Variant, TableId::Table7><<<cudaBlockCount, CU_FX_THREADS_PER_BLOCK, 0, stream>>>( FX_CUDA_ARGS ); break;
    }

    #undef FX_CUDA_ARGS
    #undef DBG_FX_INPUT_ENTRY_COUNT
}

//-----------------------------------------------------------
void GenFx( CudaK32PlotContext& cx, const uint32* devYIn, const uint32* devMetaIn, cudaStream_t stream )
{
    const bool isCompressed = (uint32)cx.table <= cx.gCfg->numDroppedTables;

    if( !isCompressed )
    {
        const bool isPairs = (uint32)cx.table == cx.gCfg->numDroppedTables+1;

        if( isPairs )
            GenFxForTable<FxVariant::InlineTable1>( cx, devYIn, devMetaIn, stream );
        else
            GenFxForTable<FxVariant::Regular>( cx, devYIn, devMetaIn, stream );
    }
    else
        GenFxForTable<FxVariant::Compressed>( cx, devYIn, devMetaIn, stream );
}

//-----------------------------------------------------------
void CudaFxHarvestK32(
    const TableId table,
    uint64*       devYOut, 
    void*         devMetaOut,
    const uint32  matchCount, 
    const Pair*   devPairsIn, 
    const uint64* devYIn,
    const void*   devMetaIn,
    cudaStream_t  stream )
{
    ASSERT( devYIn );
    ASSERT( devMetaIn );
    ASSERT( devYOut );
    ASSERT( table == TableId::Table7 || devMetaOut );
    ASSERT( devPairsIn );
    ASSERT( matchCount );

    const uint32 kthreads = 256;
    const uint32 kblocks  = CDiv( matchCount, kthreads );

    #define KERN_ARGS devYOut, devMetaOut, matchCount, devPairsIn, devYIn, devMetaIn
    #undef KERN_ARG

    switch( table )
    {
        case TableId::Table2:
            HarvestFxK32Kernel<TableId::Table2><<<kblocks, kthreads, 0, stream>>>( KERN_ARGS );
            break;
        case TableId::Table3:
            HarvestFxK32Kernel<TableId::Table3><<<kblocks, kthreads, 0, stream>>>( KERN_ARGS );
            break;
        case TableId::Table4:
            HarvestFxK32Kernel<TableId::Table4><<<kblocks, kthreads, 0, stream>>>( KERN_ARGS );
            break;
        case TableId::Table5:
            HarvestFxK32Kernel<TableId::Table5><<<kblocks, kthreads, 0, stream>>>( KERN_ARGS );
            break;
        case TableId::Table6:
            HarvestFxK32Kernel<TableId::Table6><<<kblocks, kthreads, 0, stream>>>( KERN_ARGS );
            break;
        case TableId::Table7:
            HarvestFxK32Kernel<TableId::Table7><<<kblocks, kthreads, 0, stream>>>( KERN_ARGS );
            break;
    
        default:
            Panic( "Unexpected table.");
            break;
    }
}