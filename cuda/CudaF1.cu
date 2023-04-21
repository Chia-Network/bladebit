#include "CudaF1.h"
#include "CudaUtil.h"
#include "ChiaConsts.h"

/// #NOTE: Code duplicated from chacha8.cu for now.
/// #TODO: Refactor and consolidate


#define U32TO32_LITTLE(v) CuBSwap32(v)
#define U8TO32_LITTLE(p) (*(const uint32_t *)(p))
#define U32TO8_LITTLE(p, v) (((uint32_t *)(p))[0] = U32TO32_LITTLE(v))
#define ROTL32(v, n) (((v) << (n)) | ((v) >> (32 - (n))))

#define ROTATE(v, c) (ROTL32(v, c))
#define XOR(v, w) ((v) ^ (w))
#define PLUS(v, w) ((v) + (w))
#define PLUSONE(v) (PLUS((v), 1))

#define QUARTERROUND(a, b, c, d) \
    a = PLUS(a, b);              \
    d = ROTATE(XOR(d, a), 16);   \
    c = PLUS(c, d);              \
    b = ROTATE(XOR(b, c), 12);   \
    a = PLUS(a, b);              \
    d = ROTATE(XOR(d, a), 8);    \
    c = PLUS(c, d);              \
    b = ROTATE(XOR(b, c), 7)


// 128 threads per cuda block, each thread will do one chacha block
#define CHACHA_BLOCKS_PER_CUDA_BLOCK 128ull

//-----------------------------------------------------------
__global__ void chacha8_get_keystream_cuda_k32( 
    const CudaPlotInfo info,
    const uint32_t* input,
    const uint64_t  chachaBlockBase,
    uint64*         outY, 
    uint32*         outX )
{
    extern __shared__ uint32 sharedBucketCounts[];

    const uint32 id  = threadIdx.x;
    const uint32 gid = blockIdx.x * blockDim.x + id;

    const uint64_t chachaBlock = chachaBlockBase + blockIdx.x * CHACHA_BLOCKS_PER_CUDA_BLOCK + id;


    uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
    uint32_t j0, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15;

    j0  = input[0];
    j1  = input[1];
    j2  = input[2];
    j3  = input[3];
    j4  = input[4];
    j5  = input[5];
    j6  = input[6];
    j7  = input[7];
    j8  = input[8];
    j9  = input[9];
    j10 = input[10];
    j11 = input[11];
    j12 = (uint32_t)chachaBlock;
    j13 = (uint32_t)(chachaBlock >> 32);
    j14 = input[14];
    j15 = input[15];

    // #TODO: Dispatch a different kernel to set the x's
    x0  = j0;
    x1  = j1;
    x2  = j2;
    x3  = j3;
    x4  = j4;
    x5  = j5;
    x6  = j6;
    x7  = j7;
    x8  = j8;
    x9  = j9;
    x10 = j10;
    x11 = j11;
    x12 = j12;
    x13 = j13;
    x14 = j14;
    x15 = j15;

    #pragma unroll
    for( int i = 8; i > 0; i -= 2 )
    {
        QUARTERROUND( x0, x4, x8 , x12 );
        QUARTERROUND( x1, x5, x9 , x13 );
        QUARTERROUND( x2, x6, x10, x14 );
        QUARTERROUND( x3, x7, x11, x15 );
        QUARTERROUND( x0, x5, x10, x15 );
        QUARTERROUND( x1, x6, x11, x12 );
        QUARTERROUND( x2, x7, x8 , x13 );
        QUARTERROUND( x3, x4, x9 , x14 );
    }

    const uint32 x   = (uint32)(chachaBlock * 16);    // X start offset
    const uint32 out = gid * (kF1BlockSize / sizeof(uint32));

    const uint32 xo0  = x + 0 ;
    const uint32 xo1  = x + 1 ;
    const uint32 xo2  = x + 2 ;
    const uint32 xo3  = x + 3 ;
    const uint32 xo4  = x + 4 ;
    const uint32 xo5  = x + 5 ;
    const uint32 xo6  = x + 6 ;
    const uint32 xo7  = x + 7 ;
    const uint32 xo8  = x + 8 ;
    const uint32 xo9  = x + 9 ;
    const uint32 xo10 = x + 10;
    const uint32 xo11 = x + 11;
    const uint32 xo12 = x + 12;
    const uint32 xo13 = x + 13;
    const uint32 xo14 = x + 14;
    const uint32 xo15 = x + 15;

    outY[out+0 ] = (((uint64)CuBSwap32( PLUS( x0 , j0  ) )) << kExtraBits) | (xo0  >> (info.k - kExtraBits));
    outY[out+1 ] = (((uint64)CuBSwap32( PLUS( x1 , j1  ) )) << kExtraBits) | (xo1  >> (info.k - kExtraBits));
    outY[out+2 ] = (((uint64)CuBSwap32( PLUS( x2 , j2  ) )) << kExtraBits) | (xo2  >> (info.k - kExtraBits));
    outY[out+3 ] = (((uint64)CuBSwap32( PLUS( x3 , j3  ) )) << kExtraBits) | (xo3  >> (info.k - kExtraBits));
    outY[out+4 ] = (((uint64)CuBSwap32( PLUS( x4 , j4  ) )) << kExtraBits) | (xo4  >> (info.k - kExtraBits));
    outY[out+5 ] = (((uint64)CuBSwap32( PLUS( x5 , j5  ) )) << kExtraBits) | (xo5  >> (info.k - kExtraBits));
    outY[out+6 ] = (((uint64)CuBSwap32( PLUS( x6 , j6  ) )) << kExtraBits) | (xo6  >> (info.k - kExtraBits));
    outY[out+7 ] = (((uint64)CuBSwap32( PLUS( x7 , j7  ) )) << kExtraBits) | (xo7  >> (info.k - kExtraBits));
    outY[out+8 ] = (((uint64)CuBSwap32( PLUS( x8 , j8  ) )) << kExtraBits) | (xo8  >> (info.k - kExtraBits));
    outY[out+9 ] = (((uint64)CuBSwap32( PLUS( x9 , j9  ) )) << kExtraBits) | (xo9  >> (info.k - kExtraBits));
    outY[out+10] = (((uint64)CuBSwap32( PLUS( x10, j10 ) )) << kExtraBits) | (xo10 >> (info.k - kExtraBits));
    outY[out+11] = (((uint64)CuBSwap32( PLUS( x11, j11 ) )) << kExtraBits) | (xo11 >> (info.k - kExtraBits));
    outY[out+12] = (((uint64)CuBSwap32( PLUS( x12, j12 ) )) << kExtraBits) | (xo12 >> (info.k - kExtraBits));
    outY[out+13] = (((uint64)CuBSwap32( PLUS( x13, j13 ) )) << kExtraBits) | (xo13 >> (info.k - kExtraBits));
    outY[out+14] = (((uint64)CuBSwap32( PLUS( x14, j14 ) )) << kExtraBits) | (xo14 >> (info.k - kExtraBits));
    outY[out+15] = (((uint64)CuBSwap32( PLUS( x15, j15 ) )) << kExtraBits) | (xo15 >> (info.k - kExtraBits));

    outX[out+0 ] = xo0 ;
    outX[out+1 ] = xo1 ;
    outX[out+2 ] = xo2 ;
    outX[out+3 ] = xo3 ;
    outX[out+4 ] = xo4 ;
    outX[out+5 ] = xo5 ;
    outX[out+6 ] = xo6 ;
    outX[out+7 ] = xo7 ;
    outX[out+8 ] = xo8 ;
    outX[out+9 ] = xo9 ;
    outX[out+10] = xo10;
    outX[out+11] = xo11;
    outX[out+12] = xo12;
    outX[out+13] = xo13;
    outX[out+14] = xo14;
    outX[out+15] = xo15;
}

void CudaGenF1K32(
    const CudaPlotInfo& info,
    const uint32* devChaChhaInput,
    const uint64  chachaBlockBase,
    const uint32  chachaBlockCount,
          uint64* devOutY,
          uint32* devOutX,
    cudaStream_t  stream )
{
    const uint32 cuThreads = CHACHA_BLOCKS_PER_CUDA_BLOCK;
    const uint32 cuBlocks  = CDiv( chachaBlockCount, cuThreads );
        
    chacha8_get_keystream_cuda_k32<<<cuBlocks, cuThreads, 0, stream>>>(
        info,
        devChaChhaInput,
        chachaBlockBase,
        devOutY,
        devOutX
    );
}

