#pragma once

#define FSE_STATIC_LINKING_ONLY 1
#include "fse/fse.h"
#include "fse/bitstream.h"
#undef FSE_STATIC_LINKING_ONLY

#include "CudaPlotContext.h"

#ifdef _WIN32
__pragma( pack( push, 1 ) )
typedef struct { U16 v; } unalign16;
typedef struct { U32 v; } unalign32;
typedef struct { U64 v; } unalign64;
typedef struct { size_t v; } unalignArch;
__pragma( pack( pop ) )
#endif

__constant__ unsigned CUDA_FSE_BIT_mask[32];

#define CU_FSE_PREFIX(name) FSE_error_##name
#define CU_FSE_ERROR(name) ((size_t)-CU_FSE_PREFIX(name))

__device__ __forceinline__ unsigned CUDA_ERR_isError(size_t code) { return (code > CU_FSE_ERROR(maxCode)); }
__device__ __forceinline__ unsigned CUDA_FSE_isError(size_t code) { return CUDA_ERR_isError(code); }


__device__ __forceinline__ U16 CUDA_MEM_read16(const void* ptr) { return ((const unalign16*)ptr)->v; }

__device__ __forceinline__ void CUDA_MEM_writeLEST(void* memPtr, size_t val) { ((unalign64*)memPtr)->v = (U64)val; }

__device__ __forceinline__ void CUDA_BIT_addBits(BIT_CStream_t* bitC, size_t value, unsigned nbBits)
{
    CUDA_ASSERT(BIT_MASK_SIZE == 32);
    CUDA_ASSERT(nbBits < BIT_MASK_SIZE);
    CUDA_ASSERT(nbBits + bitC->bitPos < sizeof(bitC->bitContainer) * 8);

    bitC->bitContainer |= (value & CUDA_FSE_BIT_mask[nbBits]) << bitC->bitPos;
    bitC->bitPos += nbBits;
}

__device__ __forceinline__ void CUDA_BIT_addBitsFast(BIT_CStream_t* bitC, size_t value, unsigned nbBits)
{
    CUDA_ASSERT((value>>nbBits) == 0);
    CUDA_ASSERT(nbBits + bitC->bitPos < sizeof(bitC->bitContainer) * 8);
    bitC->bitContainer |= value << bitC->bitPos;
    bitC->bitPos += nbBits;
}

__device__ __forceinline__ void CUDA_BIT_flushBits(BIT_CStream_t* bitC)
{
    size_t const nbBytes = bitC->bitPos >> 3;
    CUDA_ASSERT(bitC->bitPos < sizeof(bitC->bitContainer) * 8);
    CUDA_MEM_writeLEST(bitC->ptr, bitC->bitContainer);
    bitC->ptr += nbBytes;
    if (bitC->ptr > bitC->endPtr) bitC->ptr = bitC->endPtr;
    bitC->bitPos &= 7;
    bitC->bitContainer >>= nbBytes*8;
}

__device__ __forceinline__ void CUDA_BIT_flushBitsFast(BIT_CStream_t* bitC)
{
    size_t const nbBytes = bitC->bitPos >> 3;
    CUDA_ASSERT(bitC->bitPos < sizeof(bitC->bitContainer) * 8);
    CUDA_MEM_writeLEST(bitC->ptr, bitC->bitContainer);
    bitC->ptr += nbBytes;
    CUDA_ASSERT(bitC->ptr <= bitC->endPtr);
    bitC->bitPos &= 7;
    bitC->bitContainer >>= nbBytes*8;
}

__device__ __forceinline__ size_t CUDA_BIT_closeCStream(BIT_CStream_t* bitC)
{
    CUDA_BIT_addBitsFast(bitC, 1, 1);   /* endMark */
    CUDA_BIT_flushBits(bitC);
    if (bitC->ptr >= bitC->endPtr) return 0; /* overflow detected */
    return (bitC->ptr - bitC->startPtr) + (bitC->bitPos > 0);
}

__device__ __forceinline__ size_t CUDA_BIT_initCStream(BIT_CStream_t* bitC, void* startPtr, size_t dstCapacity)
{
    bitC->bitContainer = 0;
    bitC->bitPos = 0;
    bitC->startPtr = (char*)startPtr;
    bitC->ptr = bitC->startPtr;
    bitC->endPtr = bitC->startPtr + dstCapacity - sizeof(bitC->bitContainer);
    if (dstCapacity <= sizeof(bitC->bitContainer)) return CU_FSE_ERROR(dstSize_tooSmall);
    return 0;
}

__device__ __forceinline__ void CUDA_FSE_initCState(FSE_CState_t* statePtr, const FSE_CTable* ct)
{
    const void* ptr = ct;
    const U16* u16ptr = (const U16*) ptr;
    const U32 tableLog = CUDA_MEM_read16(ptr);
    statePtr->value = (ptrdiff_t)1<<tableLog;
    statePtr->stateTable = u16ptr+2;
    statePtr->symbolTT = ((const U32*)ct + 1 + (tableLog ? (1<<(tableLog-1)) : 1));
    statePtr->stateLog = tableLog;
}

__device__ __forceinline__ void CUDA_FSE_initCState2(FSE_CState_t* statePtr, const FSE_CTable* ct, U32 symbol)
{
    CUDA_FSE_initCState(statePtr, ct);
    {   const FSE_symbolCompressionTransform symbolTT = ((const FSE_symbolCompressionTransform*)(statePtr->symbolTT))[symbol];
        const U16* stateTable = (const U16*)(statePtr->stateTable);
        U32 nbBitsOut  = (U32)((symbolTT.deltaNbBits + (1<<15)) >> 16);
        statePtr->value = (nbBitsOut << 16) - symbolTT.deltaNbBits;
        statePtr->value = stateTable[(statePtr->value >> nbBitsOut) + symbolTT.deltaFindState];
    }
}

__device__ __forceinline__ void CUDA_FSE_encodeSymbol(BIT_CStream_t* bitC, FSE_CState_t* statePtr, U32 symbol)
{
    FSE_symbolCompressionTransform const symbolTT = ((const FSE_symbolCompressionTransform*)(statePtr->symbolTT))[symbol];
    const U16* const stateTable = (const U16*)(statePtr->stateTable);
    U32 const nbBitsOut  = (U32)((statePtr->value + symbolTT.deltaNbBits) >> 16);
    CUDA_BIT_addBits(bitC, statePtr->value, nbBitsOut);
    statePtr->value = stateTable[ (statePtr->value >> nbBitsOut) + symbolTT.deltaFindState];
}

__device__ __forceinline__ void CUDA_FSE_flushCState(BIT_CStream_t* bitC, const FSE_CState_t* statePtr)
{
    CUDA_BIT_addBits(bitC, statePtr->value, statePtr->stateLog);
    CUDA_BIT_flushBits(bitC);
}

template<int32 EntryCount>
__device__ size_t CUDA_FSE_compress_usingCTable(
    void* dst, size_t dstSize,
    const void* src, size_t srcSize,
    const FSE_CTable* ct )
{
    const byte* const istart = (const byte*) src;
    const byte* const iend = istart + srcSize;
    const byte* ip=iend;

    BIT_CStream_t bitC;
    FSE_CState_t CState1, CState2;

    /* init */
    CUDA_ASSERT( srcSize > 2 );
    CUDA_ASSERT( srcSize == (size_t)EntryCount );
    CUDA_ASSERT( (uintptr_t)(ip - istart) == (uintptr_t)EntryCount );

    // if (srcSize <= 2) return 0;
    { 
        size_t const initError = CUDA_BIT_initCStream(&bitC, dst, dstSize);
        CUDA_ASSERT( !CUDA_FSE_isError(initError) );
        
        #if _DEBUG
            // if (FSE_isError(initError)) 
            //     return 0; /* not enough space available to write a bitstream */ 
        #endif
    }

    #define FSE_FLUSHBITS(s)  CUDA_BIT_flushBitsFast(s)

    // if (srcSize & 1) 
    {
        CUDA_FSE_initCState2(&CState1, ct, *--ip);
        CUDA_FSE_initCState2(&CState2, ct, *--ip);
        CUDA_FSE_encodeSymbol(&bitC, &CState1, *--ip);
        FSE_FLUSHBITS(&bitC);
    } 
    // else {
    //     CUDA_FSE_initCState2(&CState2, ct, *--ip);
    //     CUDA_FSE_initCState2(&CState1, ct, *--ip);
    // }

    /* join to mod 4 */
    srcSize -= 2;
    if ((sizeof(bitC.bitContainer)*8 > FSE_MAX_TABLELOG*4+7 ) && (srcSize & 2)) {  /* test bit 2 */
        CUDA_FSE_encodeSymbol(&bitC, &CState2, *--ip);
        CUDA_FSE_encodeSymbol(&bitC, &CState1, *--ip);
        FSE_FLUSHBITS(&bitC);
    }

    /* 2 or 4 encoding per loop */
    // while ( ip>istart ) 
    #pragma unroll
    for( int32 i = 0; i < EntryCount / 4; i ++ )
    {
        CUDA_FSE_encodeSymbol(&bitC, &CState2, *--ip);

        // if constexpr (sizeof(bitC.bitContainer)*8 < FSE_MAX_TABLELOG*2+7 )   /* this test must be static */
        //     FSE_FLUSHBITS(&bitC);

        CUDA_FSE_encodeSymbol(&bitC, &CState1, *--ip);

        // if constexpr (sizeof(bitC.bitContainer)*8 > FSE_MAX_TABLELOG*4+7 ) {  /* this test must be static */
            CUDA_FSE_encodeSymbol(&bitC, &CState2, *--ip);
            CUDA_FSE_encodeSymbol(&bitC, &CState1, *--ip);
        // }

        FSE_FLUSHBITS(&bitC);
    }

    CUDA_FSE_flushCState(&bitC, &CState2);
    CUDA_FSE_flushCState(&bitC, &CState1);

    #undef FSE_FLUSHBITS
    return CUDA_BIT_closeCStream(&bitC);
}
