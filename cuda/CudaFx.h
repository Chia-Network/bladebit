#pragma once
#include <cuda_runtime.h>
#include "plotting/Tables.h"

struct Pair;
void CudaFxHarvestK32(
    TableId       table,
    uint64*       devYOut, 
    void*         devMetaOut,
          uint32  matchCount, 
    const Pair*   devPairsIn, 
    const uint64* devYIn,
    const void*   devMetaIn,
    cudaStream_t  stream );