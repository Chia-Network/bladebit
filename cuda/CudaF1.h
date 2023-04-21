#pragma once

#include <cuda_runtime.h>

struct CudaPlotInfo;

void CudaGenF1K32(
    const CudaPlotInfo& info,
    const uint32* devChaChhaInput,
    const uint64  chachaBlockBase,
    const uint32  chachaBlockCount,
          uint64* devOutY,
          uint32* devOutX,
    cudaStream_t  stream );
