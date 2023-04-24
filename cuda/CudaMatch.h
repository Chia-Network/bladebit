#pragma once
#include <cuda_runtime.h>

/// Unbucketized CUDA-based matching function for k32 compressed plots.
/// This method is meant to only be used with compressed plots.
cudaError CudaHarvestMatchK32(
    struct Pair*  devOutPairs,
    uint32*       devMatchCount,
    const uint32  maxMatches,
    const uint64* devYEntries,
    const uint32  entryCount,
    const uint32  matchOffset,
    cudaStream_t  stream );

/// Unbucketized CUDA-based matching function, specifically for k32.
/// The matches are deterministic. That is, you will always get the 
/// same matches given the same input, though the order of the 
// /// stored matches is not deterministic.
// cudaError CudaMatchK32(
//     struct Pair*  devOutPairs,
//     uint32*       devMatchCount,
//     uint32*       devTempGroupIndices,
//     uint32*       devGroupIndices,
//     uint32*       devGroupCount,
//     uint32        maxGroups,
//     byte*         devSortTempData,
//     const size_t  sortTempDataSize,
//     const uint64* devYEntries,
//     const uint32  entryCount,
//     cudaStream_t  stream );
