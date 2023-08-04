#pragma once
#include "CudaPlotContext.h"

typedef unsigned FSE_CTable;

void InitFSEBitMask( struct CudaK32PlotContext& cx );

void CompressC3ParksInGPU( const uint32 parkCount, uint32* devF7, byte* devParkBuffer,
                           size_t parkBufSize, const FSE_CTable* cTable, cudaStream_t stream );

void SerializePark7InGPU( const uint32 parkCount, const uint32* indices, uint64* fieldWriter,
                          const size_t parkFieldCount, cudaStream_t stream );

void CompressToParkInGPU( const uint32 parkCount, const size_t parkSize, 
    uint64* devLinePoints, byte* devParkBuffer, size_t parkBufferSize, 
    const uint32 stubBitSize, const FSE_CTable* devCTable, uint32* devParkOverrunCount, cudaStream_t stream );

__global__ void CudaCompressToPark( const uint32 parkCount, const size_t parkSize, 
    uint64* linePoints, byte* parkBuffer, size_t parkBufferSize, 
    const uint32 stubBitSize, const FSE_CTable* cTable, uint32* gParkOverrunCount );
