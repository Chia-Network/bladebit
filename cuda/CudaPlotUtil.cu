#include "CudaPlotContext.h"

//-----------------------------------------------------------
__global__ void GenSortKey( const uint32 entryCount, uint32* key )
{
    const uint32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if( gid >= entryCount )
        return;

    key[gid] = gid;
}

//-----------------------------------------------------------
void CudaK32PlotGenSortKey( const uint32 entryCount, uint32* devKey, cudaStream_t stream, bool synchronize )
{
    const uint32 threadsPerBlock = 128;
    const uint32 blockCount      = CDiv( entryCount, threadsPerBlock );
    
    if( stream == nullptr )
        stream = CU_STREAM_LEGACY;

    GenSortKey<<<blockCount, threadsPerBlock, 0, stream>>>( entryCount, devKey );
    if( synchronize )
        CudaErrCheck( cudaStreamSynchronize( stream ) );
    
}

//-----------------------------------------------------------
template<typename T>
__global__ void SortByKey( const uint32 entryCount, const uint32* key, const T* input, T* output )
{
    const uint32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if( gid >= entryCount )
        return;

    output[gid] = input[key[gid]];
}


//-----------------------------------------------------------
template<typename T>
void CudaK32PlotSortByKey( const uint32 entryCount, const uint32* devKey, const T* devInput, T* devOutput, cudaStream_t stream, bool synchronize )
{
    const uint32 threadsPerBlock = 128;
    const uint32 blockCount      = CDiv( entryCount, threadsPerBlock );
    
    if( stream == nullptr )
        stream = CU_STREAM_LEGACY;

    SortByKey<T><<<blockCount, threadsPerBlock, 0, stream>>>( entryCount, devKey, devInput, devOutput );
    if( synchronize )
        CudaErrCheck( cudaStreamSynchronize( stream ) );
}

//-----------------------------------------------------------
void CudaK32PlotSortMeta( const uint32 entryCount, const uint32* devKey, const uint32* devMetaIn, uint32* devMetaOutput, cudaStream_t stream )
{

}


template void CudaK32PlotSortByKey<uint16>( const uint32 entryCount, const uint32* devKey, const uint16* devInput, uint16* devOutput, cudaStream_t stream, bool synchronize );
template void CudaK32PlotSortByKey<uint32>( const uint32 entryCount, const uint32* devKey, const uint32* devInput, uint32* devOutput, cudaStream_t stream, bool synchronize );
template void CudaK32PlotSortByKey<uint64>( const uint32 entryCount, const uint32* devKey, const uint64* devInput, uint64* devOutput, cudaStream_t stream, bool synchronize );
template void CudaK32PlotSortByKey<K32Meta3>( const uint32 entryCount, const uint32* devKey, const K32Meta3* devInput, K32Meta3* devOutput, cudaStream_t stream, bool synchronize );
template void CudaK32PlotSortByKey<K32Meta4>( const uint32 entryCount, const uint32* devKey, const K32Meta4* devInput, K32Meta4* devOutput, cudaStream_t stream, bool synchronize );
template void CudaK32PlotSortByKey<Pair>( const uint32 entryCount, const uint32* devKey, const Pair* devInput, Pair* devOutput, cudaStream_t stream, bool synchronize );

