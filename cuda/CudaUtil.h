#pragma once

#include "pch.h"
#include "util/Util.h"
#include "util/Log.h"
#include <stdarg.h>

// CUDA windows fix
#ifdef _WIN32

    #ifdef __LITTLE_ENDIAN__
        #undef __LITTLE_ENDIAN__
        #define __LITTLE_ENDIAN__ 1
    #endif
#endif

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_profiler_api.h>

#define CuBSwap16( x ) ( ((x) >> 8) | ((x) << 8) )

#define CuBSwap32( x ) (\
    ((x) << 24) |\
    ((x) >> 24) |\
    (((x)>>8)&0xFF00) |\
    (((x)<<8)&0xFF0000) \
)

#define CuBSwap64( x ) (\
    ((x) << 56) |\
    ((x) >> 56) |\
    (((x)<<40) & 0x00FF000000000000ull) |\
    (((x)>>40) & 0x000000000000FF00ull) |\
    (((x)<<24) & 0x0000FF0000000000ull) |\
    (((x)>>24) & 0x0000000000FF0000ull) |\
    (((x)<< 8) & 0x000000FF00000000ull) |\
    (((x)>> 8) & 0x00000000FF000000ull) \
)

#if _DEBUG
    #define CUDA_ASSERT( expr ) assert( (expr) )
#else
    #define CUDA_ASSERT( expr ) 
#endif

struct CudaPlotInfo
{
    uint16 k;
    uint16 bucketCount;
    uint16 yBits;
    uint16 bucketBits;
    uint32 bucketCapacity;
    uint32 sliceCapacity;
    uint32 metaMaxSizeBytes;
};


inline void CudaErrCheck( cudaError_t err )
{
    if( err != cudaSuccess )
    {
        const char* cudaErrName = cudaGetErrorName( err );
        const char* cudaErrDesc = cudaGetErrorString( err );
        Log::Error( "CUDA error: %d (0x%-02x) %s : %s", err, err, cudaErrName, cudaErrDesc );
        Log::FlushError();
        Panic( "CUDA error %s : %s.", cudaErrName, cudaErrDesc );
    } 
    // FatalIf( err != cudaSuccess, "CUDA error: %d (0x%-02x) %s : %s",
    //          err, err, cudaGetErrorName( err ), cudaGetErrorString( err ) );
}

inline void CudaFatalCheckMsgV( cudaError_t err, const char* msg, va_list args )
{
    if( err == cudaSuccess )
        return;

    Log::Error( msg, args );
    Log::Error( "" );

    const char* cudaErrName = cudaGetErrorName( err );
    const char* cudaErrDesc = cudaGetErrorString( err );
    Log::Error( " CUDA error: %d (0x%-02x) %s : %s", err, err, cudaErrName, cudaErrDesc );
    Log::FlushError();
    Panic( " CUDA error %s : %s.", cudaErrName, cudaErrDesc );
}

inline void CudaFatalCheckMsg( cudaError_t err, const char* msg, ... )
{
    if( err == cudaSuccess )
        return;
    va_list args;
    va_start( args, msg );
    CudaFatalCheckMsgV( err, msg, args );
    va_end( args );
}

template<typename T>
inline cudaError_t CudaCallocT( T*& ptr, const size_t count )
{
    return cudaMalloc( &ptr, count * sizeof( T ) );
}

template<typename T>
inline cudaError_t CudaSafeFree( T*& ptr )
{
    if( ptr ) 
    {
        cudaError_t r = cudaFree( (void*)ptr );
        ptr = nullptr;
        return r;
    }

    return cudaSuccess;
}

template<typename T>
inline cudaError_t CudaSafeFreeHost( T*& ptr )
{
    if( ptr ) 
    {
        cudaError_t r = cudaFreeHost( (void*)ptr );
        ptr = nullptr;
        return r;
    }

    return cudaSuccess;
}