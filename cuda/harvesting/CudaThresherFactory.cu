#include "pch.h"
#include "harvesting/Thresher.h"

/// Defined in CudaThresher.cu
IThresher* CudaThresherFactory_Private( const struct GreenReaperConfig& config );

/// Declared in Thresher.h
IThresher* CudaThresherFactory::Create( const struct GreenReaperConfig& config )
{
    return CudaThresherFactory_Private( config );
}
