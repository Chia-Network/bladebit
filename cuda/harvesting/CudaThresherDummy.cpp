#include "harvesting/Thresher.h"

/// Dummy function for when CUDA is not available
IThresher *CudaThresherFactory::Create( GreenReaperConfig& config )
{
    return nullptr;
}
