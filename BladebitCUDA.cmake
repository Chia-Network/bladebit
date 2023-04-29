add_executable(bladebit_cuda
    src/main.cpp

    cuda/CudaPlotter.cu
    cuda/CudaPlotter.h
    cuda/CudaPlotContext.h
    cuda/CudaPlotPhase2.cu
    cuda/CudaPlotPhase3.cu
    cuda/CudaPlotPhase3Step2.cu
    cuda/CudaPlotPhase3Step3.cu
    cuda/CudaPlotPhase3Internal.h
    cuda/CudaParkSerializer.h
    cuda/CudaParkSerializer.cu
    cuda/chacha8.cu
    cuda/CudaF1.h
    cuda/CudaF1.cu
    cuda/CudaMatch.h
    cuda/CudaMatch.cu
    cuda/CudaFx.h
    cuda/FxCuda.cu
    cuda/CudaUtil.h
    cuda/CudaPlotUtil.cu
    cuda/GpuStreams.h
    cuda/GpuStreams.cu

    # Harvester
    cuda/harvesting/CudaThresher.cu
    cuda/harvesting/CudaThresherFactory.cu
)

target_include_directories(bladebit_cuda PRIVATE src cuda SYSTEM cuda)

target_compile_definitions(bladebit_cuda PUBLIC
    BB_CUDA_ENABLED=1
    THRUST_IGNORE_CUB_VERSION_CHECK=1)

target_compile_options(bladebit_cuda PRIVATE 
    $<${is_cuda_debug}:
        # -arch=native
        -G
    >
 )

target_link_libraries(bladebit_cuda PRIVATE bladebit_core CUDA::cudart_static CUDA::cuda_driver)

set_target_properties(bladebit_cuda PROPERTIES 
    MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>
    CUDA_RUNTIME_LIBRARY Static
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    # CUDA_ARCHITECTURES OFF
)

# target_compile_options(lib_bladebit_cuda PUBLIC 
#     $<${is_cuda_release}:
#         -gencode=arch=compute_52,code=sm_52 # Maxwell
#         -gencode=arch=compute_61,code=sm_61 # Pascal
#         -gencode=arch=compute_70,code=sm_70 # Volta 
#         -gencode=arch=compute_86,code=sm_86 # Ampere
#         -gencode=arch=compute_89,code=sm_89 # Ada
#     >)