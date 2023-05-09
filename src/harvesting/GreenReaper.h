#pragma once 
#include <stdint.h>

#ifdef GR_EXPORT
    #ifdef _WIN32
        #define GR_API __declspec(dllexport)
    #else
        #define GR_API __attribute__ ((visibility("default")))
    #endif
#elif !defined( GR_NO_IMPORT )
    #ifdef _WIN32
        #define GR_API __declspec(dllimport)
    #else
        #define GR_API
    #endif
#else
    #define GR_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GR_API_VERSION 1

#define GR_POST_PROOF_X_COUNT 64
#define GR_POST_PROOF_CMP_X_COUNT (GR_POST_PROOF_X_COUNT/2)

typedef int32_t GRBool;
#define GR_FALSE 0
#define GR_TRUE  1

typedef struct GreenReaperContext GreenReaperContext;

/// How to select GPU for harvesting.
typedef enum GRGpuRequestKind
{
    GRGpuRequestKind_None = 0,        // Disable GPU harvesting.
    GRGpuRequestKind_FirstAvailable,  // Select the device specified, or the first available, if any. If no device is available it defaults to CPU harvesting.
    GRGpuRequestKind_ExactDevice,     // Select the specified device only. If none is available, it is an error.
} GRGpuRequestKind;

typedef uint32_t GRGpuRequestKind_t;

typedef struct GreenReaperConfig
{
    uint32_t           apiVersion;
    uint32_t           threadCount;
    uint32_t           cpuOffset;
    GRBool             disableCpuAffinity;
    GRGpuRequestKind_t gpuRequest;         // What kind of GPU to select for harvesting.
    uint32_t           gpuDeviceIndex;     // Which device index to use (0 by default)

    uint32_t           _reserved[16];      // Reserved for future use
} GreenReaperConfig;

typedef enum GRResult
{
    GRResult_Failed        = 0,
    GRResult_OK            = 1,
    GRResult_OutOfMemory   = 2,
    GRResult_NoProof       = 3,  // A dropped proof due to line point compression
    GRResult_WrongVersion  = 4,
    GRResult_InvalidGPU    = 5,  // Invalid or missing GPU selection. (When GRGpuRequestKind_ExactDevice is used.)
    GRResult_InvalidArg    = 6,  // An invalid argument was passed.

} GRResult;

typedef struct GRCompressionInfo
{
    uint32_t entrySizeBits;
    uint32_t subtSizeBits;
    size_t   tableParkSize;
    double   ansRValue;
} GRCompressionInfo;

// Timings expressed in nanoseconds
// typedef struct GRProofTimings
// {
//     uint64_t totalElapsedNS;
//     uint64_t f1ElapsedNS;
//     uint64_t sortElapsedNS;
//     uint64_t matchElapsedNS;
//     uint64_t fxElapsedNS;
// };

typedef struct GRCompressedProofRequest
{
    union {
        uint64_t compressedProof[GR_POST_PROOF_CMP_X_COUNT];   // Corresponds to the x buckets in line points form
        uint64_t fullProof      [GR_POST_PROOF_X_COUNT];
    };

          uint32_t  compressionLevel;
    const uint8_t*  plotId;

    // Pass a pointer to a timings struct if 
    // you'd like detailed timings output
    // GRProofTimings* outTimings;
          
} GRCompressedProofRequest;

typedef struct GRLinePoint
{
    uint64_t hi;  // High-order bytes
    uint64_t lo;  // Low-order bytes
} GRLinePoint;

typedef struct GRCompressedQualitiesRequest
{
    // Input
    const uint8_t*  plotId;
    const uint8_t*  challenge;
    uint32_t        compressionLevel;
    GRLinePoint     xLinePoints[2];     // Line points with compressed x's

    // Output
    uint64_t        x1, x2;             // Output x qualities

} GRCompressedQualitiesRequest;

typedef struct GRApiV1
{
    GRResult (*CreateContext)( GreenReaperContext** outContext, GreenReaperConfig* config, size_t configStructSize );
    void     (*DestroyContext)( GreenReaperContext* context );
    GRResult (*PreallocateForCompressionLevel)( GreenReaperContext* context, uint32_t k, uint32_t maxCompressionLevel );
    GRResult (*FetchProofForChallenge)( GreenReaperContext* context, GRCompressedProofRequest* req );
    GRResult (*GetFetchQualitiesXPair)( GreenReaperContext* context, GRCompressedQualitiesRequest* req );
    size_t   (*GetMemoryUsage)( GreenReaperContext* context );
    GRBool   (*HasGpuDecompressor)( GreenReaperContext* context );
    GRResult (*GetCompressionInfo)( GRCompressionInfo* outInfo, size_t infoStructSize, uint32_t k, uint32_t compressionLevel );

} GRApiV1;

typedef GRApiV1 GRApi;


///
/// API
///

/// Populate an API object with all the current API's functions.
/// A single API version is ever supported per binary.
GR_API GRResult grPopulateApi( GRApi* api, size_t apiStructSize, int apiVersion );

/// Create a decompression context
GR_API GRResult grCreateContext( GreenReaperContext** outContext, GreenReaperConfig* config, size_t configStructSize );

/// Destroy decompression context
GR_API void grDestroyContext( GreenReaperContext* context );

/// Preallocate context's in-memory buffers to support a maximum compression level
GR_API GRResult grPreallocateForCompressionLevel( GreenReaperContext* context, uint32_t k, uint32_t maxCompressionLevel );

/// Full proof of space request given a challenge
GR_API GRResult grFetchProofForChallenge( GreenReaperContext* context, GRCompressedProofRequest* req );

/// Request plot qualities for a challenge
GR_API GRResult grGetFetchQualitiesXPair( GreenReaperContext* context, GRCompressedQualitiesRequest* req );

GR_API size_t grGetMemoryUsage( GreenReaperContext* context );

/// Returns true if the context has a Gpu-based decompressor created.
GR_API GRBool grHasGpuDecompressor( GreenReaperContext* context );

GR_API GRResult grGetCompressionInfo( GRCompressionInfo* outInfo, size_t infoStructSize, uint32_t k, uint32_t compressionLevel );

#ifdef __cplusplus
}
#endif
