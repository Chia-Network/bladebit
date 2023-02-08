#pragma once 
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


#define GR_POST_PROOF_X_COUNT 64
#define GR_POST_PROOF_CMP_X_COUNT (GR_POST_PROOF_X_COUNT/2)

typedef int32_t grBool;

typedef struct GreenReaperContext GreenReaperContext;

typedef struct GreenReaperConfig
{
    uint32_t threadCount;

} GreenReaperConfig;

typedef enum GRProofResult
{
    GRProofResult_Failed      = 0,
    GRProofResult_OK          = 1,
    GRProofResult_OutOfMemory = 2,

} GRProofResult;

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
        uint32_t compressedProof[GR_POST_PROOF_CMP_X_COUNT];   // Corresponds to the x buckets in line points form
        uint64_t fullProof      [GR_POST_PROOF_X_COUNT];
    };

    
          uint32_t  compressionLevel;
    const uint8_t*  plotId;

    // Pass a pointer to a timings struct if 
    // you'd like detailed timings output
    // GRProofTimings* outTimings;
          
} GRCompressedProofRequest;


GreenReaperContext* grCreateContext( GreenReaperConfig* config );
void                grDestroyContext( GreenReaperContext* context );


GRProofResult grFetchProofForChallenge( GreenReaperContext* context, GRCompressedProofRequest* req );


#ifdef __cplusplus
}
#endif