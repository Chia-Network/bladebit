#pragma once

#include "plotting/Tables.h"

struct GreenReaperContext;
struct Pair;

enum class ThresherResultKind
{
    Success = 0,    // Successfully processed a table with matches.
    NoMatches,      // No error occurred, but no matches we obtained.
    Error,          // An error has occurred when decompressing and an error code will be set.
};

enum class ThresherError
{
    None = 0,
    UnexpectedError,
    CudaError,
};

struct ThresherResult
{
    ThresherResultKind kind;
    ThresherError      error;
    i32                internalError;
};


/// Plot decompression interface
class IThresher
{
public:
    inline virtual ~IThresher() {}

    virtual bool AllocateBuffers( uint k, uint maxCompressionLevel ) = 0;

    virtual void ReleaseBuffers() = 0;

    virtual ThresherResult DecompressInitialTable(
        GreenReaperContext& cx,
        const byte plotId[32],
        uint32     entryCountPerX,
        Pair*      outPairs,
        uint64*    outY,
        void*      outMeta,
        uint32&    outMatchCount,
        uint64 x0, uint64 x1) = 0;

    virtual ThresherResult DecompressTableGroup(
        GreenReaperContext& cx,
        const TableId   table,
        uint32          entryCount,
        uint32          matchOffset,
        uint32          maxPairs,
        uint32&         outMatchCount,
        Pair*           outPairs,
        uint64*         outY,
        void*           outMeta,
        Pair*           outLPairs,  // Where to store sorted input pairs from previous table
        const Pair*     inLPairs,
        const uint64*   inY,
        const void*     inMeta ) = 0;

    // For testing
    virtual void DumpTimings() {}

    virtual void ClearTimings() {}
};


class CudaThresherFactory
{
public:
    static IThresher* Create( const struct GreenReaperConfig& config );
};

