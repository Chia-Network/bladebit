#pragma once

#include "plotting/Tables.h"

struct GreenReaperContext;
struct Pair;

/// Plot decompression interface
class IThresher
{
public:
    virtual bool AllocateBuffers( uint k, uint maxCompressionLevel ) = 0;

    virtual void ReleaseBuffers() = 0;

    virtual bool DecompressInitialTable(
        GreenReaperContext& cx,
        const byte plotId[32],
        uint32     entryCountPerX,
        Pair*      outPairs,
        uint64*    outY,
        void*      outMeta,
        uint32&    outMatchCount,
        uint64 x0, uint64 x1,
        uint32*    outErrorCode ) = 0;

    virtual bool DecompressTableGroup(
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
        const void*     inMeta,
        uint32*         outErrorCode ) = 0;

    // For testing
    virtual void DumpTimings() {}

    virtual void ClearTimings() {}
};


class CudaThresherFactory
{
public:
    static IThresher* Create( const struct GreenReaperConfig& config );
};

