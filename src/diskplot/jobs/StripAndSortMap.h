#pragma once

class ThreadPool;
#include "plotshared/MTJob.h"

struct StripMapJob : MTJob<StripMapJob>
{
    static void RunJob( ThreadPool& pool, 
        uint32 threadCount, uint32 entryCount,
        const uint64* inMap,
        uint32*       outKey,
        uint32*       outMap );

    uint32        entryCount;
    const uint64* inMap ;
    uint32*       outKey;
    uint32*       outMap;

    void Run() override;
};
