#pragma once

#include "threading/ThreadPool.h"

// Returns: Group count found, minus the last 2 ghost groups.
uint64 ScanBCGroupThread32(
    const uint64* yBuffer,
    uint64        scanStart,
    uint64        scanEnd,
    uint32*       groupIndices,
    uint32        maxGroups,
    uint32        jobId = 0 );

// Returns: Group count found, minus the last 2 ghost groups.
uint64 ScanBCGroupMT32( 
    ThreadPool&   pool, 
          uint32  threadCount,
    const uint64* yBuffer,
    const uint32  entryCount,
          uint32* tmpGroupIndices,
          uint32* outGroupIndices,
    const uint32  maxGroups
);