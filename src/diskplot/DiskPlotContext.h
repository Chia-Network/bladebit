#pragma once
#include "DiskPlotConfig.h"
#include "DiskBufferQueue.h"
#include "threading/ThreadPool.h"
#include "plotshared/MTJob.h"
#include "plotshared/PlotTypes.h"
#include "ChiaConsts.h"

// Write intervals are expressed in bytes
struct DiskWriteInterval
{
    size_t matching;    // Write interval during matching
    size_t fxGen;       // Write interval during fx generation
};

struct DiskFPBufferSizes
{
    size_t fileBlockSize;

    size_t yIO;
    size_t sortKeyIO;
    size_t mapIO;
    size_t metaAIO;
    size_t metaBIO;
    size_t pairsLeftIO;
    size_t pairsRightIO;

    size_t groupsSize;
    size_t yTemp;
    size_t metaATmp;
    size_t metaBTmp;

    size_t yOverflow;
    size_t mapOverflow;
    size_t metaAOverflow;
    size_t metaBOverflow;
    size_t pairOverflow;

    size_t crossBucketY;
    size_t crossBucketMetaA;
    size_t crossBucketMetaB;
    size_t crossBucketPairsLeft;
    size_t crossBucketPairsRight;
    size_t crossBucketTotal;

    size_t totalSize;
};

struct DiskPlotContext
{
    const char*  tmpPath;           // Path in which to allocate temporary buffers
    ThreadPool*  threadPool;

    DiskBufferQueue* ioQueue;

    size_t       heapSize;          // Size in bytes of our working heap. Some parts are preallocated.
    byte*        heapBuffer;        // Buffer allocated for in-memory work

    // Write intervals are expressed in bytes
    size_t       f1WriteInterval;
    size_t       matchWriteInterval;
    DiskWriteInterval writeIntervals[(uint)TableId::_Count];

    size_t       ioBufferSize;      // Largest write interval out of all the specified. 
    size_t       ioHeapSize;        // Allocation size for the IO heap
    byte*        ioHeap;            // Full allocation of the IO heap
    uint         ioBufferCount;     // How many single IO buffers can we allocate

    uint         threadCount;       // How many threads to use for in-memory plot work
    uint         ioThreadCount;     // How many threads to use for the disk buffer writer/reader
    bool         useDirectIO;       // Use unbuffered (direct-IO) when performing temp read/writes?

    const byte*  plotId;
    const byte*  plotMemo;
    uint32       plotMemoSize;

    uint32       bucketCounts[(uint)TableId::_Count][BB_DP_BUCKET_COUNT];
    uint64       entryCounts [(uint)TableId::_Count];

    DiskFPBufferSizes* bufferSizes;
};

