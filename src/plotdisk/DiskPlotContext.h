#pragma once
#include "DiskPlotConfig.h"
#include "DiskBufferQueue.h"
#include "threading/ThreadPool.h"
#include "threading/MTJob.h"
#include "plotting/PlotTypes.h"
#include "plotting/Tables.h"
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

    size_t       totalHeapSize;     // Complete size of our heap allocation (ioHeapSize + heapSize)
    size_t       heapSize;          // Size in bytes of our working heap. Some parts are preallocated.
    byte*        heapBuffer;        // Buffer allocated for in-memory work
    
    size_t       cacheSize;         // Size of memory cache to reserve for IO (region in file that never gets written to disk).
    byte*        cache;

    // Write intervals are expressed in bytes
    uint32       numBuckets;
    size_t       f1WriteInterval;
    size_t       matchWriteInterval;
    DiskWriteInterval writeIntervals[(uint)TableId::_Count];

    size_t       ioBufferSize;      // Largest write interval out of all the specified. 
    size_t       ioHeapSize;        // Allocation size for the IO heap
    byte*        ioHeap;            // Full allocation of the IO heap
    uint         ioBufferCount;     // How many single IO buffers can we allocate

    // uint32       threadCount;       // How many threads to use for in-memory plot work
    uint32       ioThreadCount;     // How many threads to use for the disk buffer writer/reader
    uint32       f1ThreadCount;     // How many threads to use for f1 generation
    uint32       fpThreadCount;     // How many threads to use for forward propagation
    uint32       cThreadCount;      // How many threads to use for C3 park writing and compression
    uint32       p2ThreadCount;     // How many threads to use for Phase 2
    uint32       p3ThreadCount;     // How many threads to use for Phase 3
    bool         useDirectIO;       // Use unbuffered (direct-IO) when performing temp read/writes?

    const byte*  plotId;
    const byte*  plotMemo;
    uint16       plotMemoSize;

    uint32       bucketCounts[(uint)TableId::_Count][BB_DP_BUCKET_COUNT];
    uint64       entryCounts [(uint)TableId::_Count];

    // Since back pointer table entries are not sorted along with y,
    // (instead we use a mapping table), and since their values are stored
    // in local-to-bucket coordinates, we need to know how many entries
    // were generated given an L table bucket.
    // This stores how many R entries were generated given an L table bucket,
    // including the cross-bucket entries.
    uint32       ptrTableBucketCounts[(uint)TableId::_Count][BB_DP_BUCKET_COUNT];;

    // Pointers to tables in the plot file (byte offset to where it starts in the plot file)
    // Where:
    //  0-6 = Parked tables 1-7
    //  7-9 = C1-C3 tables
    uint64       plotTablePointers[10];
    uint64       plotTableSizes   [10];

    DiskFPBufferSizes* bufferSizes;

    Duration readWaitTime;
    Duration writeWaitTime;
};

struct Phase3Data
{
    uint64 maxTableLength;
    size_t bitFieldSize;
    uint32 bucketMaxSize;
};

