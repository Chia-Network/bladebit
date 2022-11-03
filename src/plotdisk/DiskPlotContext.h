#pragma once
#include "DiskPlotConfig.h"
#include "DiskBufferQueue.h"
#include "threading/ThreadPool.h"
#include "threading/MTJob.h"
#include "plotting/PlotTypes.h"
#include "plotting/Tables.h"
#include "ChiaConsts.h"

struct GlobalPlotConfig;

struct DiskPlotConfig
{
    GlobalPlotConfig* globalCfg                = nullptr;
    const char*       tmpPath                  = nullptr;
    const char*       tmpPath2                 = nullptr;
    size_t            expectedTmpDirBlockSize  = 0;
    uint32            numBuckets               = 256;
    uint32            ioThreadCount            = 0;
    uint32            ioBufferCount            = 0;
    size_t            cacheSize                = 0;

    bool              bounded                  = true;  // Do not overflow entries
    bool              alternateBuckets         = false; // Alternate bucket writing method between interleaved and not
    bool              noTmp1DirectIO           = false; // Disable direct I/O on tmp 1
    bool              noTmp2DirectIO           = false; // Disable direct I/O on tmp 1

    uint32            f1ThreadCount            = 0;
    uint32            fpThreadCount            = 0;
    uint32            cThreadCount             = 0;
    uint32            p2ThreadCount            = 0;
    uint32            p3ThreadCount            = 0;
};

struct DiskPlotContext
{
    const DiskPlotConfig* cfg;

    const char*  tmpPath;           // Path in which to allocate temporary buffers
    const char*  tmpPath2;          // Path to store fx files and other high R/W files.
    
    size_t       tmp1BlockSize;
    size_t       tmp2BlockSize;

    ThreadPool*      threadPool;
    DiskBufferQueue* ioQueue;
    FencePool*       fencePool;

    size_t       heapSize;          // Size in bytes of our working heap.
    byte*        heapBuffer;        // Buffer allocated for in-memory work
    
    size_t       cacheSize;         // Size of memory cache to reserve for IO (region in file that never gets written to disk).
    byte*        cache;

    uint32       numBuckets;        // Divide entries into this many buckets

    uint32       ioThreadCount;     // How many threads to use for the disk buffer writer/reader
    uint32       f1ThreadCount;     // How many threads to use for f1 generation
    uint32       fpThreadCount;     // How many threads to use for forward propagation
    uint32       cThreadCount;      // How many threads to use for C3 park writing and compression
    uint32       p2ThreadCount;     // How many threads to use for Phase 2
    uint32       p3ThreadCount;     // How many threads to use for Phase 3

    const byte*  plotId;
    const byte*  plotMemo;
    uint16       plotMemoSize;

    uint32       bucketCounts[(uint)TableId::_Count][BB_DP_MAX_BUCKET_COUNT+1];
    uint64       entryCounts [(uint)TableId::_Count];

    uint32       bucketSlices[2][BB_DP_MAX_BUCKET_COUNT];


    // Since back pointer table entries are not sorted along with y,
    // (instead we use a mapping table), and since their values are stored
    // in local-to-bucket coordinates, we need to know how many entries
    // were generated given an L table bucket.
    // This stores how many R entries were generated given an L table bucket,
    // including the cross-bucket entries.
    uint32       ptrTableBucketCounts[(uint)TableId::_Count][BB_DP_MAX_BUCKET_COUNT];

    // Pointers to tables in the plot file (byte offset to where it starts in the plot file)
    // Where:
    //  0-6 = Parked tables 1-7
    //  7-9 = C1-C3 tables
    uint64       plotTablePointers[10];
    uint64       plotTableSizes   [10];

    Duration ioWaitTime;                                // Total Plot I/O wait time
    Duration p1TableWaitTime[(uint)TableId::_Count];    // Phase 1 per-table I/O wait time
    Duration p2TableWaitTime[(uint)TableId::_Count];    // Phase 2 per-table I/O wait time
    Duration p3TableWaitTime[(uint)TableId::_Count];    // Phase 3 per-table I/O wait time
    Duration cTableWaitTime;
    Duration p7WaitTime;
};
