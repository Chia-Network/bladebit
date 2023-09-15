#pragma once
#include "Config.h"
#include "ChiaConsts.h"
#include "threading/ThreadPool.h"
#include "plotting/PlotTypes.h"
#include "plotting/PlotWriter.h"
#include "plotting/GlobalPlotConfig.h"

struct PlotRequest
{
    const byte* plotId;         // Id of the plot we want to create       
    const char* outDir;         // Output plot directory
    const char* plotFileName;   // .plot.tmp file name
    const char* plotOutPath;    // Full output path for the final .plot.tmp file
    const byte* memo;           // Plot memo
    uint16      memoSize;
    bool        isFirstPlot;
    bool        IsFinalPlot;
};

struct MemPlotConfig
{
    const struct GlobalPlotConfig* gCfg;
};

///
/// Context for a in-memory plotting
///
struct MemPlotContext
{
    MemPlotConfig cfg;

    // They id of the plot being plotted
    const byte* plotId;

    const byte* plotMemo;
    uint16      plotMemoSize;

    // How many threads to use for the thread pool?
    // #TODO: Remove this, just use the thread pool's count.
    uint32      threadCount;
    
    // Thread pool to use when running jobs
    ThreadPool* threadPool;

    ///
    /// Buffers
    ///
    // Permanent table data buffers
    uint32* t1XBuffer ;       // 16 GiB
    Pair*   t2LRBuffer;       // 32 GiB
    Pair*   t3LRBuffer;       // 32 GiB
    Pair*   t4LRBuffer;       // 32 GiB
    Pair*   t5LRBuffer;       // 32 GiB
    Pair*   t6LRBuffer;       // 32 GiB
    Pair*   t7LRBuffer;       // 32 GiB
    uint32* t7YBuffer ;       // 16 GiB

    // Temporary read/write y buffers
    uint64* yBuffer0;         // 32GiB each
    uint64* yBuffer1;

    // Temporary read/write metadata buffers
    uint64* metaBuffer0;      // 64GiB each
    uint64* metaBuffer1;

    uint64  maxKBCGroups;
    uint64  maxPairs;         // Max total pairs our buffer can hold
    
    // Number of entries per-table
    uint64 entryCount[7];

    // Added by Phase 2:
    byte* usedEntries[6];   // Used entries per each table.
                            // These are only used for tables 2-6 (inclusive).
                            // These buffers map to regions in yBuffer0.

    PlotWriter* plotWriter;

    // The buffer used to write to disk the Phase 4 data.
    // This may be metaBuffer0 or a an L/R buffer from
    // table 3 and up, the highest one available to use.
    // If null, then no buffer is in use.
    byte* p4WriteBuffer;
    byte* p4WriteBufferWriter;

    // How many plots we've made so far
    uint64 plotCount;
};

