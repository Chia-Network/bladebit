#pragma once
#include "ChiaConsts.h"
#include <string>

struct PlotCheckerConfig
{
    uint64      proofCount         = 100;
    bool        noGpu              = false;
    int32       gpuIndex           = -1;
    uint32      threadCount        = 0;
    bool        disableCpuAffinity = false;
    bool        silent             = false;
    bool        hasSeed            = false;
    byte        seed[BB_PLOT_ID_LEN]{};

    bool        deletePlots     = false;    // If true, plots that fail to fetch proofs, or are below a threshold, will be deleted
    double      deleteThreshold = 0.0;      // If proofs received to proof request ratio is below this, the plot will be deleted

    struct GreenReaperContext* grContext = nullptr;
};

struct PlotCheckResult
{
    uint64      checkCount;
    uint64      proofCount;
    uint64      proofFetchFailCount;
    uint64      proofValidationFailCount;
    byte        seedUsed[BB_PLOT_ID_LEN];
    std::string error;
    bool        deleted;
};

class PlotChecker
{
public:

protected:
    PlotChecker() = default;

public:
    static PlotChecker* Create( PlotCheckerConfig& cfg );
    virtual ~PlotChecker() = default;

    // Add a plot ot the queue to be checked
    /// Returns true if the plot passed the threshold check
    virtual void CheckPlot( const char* plotPath, PlotCheckResult* outResult ) = 0;

    // Returns true if the last plot checked was deleted
    virtual bool LastPlotDeleted() = 0;
};
