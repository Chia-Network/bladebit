#include "Commands.h"
#include "plotting/GlobalPlotConfig.h"
#include "ChiaConsts.h"
#include <vector>
#include <string>

struct PlotCheckConfig
{
    GlobalPlotConfig* gCfg    = nullptr;

    uint64                   proofCount = 100;
    const char*              plotPath   = "";
    // std::vector<const char*> plotPaths{};
    byte                     seed[BB_PLOT_ID_LEN]{};
    bool                     hasSeed    = false;
    bool                     noGpu      = false;
    int32                    gpuIndex   = -1;
    bool                     silent     = false;

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
};

bool RunPlotsCheck( PlotCheckConfig& cfg, uint32 threadCount, bool disableCpuAffinity, PlotCheckResult* outResult );
