#pragma once

#include "plotting/GlobalPlotConfig.h"
#include "util/CliParser.h"
#include "PlotContext.h"
#include "plotting/IPlotter.h"

struct CudaK32PlotConfig
{
    const GlobalPlotConfig* gCfg        = nullptr;

    uint32 deviceIndex            = 0;      // Which CUDA device to use when plotting/
    bool   disableDirectDownloads = false;  // Don't allocate host tables using pinned buffers, instead
                                            // download to intermediate pinned buffers then copy to the final host buffer.
                                            // May be necessarry on Windows because of shared memory limitations (usual 50% of system memory)

    bool hybrid128Mode            = false;  // Enable hybrid disk-offload w/ 128G of RAM.
    bool hybrid64Mode             = false;  // Enable hybrid disk-offload w/ 64G of RAM.

    const char* temp1Path         = nullptr; // For 128G RAM mode
    const char* temp2Path         = nullptr; // For 64G RAM mode

    bool temp1DirectIO            = true;   // Use direct I/O for temp1 files
    bool temp2DirectIO            = true;   // Use direct I/O for temp2 files
};

class CudaK32Plotter : public IPlotter
{
public:
    inline CudaK32Plotter() {}
    inline virtual ~CudaK32Plotter() {}

    virtual void ParseCLI( const GlobalPlotConfig& gCfg, CliParser& cli ) override;
    virtual void Init() override;
    virtual void Run( const PlotRequest& req ) override;

private:
    CudaK32PlotConfig          _cfg = {};
    struct CudaK32PlotContext* _cx  = nullptr;;
};