#pragma once
#include "PlotContext.h"
#include "plotting/GlobalPlotConfig.h"
#include "plotting/IPlotter.h"

struct NumaInfo;

// This plotter performs the whole plotting process in-memory.
class MemPlotter : public IPlotter
{
public:

    inline MemPlotter() {}
    inline ~MemPlotter() {}

    void ParseCLI( const GlobalPlotConfig& gCfg, CliParser& cli ) override;
    void Init() override;
    void Run( const PlotRequest& req ) override;

private:

    template<typename T>
    T* SafeAlloc( size_t size, bool warmStart, const NumaInfo* numa );

    void BeginPlotFile( const PlotRequest& request );

    // Check if the background plot writer finished
    void WaitPlotWriter();

private:

    MemPlotContext _context = {};
};