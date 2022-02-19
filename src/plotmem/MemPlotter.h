#pragma once
#include "PlotContext.h"

struct NumaInfo;

struct MemPlotConfig
{
    uint threadCount;
    bool warmStart;
    bool noNUMA;
    bool noCPUAffinity;
};

// This plotter performs the whole plotting process in-memory.
class MemPlotter
{
public:

    MemPlotter( const MemPlotConfig& cfg );
    ~MemPlotter();

    bool Run( const PlotRequest& request );

private:

    template<typename T>
    T* SafeAlloc( size_t size, bool warmStart, const NumaInfo* numa );

    // Check if the background plot writer finished
    void WaitPlotWriter();

private:

    MemPlotContext _context;
};