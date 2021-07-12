#pragma once
#include "PlotContext.h"


// This plotter performs the whole plot process in-memory.
// At least Phases 1 and 2.
// We have not yet decided what to do with Phase 3.
class MemPlotter
{
public:

    MemPlotter( uint threadCount, bool warmStart );
    ~MemPlotter();

    bool Run( const PlotRequest& request );

private:

    template<typename T>
    T* SafeAlloc( size_t size, bool warmStart );

    // Check if the background plot writer finished
    void WaitPlotWriter();


private:

    MemPlotContext _context;
};