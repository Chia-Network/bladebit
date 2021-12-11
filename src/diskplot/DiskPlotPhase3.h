#pragma once

#include "DiskPlotContext.h"

class DiskPlotPhase3
{
public:
    DiskPlotPhase3( DiskPlotContext& context );
    ~DiskPlotPhase3();

    void Run();

private:
    DiskPlotContext& _context;
};
