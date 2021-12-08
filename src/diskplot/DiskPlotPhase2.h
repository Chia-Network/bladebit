#pragma once

#include "DiskPlotContext.h"

class DiskPlotPhase2
{
public:
    DiskPlotPhase2( DiskPlotContext& context );
    ~DiskPlotPhase2();

    void Run();

private:
    DiskPlotContext& _context;
};