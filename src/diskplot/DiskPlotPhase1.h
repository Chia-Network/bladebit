#pragma once
#include "DiskPlotContext.h"

struct GenF1Job;

class DiskPlotPhase1
{

public:
    DiskPlotPhase1( DiskPlotContext& cx );

    void Run();

private:

    void GenF1();
    static void GenF1Thread( GenF1Job* job );

private:
    DiskPlotContext& _cx;
};