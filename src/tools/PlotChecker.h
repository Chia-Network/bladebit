#pragma once

class PlotChecker
{
protected:
    PlotChecker() = default;
public:
    static PlotChecker* Create();
    virtual ~PlotChecker() = default;

    // Add a plot ot the queue to be checked
    virtual void AddPlot( const char* plotPath, uint32 checkCount, float deleteThreshHold ) = 0;

    // Output plot check status (passed or failed) to stdout from previous plot checks
    virtual void DumpCheckStats() = 0;
};
