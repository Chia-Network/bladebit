#pragma once

#include "DiskPlotContext.h"

class DiskPlotter
{
public:
    struct Config
    {
        size_t ramSizeBytes;
        uint   threadCount;
        uint   diskQueueThreadCount;
    };

    struct PlotRequest
    {
        byte*  plotId;
        byte*  plotMemo;
        uint64 plotMemoSize;
    };

public:
    DiskPlotter();
    DiskPlotter( const Config cfg );

    void Plot( const PlotRequest& req );

private:
    DiskPlotContext _cx;
};

