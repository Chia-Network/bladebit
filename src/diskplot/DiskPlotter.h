#pragma once

#include "DiskPlotContext.h"

class DiskPlotter
{
public:
    struct Config
    {
        const char*       tmpPath;
        size_t            workHeapSize;
        size_t            expectedTmpDirBlockSize;
        uint              workThreadCount;
        uint              ioThreadCount;
        size_t            ioBufferSize;
        DiskWriteInterval writeIntervals[(uint)TableId::_Count];
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

    static size_t GetHeapRequiredSize( const size_t fileBlockSize, const uint threadCount );
    

private:
    DiskPlotContext _cx;
};

