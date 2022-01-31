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
        uint              ioBufferCount;
        DiskWriteInterval writeIntervals[(uint)TableId::_Count];
        bool              enableDirectIO;
    };

    struct PlotRequest
    {
        const byte*  plotId;
        const byte*  plotMemo;
        uint16       plotMemoSize;
        const char*  plotFileName;
    };

public:
    // DiskPlotter();
    DiskPlotter( const Config cfg );

    void Plot( const PlotRequest& req );

    static void GetHeapRequiredSize( DiskFPBufferSizes& sizes, const size_t fileBlockSize, const uint threadCount );
    

private:
    DiskPlotContext   _cx;
    DiskFPBufferSizes _fpBufferSizes;
};

