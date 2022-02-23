#pragma once

#include "DiskPlotContext.h"
#include "plotting/GlobalPlotConfig.h"
class CliParser;

class DiskPlotter
{
public:
    struct Config
    {
        GlobalPlotConfig* globalCfg                = nullptr;
        const char*       tmpPath                  = nullptr;
        size_t            workHeapSize             = 0;
        size_t            expectedTmpDirBlockSize  = 0;
        uint32            ioThreadCount            = 0;
        size_t            ioBufferSize             = 0;
        uint32            ioBufferCount            = 0;
        DiskWriteInterval writeIntervals[(uint)TableId::_Count] = { 0 };
        bool              enableDirectIO           = false;

        uint32            f1ThreadCount            = 0;
        uint32            fpThreadCount            = 0;
        uint32            cThreadCount             = 0;
        uint32            p2ThreadCount            = 0;
        uint32            p3ThreadCount            = 0;
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
    
    static void ParseCommandLine( CliParser& cli, Config& cfg );

    static void PrintUsage();
private:
    DiskPlotContext   _cx;
    DiskFPBufferSizes _fpBufferSizes;
};

