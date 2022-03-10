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
        const char*       tmpPath2                 = nullptr;
        size_t            workHeapSize             = 0;
        size_t            expectedTmpDirBlockSize  = 0;
        uint32            numBuckets               = 256;
        uint32            ioThreadCount            = 0;
        size_t            ioBufferSize             = 0;
        uint32            ioBufferCount            = 0;
        size_t            cacheSize                = 0;

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

    static bool   GetTmpPathsBlockSizes(  const char* tmpPath1, const char* tmpPath2, size_t& tmpPath1Size, size_t& tmpPath2Size );
    static size_t GetRequiredSizeForBuckets( const uint32 numBuckets, const char* tmpPath1, const char* tmpPath2 );
    static size_t GetRequiredSizeForBuckets( const uint32 numBuckets, const size_t fxBlockSize, const size_t pairsBlockSize );
    
    static void ParseCommandLine( CliParser& cli, Config& cfg );

    static void PrintUsage();


private:
    DiskPlotContext   _cx;
};

