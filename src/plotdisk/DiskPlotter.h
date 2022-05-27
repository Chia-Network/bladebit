#pragma once

#include "DiskPlotContext.h"
#include "plotting/GlobalPlotConfig.h"
class CliParser;

class DiskPlotter
{
public:
    using Config = DiskPlotConfig;

    struct PlotRequest
    {
        const byte*  plotId;
        const byte*  plotMemo;
        uint16       plotMemoSize;
        const char*  plotFileName;
    };

public:
    // DiskPlotter();
    DiskPlotter( const Config& cfg );

    void Plot( const PlotRequest& req );

    static bool   GetTmpPathsBlockSizes(  const char* tmpPath1, const char* tmpPath2, size_t& tmpPath1Size, size_t& tmpPath2Size );
    static size_t GetRequiredSizeForBuckets( const bool bounded, const uint32 numBuckets, const char* tmpPath1, const char* tmpPath2, const uint32 threadCount );
    static size_t GetRequiredSizeForBuckets( const bool bounded, const uint32 numBuckets, const size_t fxBlockSize, const size_t pairsBlockSize, const uint32 threadCount );
    
    static void ParseCommandLine( CliParser& cli, Config& cfg );

    static void PrintUsage();

private:
    DiskPlotContext   _cx;
    Config            _cfg;
};

