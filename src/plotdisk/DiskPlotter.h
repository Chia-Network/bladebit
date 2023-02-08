#pragma once

#include "DiskPlotContext.h"
#include "plotting/GlobalPlotConfig.h"
#include "plotting/IPlotter.h"

class DiskPlotter : public IPlotter
{
public:
    using Config = DiskPlotConfig;

    // struct PlotRequest
    // {
    //     const byte*  plotId;
    //     const byte*  plotMemo;
    //     uint16       plotMemoSize;
    //     const char*  plotFileName;
    // };

public:
    DiskPlotter();

    void ParseCLI( const GlobalPlotConfig& gCfg, CliParser& cli ) override;
    void Init()  override;
    void Run( const PlotRequest& req ) override;


    static bool   GetTmpPathsBlockSizes( const char* tmpPath1, const char* tmpPath2, size_t& tmpPath1Size, size_t& tmpPath2Size );
    static size_t GetRequiredSizeForBuckets( const bool bounded, const uint32 numBuckets, const char* tmpPath1, const char* tmpPath2, const uint32 threadCount );
    static size_t GetRequiredSizeForBuckets( const bool bounded, const uint32 numBuckets, const size_t fxBlockSize, const size_t pairsBlockSize, const uint32 threadCount );

    static void PrintUsage();

private:
    DiskPlotContext   _cx  = {};
    Config            _cfg = {};
};

