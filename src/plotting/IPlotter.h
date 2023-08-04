#pragma once

class CliParser;
struct GlobalPlotConfig;
struct PlotRequest;

class IPlotter
{
public:
    virtual void ParseCLI( const GlobalPlotConfig& gCfg, CliParser& cli ) = 0;
    virtual void Init() = 0;
    virtual void Run( const PlotRequest& req ) = 0;
};
