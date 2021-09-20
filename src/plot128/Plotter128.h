#pragma once

#include "Plot128Context.h"

// 128GiB of RAM required to plot.
// This plotter is a hybrid disk/ram plotter
// targeted ad desktop machines stocked-up with 128 GB of RAM.
class Plotter128
{
public:
    struct Config
    {

    };

    struct PlotRequest
    {

    };

public:
    Plotter128( const Config& cfg );
    ~Plotter128();

    void Plot( const PlotRequest& req );

private:
    Plot128Context _cx;
};