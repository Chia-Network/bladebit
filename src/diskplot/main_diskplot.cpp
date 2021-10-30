#include <stdlib.h>
#include "DiskPlotter.h"
#include "SysHost.h"

struct GlobalConfig
{

};

struct DiskPlotConfig
{
    const char* tmpPath                 = nullptr;
    uint        workThreadCount         = 0;
    uint        ioThreadCount           = 1;
    uint        ioCommandQueueSize      = 512;
    size_t      workHeapSize            = 3ull   GB;
    size_t      f1WriteIntervalSize     = 128ull MB;
    size_t      groupWriteIntervalSize  = 32ull  MB;
    size_t      fxWriteIntervalSize     = 32ull  MB;
};

//-----------------------------------------------------------
int main( int argc, const char* argv[] )
{
    argc--;
    argv++;

    DiskPlotter::Config plotCfg;
    ZeroMem( &plotCfg );

    FatalIf( argc < 1, "Please specify a temporary path." );
    plotCfg.tmpPath = argv[argc - 1];

    DiskPlotter plotter( plotCfg );

    DiskPlotter::PlotRequest req;
    plotter.Plot( req );

    exit( 0 );
}