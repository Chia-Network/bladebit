#include <stdlib.h>
#include "DiskPlotter.h"
#include "SysHost.h"

//-----------------------------------------------------------
int main( int argc, const char* argv[] )
{
    DiskPlotter::Config plotCfg;
    ZeroMem( &plotCfg );

    plotCfg.tmpPath = argv[argc - 1];

    DiskPlotter plotter( plotCfg );

    DiskPlotter::PlotRequest req;
    plotter.Plot( req );

    exit( 0 );
}