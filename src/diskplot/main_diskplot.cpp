#include <stdlib.h>
#include "DiskPlotter.h"
#include "SysHost.h"

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