#include <stdlib.h>
#include "DiskPlotter.h"
#include "SysHost.h"

//-----------------------------------------------------------
int main( int arg, const char* argv[] )
{
    DiskPlotter::Config plotCfg;
    ZeroMem( &plotCfg );

    DiskPlotter plotter( plotCfg );

    DiskPlotter::PlotRequest req;
    plotter.Plot( req );

    exit( 0 );
}