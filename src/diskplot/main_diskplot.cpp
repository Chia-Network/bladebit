#include <stdlib.h>
#include "DiskPlotter.h"
#include "SysHost.h"

//-----------------------------------------------------------
int main( int arg, const char* argv[] )
{
    DiskPlotter::Config plotCfg;
    DiskPlotter plotter( plotCfg );

    DiskPlotter::PlotRequest req;
    plotter.Plot( req );

    exit( 0 );
}