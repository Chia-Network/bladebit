#include <stdlib.h>
#include "Plotter128.h"

//-----------------------------------------------------------
int main( int arg, const char* argv[] )
{
    Plotter128::Config plotCfg;
    Plotter128 plotter( plotCfg );

    Plotter128::PlotRequest req;
    plotter.Plot( req );

    exit( 0 );
}