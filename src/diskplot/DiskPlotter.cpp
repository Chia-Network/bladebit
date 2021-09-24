#include "DiskPlotter.h"
#include "util/Log.h"
#include "Util.h"

#include "DiskPlotPhase1.h"

//-----------------------------------------------------------
DiskPlotter::DiskPlotter()
{

}

//-----------------------------------------------------------
DiskPlotter::DiskPlotter( const Config cfg )
{
    ZeroMem( &_cx );


}

//-----------------------------------------------------------
void DiskPlotter::Plot( const PlotRequest& req )
{
    Log::Line( "Started plot." );

    {
        DiskPlotPhase1 phase1( _cx );
        phase1.Run();
    }

    Log::Line( "Finished plotting in %.2lf seconds.", 0.0 );
}
