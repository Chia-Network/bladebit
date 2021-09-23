#include "DiskPlotter.h"
#include "util/Log.h"
#include "Util.h"

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

    Log::Line( "Finished plotting in %.2lf seconds.", 0.0 );
}
