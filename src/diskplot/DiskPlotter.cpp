#include "DiskPlotter.h"
#include "util/Log.h"
#include "Util.h"

#include "DiskPlotPhase1.h"
#include "SysHost.h"

//-----------------------------------------------------------
DiskPlotter::DiskPlotter()
{

}

//-----------------------------------------------------------
DiskPlotter::DiskPlotter( const Config cfg )
{
    ZeroMem( &_cx );

    _cx.bufferSizeBytes = 2 GB + 512 MB;
    
    Log::Line( "Allocating a working buffer of %.2lf MiB", (double)_cx.bufferSizeBytes BtoMB );

    _cx.workBuffer = (byte*)SysHost::VirtualAlloc( _cx.bufferSizeBytes );
    FatalIf( !_cx.workBuffer, "Failed to allocated work buffer. Make sure you have enough free memory." );

    _cx.threadPool    = new ThreadPool( _cx.threadCount, ThreadPool::Mode::Fixed, false );
    
    // Test values
    _cx.threadCount   = SysHost::GetLogicalCPUCount();
    _cx.diskFlushSize = 128 MB;
    
    static const byte plotId[32] = {
        22, 24, 11, 3, 1, 15, 11, 6, 
        23, 22, 22, 24, 11, 3, 1, 15,
        11, 6, 23, 22, 22, 24, 11, 3,
        1, 15, 11, 6, 23, 22, 5, 28
    };

    static const uint plotMemoSize     = 128;
    static byte plotMemo[plotMemoSize] = { 0 };

    _cx.plotId       = plotId;
    _cx.plotMemoSize = plotMemoSize;
    _cx.plotMemo     = plotMemo;
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
