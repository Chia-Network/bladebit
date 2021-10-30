#include "Plotter128.h"

#include "SysHost.h"
#include "Util.h"
#include "util/Log.h"
#include "threading/ThreadPool.h"

#include <thread>

#include "Plot128Phase1.h"

//-----------------------------------------------------------
Plotter128::Plotter128( const Config& cfg )
{

}

//-----------------------------------------------------------
Plotter128::~Plotter128()
{

}

//-----------------------------------------------------------
void Plotter128::Plot( const PlotRequest& req )
{
    ZeroMem( &_cx );

    // Allocate buffers
    const size_t totalBufferSize =
        16ull GB +   // y
        64ull GB +   // meta
        24ull GB +   // pairs
        16ull GB;    // sort extra

    Log::Line( "Allocating %llu GiB for memory.", totalBufferSize BtoGB );

    _cx.buffer = (byte*)SysHost::VirtualAlloc( totalBufferSize );
    if( !_cx.buffer )
        Fatal( "Buffer allocation failed." );

    {
        Log::Line( "Starting Phase 1" );
        Plot128Phase1 phase1( _cx );
        phase1.Run();
        Log::Line( "Finished Phase 1" );
    }
}
