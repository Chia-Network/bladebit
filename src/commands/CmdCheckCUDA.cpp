#include "util/CliParser.h"
#include "plotting/GlobalPlotConfig.h"

#if BB_CUDA_ENABLED
    #include <cuda_runtime.h>
#endif
static const char _help[] = R"(cudacheck [OPTIONS]

Determines if the host has CUDA installed or not.
If it does detect CUDA, the exit code is 0 and
the number of CUDA devices is displayed.
Otherwise, the exit code is 1.

OPTIONS:
 -h, --help       : Display this help message and exit.
 -j, --json       : Output in JSON.
)";


void CmdCheckCUDAHelp()
{
    Log::Write( _help );
    Log::Flush();

}

void CmdCheckCUDA( GlobalPlotConfig& gCfg, CliParser& cli )
{
    bool json = false;

    while( cli.HasArgs() )
    {
        if( cli.ReadSwitch( json, "-j", "--json" ) )
        {
            continue;
        }
        if( cli.ArgConsume( "-h", "--help" ) )
        {
            CmdCheckCUDAHelp();
            Exit(0);
        }
        else
        {
            Fatal( "Unexpected argument '%s'", cli.Arg() );
        }
    }

    int  deviceCount = 0;
    bool success     = false;
    
    #if BB_CUDA_ENABLED
        success = cudaGetDeviceCount( &deviceCount ) == cudaSuccess;
    #endif

    if( json )
    {
        Log::Write( "{ \"enabled\": %s, \"device_count\": %d }",
            success ? "true" : "false",
            deviceCount );
    }
    else
    {
        if( success )
            Log::Line( "CUDA is enabled with %d devices.", deviceCount );
        else
            Log::Line( "CUDA is NOT available." );

    }

    Log::Flush();
    Exit( deviceCount > 0 ? 0 : -1 );
}
