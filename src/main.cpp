#include "plotting/GlobalPlotConfig.h"
#include "util/CliParser.h"
#include "plotdisk/DiskPlotter.h"
#include "Version.h"

#if PLATFORM_IS_LINUX
    #include <sys/resource.h>
#endif

static void ParseCommandLine( GlobalPlotConfig& cfg, int argc, const char* argv[] );

struct Plotter 
{
    union {
        void* _ptr;
        DiskPlotter* disk;
    };
};
    
Plotter _plotter;

//-----------------------------------------------------------
int main( int argc, const char* argv[] )
{
    // Install a crash handler to dump our stack traces
    SysHost::InstallCrashHandler();

#if _DEBUG
    Log::Line( "*** Warning: Debug mode is ENABLED ***" );
#endif

    ZeroMem( &_plotter );

    GlobalPlotConfig cfg;
    ParseCommandLine( cfg, --argc, ++argv );

    FatalIf( !_plotter._ptr, "No command chosen." );


    const int64 plotCount = cfg.plotCount > 0 ? (int64)cfg.plotCount : std::numeric_limits<int64>::max();
    // int64 failCount = 0;

    byte   plotId  [BB_PLOT_ID_LEN];
    byte   plotMemo[BB_PLOT_MEMO_MAX_SIZE];
    uint16 plotMemoSize = 0;

    char   plotIdStr[BB_PLOT_ID_LEN*2+1];


    // Prepare the output path
    size_t outputFolderLen = strlen( cfg.outputFolder );
    char*  plotOutPath     = new char[outputFolderLen + BB_PLOT_FILE_LEN_TMP + 2]; // + '/' + null terminator

    if( outputFolderLen )
    {
        memcpy( plotOutPath, cfg.outputFolder, outputFolderLen );

        // Add a trailing slash, if we need one.
        if( plotOutPath[outputFolderLen-1] != '/' && plotOutPath[outputFolderLen-1] != '\\' )
            plotOutPath[outputFolderLen++] = '/';
    }

    // Start plotting
    for( int64 i = 0; i < plotCount; i++ )
    {
        // Generate a plot id and memo
        PlotTools::GeneratePlotIdAndMemo( plotId, plotMemo, plotMemoSize,
                                          cfg.farmerPublicKey, cfg.poolPublicKey, cfg.poolContractPuzzleHash );

        // Apply debug plot id and/or memo
        if( cfg.plotIdStr )
            HexStrToBytes( cfg.plotIdStr, BB_PLOT_ID_LEN*2, plotId, BB_PLOT_ID_LEN );

        if( cfg.plotMemoStr )
        {
            const size_t memoLen = strlen( cfg.plotMemoStr );
            HexStrToBytes( cfg.plotMemoStr, memoLen, plotMemo, memoLen/2 );
        }

        // Convert plot id to string
        PlotTools::PlotIdToString( plotId, plotIdStr );

        // Set the plot file name
        const char* plotFileName = plotOutPath + outputFolderLen;
        PlotTools::GenPlotFileName( plotId, (char*)plotFileName );

        // Begin plot
        if( cfg.plotCount == 0 )
            Log::Line( "Generating plot %lld: %s", i+1, plotIdStr );
        else
            Log::Line( "Generating plot %lld / %u: %s", i+1, cfg.plotCount, plotIdStr );

        if( cfg.showMemo )
        {
            char plotMemoStr[BB_PLOT_MEMO_MAX_SIZE*2+1];

            size_t numEncoded = 0;
            BytesToHexStr( plotMemo, plotMemoSize, plotMemoStr, sizeof( plotMemoStr ) - 1, numEncoded );
            plotMemoStr[numEncoded*2] = 0;

            Log::Line( "Plot Memo: %s", plotMemoStr );
        }
        Log::Line( "" );

        if( _plotter.disk )
        {
            auto& plotter = *_plotter.disk;
            
            DiskPlotter::PlotRequest req;
            req.plotId       = plotId;
            req.plotMemo     = plotMemo;
            req.plotMemoSize = plotMemoSize;
            req.plotFileName = plotFileName;
            plotter.Plot( req );
        }
    }
}

//-----------------------------------------------------------
void ParseCommandLine( GlobalPlotConfig& cfg, int argc, const char* argv[] )
{
    CliParser cli( argc, argv );

    const char* farmerPublicKey     = nullptr;
    const char* poolPublicKey       = nullptr;
    const char* poolContractAddress = nullptr;

    while( cli.HasArgs() )
    {
        if( cli.ReadValue( cfg.threadCount, "-t", "--threads" ) )
            continue;
        else if( cli.ReadValue( cfg.plotCount, "-n", "--count" ) )
            continue;
        else if( cli.ReadValue( farmerPublicKey, "-f", "--farmer-key" ) )
            continue;
        else if( cli.ReadValue( poolPublicKey, "-p", "--pool-key" ) )
            continue;
        else if( cli.ReadValue( poolContractAddress, "-c", "--pool-contract" ) )
            continue;
        else if( cli.ReadSwitch( cfg.warmStart, "-w", "--warm-start" ) )
            continue;
        else if( cli.ReadValue( cfg.plotIdStr, "-i", "--plot-id" ) )
            continue;
        else if( cli.ReadValue( cfg.plotMemoStr, "--memo" ) )
            continue;
        else if( cli.ReadSwitch( cfg.showMemo, "--show-memo" ) )
            continue;
        else if( cli.ReadSwitch( cfg.disableNuma, "-m", "--no-numa" ) )
            continue;
        else if( cli.ReadSwitch( cfg.disableCpuAffinity, "--no-cpu-affinity" ) )
            continue;
        else if( cli.ArgMatch( "-v", "--verbose" ) )
        {
            Log::SetVerbose( true );
        }
        else if( cli.ArgMatch( "--memory" ) )
        {
            // #TODO: We should move the required part to the memplot command
            // #TODO: Get this value from Memplotter
            const size_t requiredMem  = 416ull GB;
            const size_t availableMem = SysHost::GetAvailableSystemMemory();
            const size_t totalMem     = SysHost::GetTotalSystemMemory();

            Log::Line( "required : %llu", requiredMem  );
            Log::Line( "total    : %llu", totalMem     );
            Log::Line( "available: %llu", availableMem );

            exit( 0 );
        }
        else if( cli.ArgMatch( "--memory-json" ) )
        {
            // #TODO: Get this value from Memplotter
            const size_t requiredMem  = 416ull GB;
            const size_t availableMem = SysHost::GetAvailableSystemMemory();
            const size_t totalMem     = SysHost::GetTotalSystemMemory();

            Log::Line( "{ \"required\": %llu, \"total\": %llu, \"available\": %llu }",
                         requiredMem, totalMem, availableMem );

            exit( 0 );
        }
        else if( cli.ArgMatch( "--version" ) )
        {
            Log::Line( BLADEBIT_VERSION_STR );
            exit( 0 );
        }
        else if( cli.ArgMatch( "-h", "--help" ) )
        {
            // #TODO: Print Help
            Log::Line( "Unimplemented!" );
            exit( 0 );
        }
        // Commands
        else if( cli.ArgMatch( "help" ) )
        {
            if( cli.HasArgs() )
            {
                // const char* cmd = cli.Arg();

                // #TODO: Print help for a specific command
            }

            // #TODO: Print main help
            Log::Line( "Unimplemented!" );
            exit( 0 );
        }
        // else if( cli.ArgMatch( "memplot" ) )
        // {

        // }
        else if( cli.ArgMatch( "diskplot" ) )
        {
            // Increase the file size limit on linux
            #if PLATFORM_IS_LINUX
                struct rlimit limit;
                getrlimit( RLIMIT_NOFILE, &limit );

                if( limit.rlim_cur < limit.rlim_max )
                {
                    Log::Line( "Increasing the file limit from %u to %u", limit.rlim_cur, limit.rlim_max );

                    limit.rlim_cur = limit.rlim_max;
                    if( setrlimit( RLIMIT_NOFILE, &limit ) != 0 )
                    {
                        const int err = errno;
                        Log::Line( "*** Warning: Failed to increase file limit to with error %d (0x%02x). Plotting may fail ***", err, err );
                    }
                }

            #endif

            
            cli.NextArg();
            DiskPlotter::Config diskCfg;
            DiskPlotter::ParseCommandLine( cli, diskCfg );

            diskCfg.globalCfg = &cfg;
            _plotter.disk = new DiskPlotter( diskCfg );

            break;
        }
        else
        {
            Fatal( "Unexpected argument '%s'", cli.Arg() );
        }
    }

    // The remainder should be output folders
    while( cli.HasArgs() )
    {
        cfg.outputFolder = cli.Arg();
        cli.NextArg();
    }

    // Validation
    FatalIf( farmerPublicKey == nullptr, "A farmer public key must be specified." );
    FatalIf( !KeyTools::HexPKeyToG1Element( farmerPublicKey, cfg.farmerPublicKey ),
        "Invalid farmer public key '%s'", farmerPublicKey );

    if( poolContractAddress )
    {
        cfg.poolContractPuzzleHash = new PuzzleHash();
        FatalIf( !PuzzleHash::FromAddress( *cfg.poolContractPuzzleHash, poolContractAddress ),
            "Invalid pool contract puzzle hash '%s'", poolContractAddress );
    }
    else if( poolPublicKey )
    {
        cfg.poolPublicKey = new bls::G1Element();
        FatalIf( !KeyTools::HexPKeyToG1Element( poolPublicKey, *cfg.poolPublicKey ),
                 "Invalid pool public key '%s'", poolPublicKey );
    }
    else
        Fatal( "Error: Either a pool public key or a pool contract address must be specified." );


    const uint maxThreads = SysHost::GetLogicalCPUCount();
    if( cfg.threadCount == 0 )
        cfg.threadCount = maxThreads;
    else if( cfg.threadCount > maxThreads )
    {
        Log::Write( "Warning: Lowering thread count from %u to %u, the native maximum.",
                    cfg.threadCount, maxThreads );

        cfg.threadCount = maxThreads;
    }

    FatalIf( cfg.outputFolder == nullptr, "An output folder must be specified." );


    if( cfg.plotIdStr )
    {
        const size_t len = strlen( cfg.plotIdStr );
        if( len < 64 && len != 66 )
            Fatal( "Invalid plot id." );
        
        if( len == 66 )
        {
            if( cfg.plotIdStr[0] == '0' && cfg.plotIdStr[1] == 'x' )
                cfg.plotIdStr += 2;
            else
                Fatal( "Invalid plot id." );
        }
    }
    if( cfg.plotMemoStr )
    {
        size_t len = strlen( cfg.plotMemoStr );
        if( len > 2 && cfg.plotMemoStr[0] == '0' && cfg.plotMemoStr[1] == 'x' )
        {
            cfg.plotMemoStr += 2;
            len -= 2;
        }
        
        if( len/2 != (48 + 48 + 32) && len != (32 + 48 + 32) )
            Fatal( "Invalid plot memo." );
    }

    // Config Summary
    Log::Line( "" );
    Log::Line( "[Global Plotting Config]" );
    if( cfg.plotCount == 0 )
        Log::Line( " Will create plots indefinitely." );
    else
        Log::Line( " Will create %u plots.", cfg.plotCount );

    Log::Line( " Thread count          : %d", cfg.threadCount );
    Log::Line( " Warm start enabled    : %s", cfg.warmStart ? "true" : "false" );
    Log::Line( " NUMA disabled         : %s", cfg.disableNuma ? "true" : "false" );
    Log::Line( " CPU affinity disabled : %s", cfg.disableCpuAffinity ? "true" : "false" );

    Log::Line( " Farmer public key     : %s", farmerPublicKey );
    

    if( poolContractAddress )
        Log::Line( " Pool contract address : %s", poolContractAddress );
    else if( cfg.poolPublicKey )
        Log::Line( " Pool public key       : %s", poolPublicKey   );

    Log::Line( "" );
}

