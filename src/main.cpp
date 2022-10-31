#include "plotting/GlobalPlotConfig.h"
#include "util/CliParser.h"
#include "plotdisk/DiskPlotter.h"
#include "plotmem/MemPlotter.h"
#include "Version.h"

#if PLATFORM_IS_UNIX
    #include <sys/resource.h>
#endif

static void ParseCommandLine( GlobalPlotConfig& cfg, int argc, const char* argv[] );
static void PrintUsage();

// See IOTester.cpp
void IOTestMain( GlobalPlotConfig& gCfg, CliParser& cli );
void IOTestPrintUsage();

// MemTester.cpp
void MemTestMain( GlobalPlotConfig& gCfg, CliParser& cli );
void MemTestPrintUsage();

// PlotValidator.cpp
void PlotValidatorMain( GlobalPlotConfig& gCfg, CliParser& cli );
void PlotValidatorPrintUsage();

// PlotComparer.cpp
void PlotCompareMain( GlobalPlotConfig& gCfg, CliParser& cli );
void PlotCompareMainPrintUsage();


enum class PlotterType
{
    None = 0,
    Ram,
    Disk
};

struct Plotter 
{
    PlotterType type;
    union {
        void* _ptr;
        DiskPlotter* disk;
        MemPlotter*  mem;
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

    _plotter = {};

    GlobalPlotConfig cfg;
    ParseCommandLine( cfg, --argc, ++argv );

    FatalIf( !_plotter._ptr, "No plot command chosen." );


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

        if( _plotter.type == PlotterType::Ram )
        {
            auto& plotter = *_plotter.mem;

            PlotRequest req = {};
            req.plotId      = plotId;
            req.memo        = plotMemo;
            req.memoSize    = plotMemoSize;
            req.outPath     = plotOutPath;
            req.IsFinalPlot = i == plotCount-1;
            
            plotter.Run( req );
        }
        else if( _plotter.type == PlotterType::Disk )
        {
            auto& plotter = *_plotter.disk;
            
            DiskPlotter::PlotRequest req;
            req.plotId       = plotId;
            req.plotMemo     = plotMemo;
            req.plotMemoSize = plotMemoSize;
            req.plotFileName = plotFileName;
            plotter.Plot( req );
        }
        else
        {
            Fatal( "Unknown plotter type." );
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

    DiskPlotter::Config diskCfg = {};
    MemPlotConfig       ramCfg  = {};

    while( cli.HasArgs() )
    {
        if( cli.ReadU32( cfg.threadCount, "-t", "--threads" ) )
            continue;
        else if( cli.ReadU32( cfg.plotCount, "-n", "--count" ) )
            continue;
        else if( cli.ReadStr( farmerPublicKey, "-f", "--farmer-key" ) )
            continue;
        else if( cli.ReadStr( poolPublicKey, "-p", "--pool-key" ) )
            continue;
        else if( cli.ReadStr( poolContractAddress, "-c", "--pool-contract" ) )
            continue;
        else if( cli.ReadSwitch( cfg.warmStart, "-w", "--warm-start" ) )
            continue;
        else if( cli.ReadStr( cfg.plotIdStr, "-i", "--plot-id" ) )
            continue;
        else if( cli.ReadStr( cfg.plotMemoStr, "--memo" ) )
            continue;
        else if( cli.ReadSwitch( cfg.showMemo, "--show-memo" ) )
            continue;
        else if( cli.ReadSwitch( cfg.disableNuma, "-m", "--no-numa" ) )
            continue;
        else if( cli.ReadSwitch( cfg.disableCpuAffinity, "--no-cpu-affinity" ) )
            continue;
        else if( cli.ArgConsume( "-v", "--verbose" ) )
        {
            Log::SetVerbose( true );
        }
        else if( cli.ArgConsume( "--memory" ) )
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
        else if( cli.ArgConsume( "--memory-json" ) )
        {
            // #TODO: Get this value from Memplotter
            const size_t requiredMem  = 416ull GB;
            const size_t availableMem = SysHost::GetAvailableSystemMemory();
            const size_t totalMem     = SysHost::GetTotalSystemMemory();

            Log::Line( "{ \"required\": %llu, \"total\": %llu, \"available\": %llu }",
                         requiredMem, totalMem, availableMem );

            exit( 0 );
        }
        else if( cli.ArgConsume( "--version" ) )
        {
            Log::Line( BLADEBIT_VERSION_STR );
            exit( 0 );
        }
        else if( cli.ArgConsume( "-h", "--help" ) )
        {
            PrintUsage();
            exit( 0 );
        }
        else if( cli.ArgConsume( "--about" ) )
        {
            Log::Line( "BladeBit Chia Plotter" );
            Log::Line( "Version      : %s", BLADEBIT_VERSION_STR   );
            Log::Line( "Git Commit   : %s", BLADEBIT_GIT_COMMIT    );
            Log::Line( "Compiled With: %s", BBGetCompilerVersion() );
            
            exit( 0 );
        }

        // Commands
        else if( cli.ArgConsume( "diskplot" ) )
        {
            // Increase the file size limit on linux
            #if PLATFORM_IS_UNIX
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

            // DiskPlotter::Config diskCfg;
            diskCfg.globalCfg = &cfg;
            DiskPlotter::ParseCommandLine( cli, diskCfg );

            _plotter.type = PlotterType::Disk;
            break;
        }
        else if( cli.ArgConsume( "ramplot" ) )
        {
            ramCfg.threadCount   = cfg.threadCount == 0 ? 
                                    SysHost::GetLogicalCPUCount() : 
                                    bbclamp( cfg.threadCount, 1u, SysHost::GetLogicalCPUCount() );
            ramCfg.warmStart     = cfg.warmStart;
            ramCfg.gCfg          = &cfg;

            _plotter.type = PlotterType::Ram;
            break;
        }
        else if( cli.ArgConsume( "iotest" ) )
        {
            IOTestMain( cfg, cli );
            exit( 0 );
        }
        else if( cli.ArgConsume( "memtest" ) )
        {
            MemTestMain( cfg, cli );
            exit( 0 );
        }
        else if( cli.ArgConsume( "validate" ) )
        {
            PlotValidatorMain( cfg, cli );
            exit( 0 );
        }
        else if( cli.ArgConsume( "plotcmp" ) )
        {
            PlotCompareMain( cfg, cli );
            exit( 0 );
        }
        else if( cli.ArgConsume( "help" ) )
        {
            if( cli.HasArgs() )
            {
                if( cli.ArgMatch( "diskplot" ) )
                    DiskPlotter::PrintUsage();
                else if( cli.ArgMatch( "iotest" ) )
                    IOTestPrintUsage();
                else if( cli.ArgMatch( "memtest" ) )
                    MemTestPrintUsage();
                else if( cli.ArgMatch( "validate" ) )
                    PlotValidatorPrintUsage();
                else if( cli.ArgMatch( "plotcmp" ) )
                    PlotCompareMainPrintUsage();
                else
                    Fatal( "Unknown command '%s'.", cli.Arg() );

                exit( 0 );
            }

            Log::Line( "help [<command>]" );
            Log::Line( "Display help text for a command." );
            Log::Line( "" );
            PrintUsage();
            exit( 0 );
        }
        // else if( cli.ArgMatch( "memplot" ) )
        // {

        // }
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
    Log::Line( "Bladebit Chia Plotter" );
    Log::Line( "Version      : %s", BLADEBIT_VERSION_STR   );
    Log::Line( "Git Commit   : %s", BLADEBIT_GIT_COMMIT    );
    Log::Line( "Compiled With: %s", BBGetCompilerVersion() );
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

    Log::Line( " Output path           : %s", cfg.outputFolder );

    Log::Line( "" );

    // Create plotter
    switch( _plotter.type )
    {
        case PlotterType::Disk:
            _plotter.disk = new DiskPlotter( diskCfg );
            break;

        case PlotterType::Ram:
            _plotter.mem  = new MemPlotter( ramCfg );
            break;
        
        default:
            Fatal( "No plotter chosen." );
            break;
    }
    Log::Line( "" );
}


//-----------------------------------------------------------
static const char* USAGE = "bladebit [GLOBAL_OPTIONS] <command> [COMMAND_OPTIONS]\n"
R"(
[COMMANDS]
 diskplot   : Create a plot by making use of a disk.
 ramplot    : Create a plot completely in-ram.
 iotest     : Perform a write and read test on a specified disk.
 memtest    : Perform a memory (RAM) copy test.
 validate   : Validates all entries in a plot to ensure they all evaluate to a valid proof.
 help       : Output this help message, or help for a specific command, if specified.

[GLOBAL_OPTIONS]:
 -h, --help           : Shows this message and exits.

 -t, --threads        : Maximum number of threads to use.
                        By default, this is set to the maximum number of logical cpus present.
 
 -n, --count          : Number of plots to create. Default = 1.

 -f, --farmer-key     : Farmer public key, specified in hexadecimal format.
                        *REQUIRED*

 -c, --pool-contract  : Pool contract address.
                        Use this if you are creating pool plots.
                        *A pool contract address or a pool public key must be specified.*

 -p, --pool-key       : Pool public key, specified in hexadecimal format.
                        Use this if you are creating OG plots.
                        Only used if a pool contract address is not specified.

 -w, --warm-start     : Touch all pages of buffer allocations before starting to plot.

 -i, --plot-id        : Specify a plot id for debugging.

 --memo               : Specify a plot memo for debugging.

 --show-memo          : Output the memo of the next plot the be plotted.

 -v, --verbose        : Enable verbose output.

 -m, --no-numa        : Disable automatic NUMA aware memory binding.
                        If you set this parameter in a NUMA system you
                        will likely get degraded performance.

 --no-cpu-affinity    : Disable assigning automatic thread affinity.
                        This is useful when running multiple simultaneous
                        instances of Bladebit as you can manually
                        assign thread affinity yourself when launching Bladebit.
 
 --memory             : Display system memory available, in bytes, and the 
                        required memory to run Bladebit, in bytes.
 
 --memory-json        : Same as --memory, but formats the output as json.

 --version            : Display current version.


[EXAMPLES]
bladebit --help

bladebit help diskplot

# Simple config:
bladebit -t 24 -f <farmer_pub_key> -c <contract_address> diskplot -t1 /my/temporary/plot/dir /my/output/dir

# With fine-grained control over threads per phase/section (see bladebit -h diskplot):
bladebit -t 30 -f <farmer_pub_key> -c <contract_address> diskplot --f1-threads 16 --c-threads 16 --p2-threads 8 -t1 /my/temporary/plot/dir /my/output/dir
)";

//-----------------------------------------------------------
void PrintUsage()
{
    Log::Line( USAGE );
}