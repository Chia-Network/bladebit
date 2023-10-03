#include "threading/MTJob.h"
#include "util/CliParser.h"
#include "tools/PlotReader.h"
#include "plotting/GlobalPlotConfig.h"
#include "plotting/PlotValidation.h"
#include "plotting/f1/F1Gen.h"
#include "tools/PlotChecker.h"
#include "harvesting/GreenReaper.h"


struct PlotCheckConfig
{
    GlobalPlotConfig* gCfg    = nullptr;

    uint64                   proofCount = 100;
    std::vector<const char*> plotPaths{};
    byte                     seed[BB_PLOT_ID_LEN]{};
    bool                     hasSeed    = false;
    bool                     noGpu      = false;
    int32                    gpuIndex   = -1;
};

void CmdPlotsCheckHelp();

//-----------------------------------------------------------
void CmdPlotsCheckMain( GlobalPlotConfig& gCfg, CliParser& cli )
{
    PlotCheckConfig cfg = {};
    cfg.gCfg = &gCfg;

    while( cli.HasArgs() )
    {
        if( cli.ArgConsume( "-h", "--help" ) )
        {
            CmdPlotsCheckHelp();
            Exit( 0 );
        }
        if( cli.ReadHexStrAsBytes( cfg.seed, sizeof( cfg.seed ), "-s", "--seed" ) )
        {
            cfg.hasSeed = true;
        }
        else if( cli.ReadU64( cfg.proofCount, "-n", "--iterations" ) ) continue;
        else if( cli.ReadSwitch( cfg.noGpu, "-g", "--no-gpu" ) ) continue;
        else if( cli.ReadI32( cfg.gpuIndex, "-d", "--device" ) ) continue;
        else
            break;
    }

    FatalIf( !cli.HasArgs(), "Expected a path to a plot file." );
    do
    {
        cfg.plotPaths.push_back( cli.Arg() );
        cli.NextArg();
    }
    while( cli.HasArgs() );

    
    // GreenReaperContext* grContext = nullptr;
    // {
    //     // Pre-create decompressor here?
    //     grCreateContext( &grcontext, grCfg, sizeof( GreenReaperConfig ) )
    // }

        // const bool hasGPU = grHasGpuDecompressor( reader.GetDecompressorContext() );
        // if( hasGPU && !cfg.silent )
        //     Log::Line( "Using GPU for decompression." );
        // else if( !cfg.silent )
        //     Log::Line( "No GPU was selected for decompression." );

    PlotCheckerConfig checkerCfg{
        .proofCount         = cfg.proofCount,
        .noGpu              = cfg.noGpu,
        .gpuIndex           = cfg.gpuIndex,
        .threadCount        = gCfg.threadCount,
        .disableCpuAffinity = gCfg.disableCpuAffinity,
        .silent             = false,
        .hasSeed            = cfg.hasSeed,
        .deletePlots        = false,
        .deleteThreshold    = 0.0
    };

    static_assert( sizeof( checkerCfg.seed ) == sizeof( cfg.seed ) );
    if( cfg.hasSeed )
        memcpy( checkerCfg.seed, cfg.seed, sizeof( checkerCfg.seed ) );

    ptr<PlotChecker> checker( PlotChecker::Create( checkerCfg ) );

    for( auto* plotPath : cfg.plotPaths )
    {
        PlotCheckResult result{};
        checker->CheckPlot( plotPath, &result );
        if( !result.error.empty() )
        {
            Fatal( result.error.c_str() );
        }

        Log::NewLine();

        // Log::Line( "%llu / %llu (%.2lf%%) valid proofs found.",
        //     (llu)result.proofCount, (llu)cfg.proofCount, ((double)result.proofCount / cfg.proofCount) * 100.0 );
    }

}

//-----------------------------------------------------------
void CmdPlotsCheckHelp()
{

}
