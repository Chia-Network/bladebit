#include "Commands.h"
#include "plotting/GlobalPlotConfig.h"
#include "threading/MTJob.h"
#include "tools/PlotReader.h"
#include "plotting/PlotValidation.h"
#include "plotting/f1/F1Gen.h"


struct Config
{
    GlobalPlotConfig* gCfg    = nullptr;

    uint64      proofCount = 100;
    const char* plotPath   = "";
};

void CmdPlotsCheckHelp();


//-----------------------------------------------------------
void CmdPlotsCheckMain( GlobalPlotConfig& gCfg, CliParser& cli )
{
    Config cfg = {};
    cfg.gCfg = &gCfg;

    while( cli.HasArgs() )
    {
        if( cli.ArgConsume( "-h", "--help" ) )
        {
            CmdPlotsCheckHelp();
            Exit( 0 );
        }
        else if( cli.ReadU64( cfg.proofCount, "-n", "--iterations" ) ) continue;
        else
            break;
    }

    FatalIf( !cli.HasArgs(), "Expected a path to a plot file." );
    {
        cfg.plotPath = cli.Arg();
        cli.NextArg();

        if( cli.HasArgs() )
        {
            Fatal( "Unexpected argument '%s'.", cli.Arg() );
            Exit( 1 );
        }
    }

    cfg.proofCount = std::max( cfg.proofCount, (uint64)1 );

    FilePlot plot;
    FatalIf( !plot.Open( cfg.plotPath ), "Failed to open plot file at '%s' with error %d.", cfg.plotPath, plot.GetError() );

    const uint32 threadCount = gCfg.threadCount == 0 ? SysHost::GetLogicalCPUCount() :
                                std::min( (uint32)MAX_THREADS, std::min( gCfg.threadCount, SysHost::GetLogicalCPUCount() ) );

    PlotReader reader( plot );
    reader.ConfigDecompressor( threadCount, gCfg.disableCpuAffinity );


    const uint32 k = plot.K();


    byte AlignAs(8) seed[BB_PLOT_ID_LEN] = {};
    SysHost::Random( seed, sizeof( seed ) );

    {
        std::string seedHex = BytesToHexStdString( seed, sizeof( seed ) );
        Log::Line( "Checking %llu random proofs with seed 0x%s...", (llu)cfg.proofCount, seedHex.c_str() );
    }
    Log::Line( "Plot compression level: %u", plot.CompressionLevel() );

    const uint64 f7Mask = (1ull << k) - 1;

    uint64 prevF7     = 0;
    uint64 proofCount = 0;

    uint64 proofXs[BB_PLOT_PROOF_X_COUNT];

    uint64 nextPercentage = 10;

    for( uint64 i = 0; i < cfg.proofCount; i++ )
    {
        const uint64 f7 = F1GenSingleForK( k, seed, prevF7 ) & f7Mask;
        prevF7 = f7;

        uint64 startP7Idx = 0;
        const uint64 nF7Proofs = reader.GetP7IndicesForF7( f7, startP7Idx );

        for( uint64 j = 0; j < nF7Proofs; j++ )
        {
            uint64 p7Entry;
            if( !reader.ReadP7Entry( startP7Idx + j, p7Entry ) )
            {
                // #TODO: Handle error
                continue;
            }

            const auto r = reader.FetchProof( p7Entry, proofXs );
            if( r == ProofFetchResult::OK )
            {
                // Convert to 
                uint64 outF7 = 0;
                if( PlotValidation::ValidateFullProof( k, plot.PlotId(), proofXs, outF7 ) )
                {
                    if( f7 == outF7 )
                    {
                        proofCount++;
                    }
                    else {}// #TODO: Handle error
                }
                else
                {
                    // #TODO: Handle error
                }

            }
            else
            {   
                // #TODO: Handle error
                continue;
            }
        }

        const double percent = i / (double)cfg.proofCount * 100.0;
        if( (uint64)percent == nextPercentage )
        {
            Log::Line( " %llu%%...", (llu)nextPercentage );
            nextPercentage += 10;
        }
    }

    Log::Line( "%llu / %llu (%.2lf%%) valid proofs found.",
        (llu)proofCount, (llu)cfg.proofCount, ((double)proofCount / cfg.proofCount) * 100.0 );
}

//-----------------------------------------------------------
void CmdPlotsCheckHelp()
{

}