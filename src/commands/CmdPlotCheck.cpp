#include "CmdPlotCheckInternal.h"
#include "threading/MTJob.h"
#include "tools/PlotReader.h"
#include "plotting/PlotValidation.h"
#include "plotting/f1/F1Gen.h"
#include "harvesting/GreenReaper.h"


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
    {
        cfg.plotPath = cli.Arg();
        cli.NextArg();

        if( cli.HasArgs() )
        {
            Fatal( "Unexpected argument '%s'.", cli.Arg() );
            Exit( 1 );
        }
    }

    PlotCheckResult result{};
    if( !RunPlotsCheck( cfg, gCfg.threadCount, gCfg.disableCpuAffinity, &result ) )
    {
        Fatal( result.error.c_str() );
    }

    Log::Line( "%llu / %llu (%.2lf%%) valid proofs found.",
        (llu)result.proofCount, (llu)cfg.proofCount, ((double)result.proofCount / cfg.proofCount) * 100.0 );
}

//-----------------------------------------------------------
bool RunPlotsCheck( PlotCheckConfig& cfg, uint32 threadCount, const bool disableCpuAffinity, PlotCheckResult* outResult )
{
    if( outResult )
    {
        outResult->proofFetchFailCount      = 0;
        outResult->proofValidationFailCount = 0;
    }

    cfg.proofCount = std::max( cfg.proofCount, (uint64)1 );

    FilePlot plot;
    if( !plot.Open( cfg.plotPath ) )
    {
        if( outResult )
        {
            std::stringstream err; err << "Failed to open plot file at '" << cfg.plotPath << "' with error " << plot.GetError() << ".";
            outResult->error = err.str();
        }
        
        return false;
    }

    threadCount = threadCount == 0 ? SysHost::GetLogicalCPUCount() :
                                std::min( (uint32)MAX_THREADS, std::min( threadCount, SysHost::GetLogicalCPUCount() ) );

    const bool useGpu = plot.CompressionLevel() > 0 && !cfg.noGpu;

    PlotReader reader( plot );

    if( cfg.grContext )
        reader.AssignDecompressionContext( cfg.grContext );
    else
        reader.ConfigDecompressor( threadCount, disableCpuAffinity, 0, useGpu, (int)cfg.gpuIndex );

    const uint32 k = plot.K();

    byte AlignAs(8) seed[BB_PLOT_ID_LEN] = {};

    if( !cfg.hasSeed )
        SysHost::Random( seed, sizeof( seed ) );
    else
        memcpy( seed, cfg.seed, sizeof( cfg.seed ) );

    {
        std::string seedHex = BytesToHexStdString( seed, sizeof( seed ) );
        if( !cfg.silent )
            Log::Line( "Checking %llu random proofs with seed 0x%s...", (llu)cfg.proofCount, seedHex.c_str() );
    }

    if( !cfg.silent )
        Log::Line( "Plot compression level: %u", plot.CompressionLevel() );

    if( !cfg.grContext && plot.CompressionLevel() > 0 && useGpu )
    {
        const bool hasGPU = grHasGpuDecompressor( reader.GetDecompressorContext() );
        if( hasGPU && !cfg.silent )
            Log::Line( "Using GPU for decompression." );
        else if( !cfg.silent )
            Log::Line( "No GPU was selected for decompression." );
    }

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
                if( outResult )
                    outResult->proofFetchFailCount ++;

                continue;
            }

            const auto r = reader.FetchProof( p7Entry, proofXs );
            if( r == ProofFetchResult::OK )
            {
                // Convert to 
                uint64 outF7 = 0;
                if( PlotValidation::ValidateFullProof( k, plot.PlotId(), proofXs, outF7 ) && outF7 == f7 )
                {
                    proofCount++;
                }
                else if( outResult )
                {
                    outResult->proofValidationFailCount++;
                }
            }
            else
            {
                if( r != ProofFetchResult::NoProof && outResult )
                    outResult->proofFetchFailCount ++;
            }
        }

        const double percent = i / (double)cfg.proofCount * 100.0;
        if( (uint64)percent == nextPercentage )
        {
            if( !cfg.silent )
                Log::Line( " %llu%%...", (llu)nextPercentage );
            nextPercentage += 10;
        }
    }

    if( outResult )
    {
        outResult->checkCount = cfg.proofCount;
        outResult->proofCount = proofCount;
        outResult->error.clear();
        static_assert( sizeof(PlotCheckResult::seedUsed) == sizeof(seed) );
        memcpy( outResult->seedUsed, seed, sizeof( outResult->seedUsed ) );
    }

    return true;
}

//-----------------------------------------------------------
void CmdPlotsCheckHelp()
{

}
