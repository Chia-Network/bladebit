#include "PlotChecker.h"
#include "tools/PlotReader.h"
#include "plotting/PlotValidation.h"
#include "harvesting/GreenReaper.h"
#include "plotting/f1/F1Gen.h"

class PlotCheckerImpl : public PlotChecker
{
    PlotCheckerConfig _cfg;
    bool              _lastPlotDeleted = false;
public:

    //-----------------------------------------------------------
    PlotCheckerImpl( PlotCheckerConfig& cfg )
        : _cfg( cfg )
    {}

    //-----------------------------------------------------------
    ~PlotCheckerImpl() override = default;

    //-----------------------------------------------------------
    void CheckPlot( const char* plotPath, PlotCheckResult* outResult ) override
    {
        _lastPlotDeleted = false;

        PlotCheckResult result{};
        PerformPlotCheck( plotPath, result );

        if( !result.error.empty() )
        {
            if( !_cfg.silent )
            {
                Log::Line( "An error occured checking the plot: %s.", result.error.c_str() );

                if( _cfg.deletePlots )
                    Log::Line( "Any actions against plot '%s' will be ignored.", plotPath );
            }
        }

        // Check threshold for plot deletion
        const double passRate = result.proofCount / (double)result.checkCount;

        // Print stats
        if( !_cfg.silent )
        {
            std::string seedHex = BytesToHexStdString( result.seedUsed, sizeof( result.seedUsed ) );
            Log::Line( "Seed used: 0x%s", seedHex.c_str() );
            Log::Line( "Proofs requested/fetched: %llu / %llu ( %.3lf%% )", result.proofCount, result.checkCount, passRate * 100.0 );

            if( result.proofFetchFailCount > 0 )
                Log::Line( "Proof fetches failed    : %llu ( %.3lf%% )", result.proofFetchFailCount, result.proofFetchFailCount / (double)result.checkCount * 100.0 );
            if( result.proofValidationFailCount > 0 )
                Log::Line( "Proof validation failed : %llu ( %.3lf%% )", result.proofValidationFailCount, result.proofValidationFailCount / (double)result.checkCount * 100.0 );
            Log::NewLine();
        }

        // Delete the plot if it's below the set threshold
        if( _cfg.deletePlots )
        {
            if( result.proofFetchFailCount > 0 || passRate < _cfg.deleteThreshold )
            {
                if( !_cfg.silent )
                {
                    if( result.proofFetchFailCount > 0 )
                    {
                        Log::Line( "WARNING: Deleting plot '%s' as it failed to fetch some proofs. This might indicate corrupt plot file.", plotPath );
                    }
                    else
                    {
                        Log::Line( "WARNING: Deleting plot '%s' as it is below the proof threshold: %.3lf / %.3lf.",
                            plotPath, passRate, _cfg.deleteThreshold );
                    }
                    Log::NewLine();
                }

                remove( plotPath );
                result.deleted   = true;
                _lastPlotDeleted = true;
            }
            else
            {
                Log::Line( "Plot is OK. It passed the proof threshold of %.3lf%%", _cfg.deleteThreshold * 100.0 );
                Log::NewLine();
            }
        }

        if( outResult )
            *outResult = result;
    }
    
    //-----------------------------------------------------------
    void PerformPlotCheck( const char* plotPath, PlotCheckResult& result )
    {
        FilePlot plot;
        if( !plot.Open( plotPath ) )
        {
            std::stringstream err; err << "Failed to open plot file at '" << plotPath << "' with error " << plot.GetError() << ".";
            result.error = err.str();
            return;
        }

        const uint32 threadCount = _cfg.threadCount == 0 ? SysHost::GetLogicalCPUCount() :
                                        std::min( (uint32)MAX_THREADS, std::min( _cfg.threadCount, SysHost::GetLogicalCPUCount() ) );

        const bool useGpu = plot.CompressionLevel() > 0 && !_cfg.noGpu;

        PlotReader reader( plot );

        if( _cfg.grContext )
            reader.AssignDecompressionContext( _cfg.grContext );
        else
            reader.ConfigDecompressor( threadCount, _cfg.disableCpuAffinity, 0, useGpu, (int)_cfg.gpuIndex );

        const uint32 k = plot.K();

        byte AlignAs(8) seed[BB_PLOT_ID_LEN] = {};

        if( !_cfg.hasSeed )
            SysHost::Random( seed, sizeof( seed ) );
        else
            memcpy( seed, _cfg.seed, sizeof( _cfg.seed ) );

        {
            std::string seedHex = BytesToHexStdString( seed, sizeof( seed ) );
            if( !_cfg.silent )
                Log::Line( "Checking %llu random proofs with seed 0x%s...", (llu)_cfg.proofCount, seedHex.c_str() );
        }

        if( !_cfg.silent )
            Log::Line( "Plot compression level: %u", plot.CompressionLevel() );

        if( !_cfg.grContext && plot.CompressionLevel() > 0 && useGpu )
        {
            const bool hasGPU = grHasGpuDecompressor( reader.GetDecompressorContext() );
            if( hasGPU && !_cfg.silent )
                Log::Line( "Using GPU for decompression." );
            else if( !_cfg.silent )
                Log::Line( "No GPU was selected for decompression." );
        }

        const uint64 f7Mask = (1ull << k) - 1;

        uint64 prevF7     = 0;
        uint64 proofCount = 0;

        uint64 proofXs[BB_PLOT_PROOF_X_COUNT];

        uint64 nextPercentage = 10;

        for( uint64 i = 0; i < _cfg.proofCount; i++ )
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
                    result.proofFetchFailCount ++;
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
                    else
                    {
                        result.proofValidationFailCount++;
                    }
                }
                else
                {
                    if( r != ProofFetchResult::NoProof )
                        result.proofFetchFailCount ++;
                }
            }

            const double percent = i / (double)_cfg.proofCount * 100.0;
            if( (uint64)percent == nextPercentage )
            {
                if( !_cfg.silent )
                    Log::Line( " %llu%%...", (llu)nextPercentage );
                nextPercentage += 10;
            }
        }

        result.checkCount = _cfg.proofCount;
        result.proofCount = proofCount;
        result.error.clear();
        static_assert( sizeof(PlotCheckResult::seedUsed) == sizeof(seed) );
        memcpy( result.seedUsed, seed, sizeof( result.seedUsed ) );
    }

    //-----------------------------------------------------------
    bool LastPlotDeleted() override
    {
        return _lastPlotDeleted;
    }
};

//-----------------------------------------------------------
PlotChecker* PlotChecker::Create( PlotCheckerConfig& cfg )
{
    return new PlotCheckerImpl( cfg );
}
