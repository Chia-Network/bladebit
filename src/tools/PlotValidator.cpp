#include "plotting/CTables.h"
#include "ChiaConsts.h"
#include "util/Log.h"
#include "util/BitView.h"
#include "io/FileStream.h"
#include "PlotTools.h"
#include "PlotReader.h"
#include "plotting/PlotTools.h"
#include "plotmem/LPGen.h"
#include "pos/chacha8.h"
#include "b3/blake3.h"
#include "threading/MTJob.h"
#include "util/CliParser.h"
#include "plotting/GlobalPlotConfig.h"
#include <mutex>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#define PROOF_X_COUNT       64
#define MAX_K_SIZE          48
#define MAX_META_MULTIPLIER 4
#define MAX_Y_BIT_SIZE      ( MAX_K_SIZE + kExtraBits )
#define MAX_META_BIT_SIZE   ( MAX_K_SIZE * MAX_META_MULTIPLIER )
#define MAX_FX_BIT_SIZE     ( MAX_Y_BIT_SIZE + MAX_META_BIT_SIZE + MAX_META_BIT_SIZE )

#define COLOR_NONE       "\033[0m"
#define COLOR_RED        "\033[31m"
#define COLOR_GREEN      "\033[32m"
#define COLOR_RED_BOLD   "\033[1m\033[31m"
#define COLOR_GREEN_BOLD "\033[1m\033[32m"

typedef Bits<MAX_Y_BIT_SIZE>    YBits;
typedef Bits<MAX_META_BIT_SIZE> MetaBits;
typedef Bits<MAX_FX_BIT_SIZE>   FxBits;

// #TODO: Add C1 & C2 table validation

//-----------------------------------------------------------
const char USAGE[] = R"(validate [OPTIONS] <plot_path>

Validates all of a plot's values to ensure they all contain valid proofs.

[NOTES]
You can specify the thread count in the bladebit global option '-t'.

[ARGUMENTS]
<plot_path>   : Path to the plot file to be validated.

[OPTIOINS]
 -m, --in-ram : Loads the whole plot file into memory before validating.

 -o, --offset : Percentage offset at which to start validating.
                Ex (start at 50%): bladebit validate -o 50 /path/to/my/plot

 -u, --unpack : Decompress the plot into memory before validating.
                This decreases validation time substantially but
                it requires around 128GiB of RAM for k=32.
                This is only supported for plots with k=32 and below.

 --f7 <f7>    : Specify an f7 to find and validate in the plot.

 -h, --help   : Print this help message and exit.
)";

void PlotValidatorPrintUsage()
{
    Log::Line( USAGE );
}



struct UnpackedK32Plot
{
    // Table 1 == X's in line point form
    // Table 6 is sorted on p7
    IPlotFile*   plot = nullptr;
    Span<Pair>   tables[6];
    Span<uint32> f7;

    //-----------------------------------------------------------
    inline const Span<Pair> Table( const TableId table ) const
    {
        if( table >= TableId::Table7 )
        {
            ASSERT( 0 );
            return Span<Pair>();
        }

        return tables[(int)table];
    }
    
    static UnpackedK32Plot Load( IPlotFile** plotFile, ThreadPool& pool, uint32 threadCount );
    bool FetchProof( const  uint64 index, uint64 fullProofXs[PROOF_X_COUNT] );
};


static void GetProofF1( uint32 k, const byte plotId[BB_PLOT_ID_LEN], uint64 fullProofXs[PROOF_X_COUNT], uint64 fx[PROOF_X_COUNT] );

template<bool Use64BitLpToSquare>
static bool FetchProof( PlotReader& plot, uint64 t6LPIndex, uint64 fullProofXs[PROOF_X_COUNT] );

static void GetProofForChallenge( const char* plotPath, const char* challengeHex );
static bool ValidateFullProof( const uint32 k, const byte plotId[BB_PLOT_ID_LEN], uint64 fullProofXs[PROOF_X_COUNT], uint64& outF7 );
static void ReorderProof( PlotReader& plot, uint64 fullProofXs[PROOF_X_COUNT] );
static void GetProofF1( uint32 k, const byte plotId[BB_PLOT_ID_LEN], uint64 fullProofXs[PROOF_X_COUNT], uint64 fx[PROOF_X_COUNT] );

static uint64 BytesToUInt64( const byte bytes[8] );
static uint64 SliceUInt64FromBits( const byte* bytes, uint32 bitOffset, uint32 bitCount );

static bool FxMatch( uint64 yL, uint64 yR );

static void FxGen( const TableId table, const uint32 k, 
                   const uint64 y, const MetaBits& metaL, const MetaBits& metaR,
                   uint64& outY, MetaBits& outMeta );

static bool ValidatePlot( const ValidatePlotOptions& options );

static uint64 ValidateInMemory( UnpackedK32Plot& plot, ThreadPool& pool );

// Thread-safe log
static std::mutex _logLock;
static void TVLog( const uint32 id, const char* msg, va_list args );
// static void TLog( const uint32 id, const char* msg, ... );

struct ValidateJob : MTJob<ValidateJob>
{
    IPlotFile*       plotFile;
    UnpackedK32Plot* unpackedPlot;  // If set, this will be used instead
    uint64           failCount;
    float            startOffset;

    void Run() override;
    void Log( const char* msg, ... );
};


//-----------------------------------------------------------
void PlotValidatorMain( GlobalPlotConfig& gCfg, CliParser& cli )
{
    ValidatePlotOptions opts;

    const char* challenge = nullptr;
    int64       f7        = -1;

    while( cli.HasArgs() )
    {
        if( cli.ReadSwitch( opts.inRAM, "-m", "--in-ram" ) )
            continue;
        else if( cli.ReadSwitch( opts.unpacked, "-u", "--unpack" ) )
            continue;
        else if( cli.ReadF32( opts.startOffset, "-o", "--offset" ) )
            continue;
        else if( cli.ReadStr( challenge, "--prove" ) )
            continue;
        else if( cli.ReadI64( f7, "--f7" ) )    // Same as proof, but the challenge is made from an f7
            continue;
        else if( cli.ArgConsume( "-h", "--help" ) )
        {
            PlotValidatorPrintUsage();
            exit( 0 );
        }
        else if( cli.IsLastArg() )
        {
            opts.plotPath = cli.ArgConsume();
        }
        else
        {
            Fatal( "Unexpected argument '%s'.", cli.Arg() );
        }
    }

    // Check for f7
    if( f7 >= 0 )
    {
        challenge = new char[65];
        sprintf( (char*)challenge, "%08llx", f7 );
        memset( (void*)(challenge+8), '0', 64-8 );
        ((char*)challenge)[64] = 0;
    }

    // Check for challenge
    if( challenge != nullptr )
    {
        GetProofForChallenge( opts.plotPath.c_str(), challenge );
        Exit( 0 );
    }

    const uint32 maxThreads = SysHost::GetLogicalCPUCount();

    opts.threadCount = gCfg.threadCount == 0 ? maxThreads : std::min( maxThreads, gCfg.threadCount );
    opts.startOffset = std::max( std::min( opts.startOffset / 100.f, 100.f ), 0.f );

    ValidatePlot( opts );

    exit( 0 );
}

//-----------------------------------------------------------
bool ValidatePlot( const ValidatePlotOptions& options )
{
    LoadLTargets();

    const uint32 threadCount = options.threadCount;

    IPlotFile*  plotFile  = nullptr;
    IPlotFile** plotFiles = new IPlotFile*[threadCount];

    if( options.inRAM && !options.unpacked )
    {
        auto* memPlot = new MemoryPlot();
        plotFile = memPlot;
        
        Log::Line( "Reading plot file into memory..." );
        if( memPlot->Open( options.plotPath.c_str() ) )
        {
            for( uint32 i = 0; i < threadCount; i++ )
                plotFiles[i] = new MemoryPlot( *memPlot );
        }
    }
    else
    {
        auto* filePlot = new FilePlot();
        plotFile = filePlot;

        if( filePlot->Open( options.plotPath.c_str() ) )
        {
            for( uint32 i = 0; i < threadCount; i++ )
                plotFiles[i] = new FilePlot( *filePlot );
        }
    }

    FatalIf( !plotFile->IsOpen(), "Failed to open plot at path '%s'.", options.plotPath.c_str() );
    FatalIf( options.unpacked && plotFile->K() != 32, "Unpacked plots are only supported for k=32 plots." );

    Log::Line( "Validating plot %s", options.plotPath.c_str() );
    Log::Line( "K               : %u", plotFile->K() );
    Log::Line( "Unpacked        : %s", options.unpacked? "true" : "false" );;

    const uint64 plotC3ParkCount = plotFile->TableSize( PlotTable::C1 ) / sizeof( uint32 ) - 1;
    Log::Line( "Maximum C3 Parks: %llu", plotC3ParkCount );
    Log::Line( "" );


    // Duplicate the plot file,     
    ThreadPool pool( threadCount );
    
    UnpackedK32Plot unpackedPlot;
    if( options.unpacked )
    {
        unpackedPlot      = UnpackedK32Plot::Load( plotFiles, pool, threadCount );
        unpackedPlot.plot = plotFile;

        const auto timer = TimerBegin();
        const uint64 failedCount = ValidateInMemory( unpackedPlot, pool );
        const double elapsed = TimerEnd( timer );

        const uint64 min = (uint64)(elapsed / 60);
        const uint64 sec = (uint64)( elapsed - (double)(min * 60) );

        Log::Line( "" );
        Log::Line( "Finished validating plot in %.lf seconds ( %llu:%llu min ).", 
            elapsed, min, sec );

        Log::Line( "[ %s%s%s ] Valid Proofs: %llu / %llu", 
            failedCount == 0 ? COLOR_GREEN_BOLD : COLOR_RED_BOLD,
            failedCount ? "FAILED" : "SUCCESS", COLOR_NONE,
            unpackedPlot.f7.Length() - failedCount, unpackedPlot.f7.Length() );

        exit( failedCount == 0 ? 0 : 1 );
    }

    MTJobRunner<ValidateJob> jobs( pool );
    
    for( uint32 i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];

        job.plotFile     = plotFiles[i];
        job.unpackedPlot = options.unpacked ? &unpackedPlot : nullptr;
        job.startOffset  = options.startOffset;
        job.failCount    = 0;
    }

    jobs.Run( threadCount );

    uint64 proofFailCount = 0;
    for( uint32 i = 0; i < threadCount; i++ )
        proofFailCount += jobs[i].failCount;

    if( proofFailCount )
        Log::Line( "Plot has %llu invalid proofs." );
    else
        Log::Line( "Perfect plot! All proofs are valid." );

    return proofFailCount == 0;
}

//-----------------------------------------------------------
uint64 ValidateInMemory( UnpackedK32Plot& plot, ThreadPool& pool )
{
    Log::Line( "Validating plot in-memory" );
    Log::Line( "F7 entry count: %llu", plot.f7.Length() );
    Log::Line( "" );

    std::atomic<uint64> totalFailures = 0;

    AnonMTJob::Run( pool, [&]( AnonMTJob* self ) {

        auto Log = [=]( const char* msg, ... ) {
            va_list args;
            va_start( args, msg );
            TVLog( self->_jobId, msg, args );
            va_end( args );
        };

        uint64 count, offset, end;
        GetThreadOffsets( self, (uint64)plot.f7.Length(), count, offset, end );

        Log( "Validating proofs %10llu...%-10llu", offset, end );
        self->SyncThreads();

        const uint64 reportInterval = kCheckpoint1Interval * 20; // Report every 20 parks

        uint64 fullProofXs[PROOF_X_COUNT];
        uint64 failedCount = 0;

        for( uint64 i = offset; i < end; i++ )
        {
            if( plot.FetchProof( i, fullProofXs ) )
            {
                uint64 outF7 = 0;
                if( ValidateFullProof( plot.plot->K(), plot.plot->PlotId(), fullProofXs, outF7 ) )
                {
                    const uint32 expectedF7 = plot.f7[i];

                    if( expectedF7 != outF7 )
                    {
                        if( failedCount++ == 0 )
                        {
                            Log( "Proof failed: Expected %llu but got %llu @ %llu (park %llu).", 
                                (uint64)expectedF7, outF7, i, i / kCheckpoint1Interval );
                        }
                    }
                }
                else
                {
                    if( failedCount++ == 0 )
                    {
                        Log( "Proof failed: Validation error for f7 %llu @ %llu (park %llu).", 
                             (uint64)plot.f7[i], i, i / kCheckpoint1Interval );
                    }
                }
            }
            else
            {
                if( failedCount++ == 0 )
                {
                    Log( "Proof failed: Fetch failure for f7 %llu @ %llu (park %llu).", 
                            (uint64)plot.f7[i], i, i / kCheckpoint1Interval );
                }
            }

            const uint64 proofsChecked = i - offset;
            if( ( proofsChecked > 0 && proofsChecked % reportInterval == 0 ) || i + 1 == end )
            {
                Log( "Proofs validated: %10llu / %-10llu ( %.2lf %% ) [ %sFailed: %llu%s ]",
                    proofsChecked, count, 
                    (double)proofsChecked / count * 100.0,
                    failedCount > 0 ? COLOR_RED_BOLD : COLOR_GREEN_BOLD, 
                    failedCount,
                    COLOR_NONE );
            }
        }

        uint64 f = totalFailures.load( std::memory_order_acquire );

        while( !totalFailures.compare_exchange_weak( 
            f, f + failedCount,
            std::memory_order_release, 
            std::memory_order_relaxed ) );
    });

    return totalFailures;
}


//-----------------------------------------------------------
void ValidateJob::Log( const char* msg, ... )
{
    va_list args;
    va_start( args, msg );
    TVLog( JobId(), msg, args );
    va_end( args );
}

//-----------------------------------------------------------
void ValidateJob::Run()
{
    PlotReader plot( *plotFile );
    
    const uint32 k = plotFile->K();

    uint64 c3ParkCount = 0;
    uint64 c3ParkEnd   = 0;
    
    const uint32 threadCount     = this->JobCount();
    const uint64 plotC3ParkCount = plotFile->TableSize( PlotTable::C1 ) / sizeof( uint32 ) - 1;
    
    c3ParkCount = plotC3ParkCount / threadCount;

    uint64 startC3Park = this->JobId() * c3ParkCount;

    {
        uint64 trailingParks = plotC3ParkCount - c3ParkCount * threadCount;
        
        if( this->JobId() < trailingParks )
            c3ParkCount ++;

        startC3Park += std::min( trailingParks, (uint64)this->JobId() );
    }

    c3ParkEnd = startC3Park + c3ParkCount;

    if( startOffset > 0.0f )
    {
        startC3Park += std::min( c3ParkCount, (uint64)( c3ParkCount * startOffset ) );
        c3ParkCount = c3ParkEnd - startC3Park;
    }

    Log( "Park range: %10llu..%-10llu  Park count: %llu", startC3Park, c3ParkEnd, c3ParkCount );

    ///
    /// Start validating C3 parks
    ///
    uint64* f7Entries = bbcalloc<uint64>( kCheckpoint1Interval );
    memset( f7Entries, 0, kCheckpoint1Interval * sizeof( uint64 ) );

    uint64* p7Entries = bbcalloc<uint64>( kEntriesPerPark );
    memset( p7Entries, 0, sizeof( kEntriesPerPark ) * sizeof( uint64 ) );


    uint64 curPark7 = 0;
    if( JobId() == 0 )
        FatalIf( !plot.ReadP7Entries( 0, p7Entries ), "Failed to read P7 0." );
    
    uint64 proofFailCount = 0;
    uint64 fullProofXs[PROOF_X_COUNT];

    for( uint64 c3ParkIdx = startC3Park; c3ParkIdx < c3ParkEnd; c3ParkIdx++ )
    {
        const auto timer = TimerBegin();

        const int64 f7EntryCount = plot.ReadC3Park( c3ParkIdx, f7Entries );

        FatalIf( f7EntryCount < 0, "Could not read C3 park %llu.", c3ParkIdx );
        ASSERT( f7EntryCount <= kCheckpoint1Interval );

        const uint64 f7IdxBase = c3ParkIdx * kCheckpoint1Interval;

        for( uint32 e = 0; e < (uint32)f7EntryCount; e++ )
        {
            const uint64 f7Idx       = f7IdxBase + e;
            const uint64 p7ParkIndex = f7Idx / kEntriesPerPark;
            const uint64 f7          = f7Entries[e];

            if( p7ParkIndex != curPark7 )
            {
                // ASSERT( p7ParkIndex == curPark7+1 );
                curPark7 = p7ParkIndex;

                FatalIf( !plot.ReadP7Entries( p7ParkIndex, p7Entries ), "Failed to read P7 %llu.", p7ParkIndex );
            }

            const uint64 p7LocalIdx = f7Idx - p7ParkIndex * kEntriesPerPark;
            const uint64 t6Index    = p7Entries[p7LocalIdx];

            bool success = true;

            if( k <= 32 )
            {
                success = FetchProof<true>( plot, t6Index, fullProofXs );
            }
            else
                success = FetchProof<false>( plot, t6Index, fullProofXs );

            if( success )
            {
                // ReorderProof( plot, fullProofXs );   // <-- No need for this for validation
                
                // Now we can validate the proof
                uint64 outF7;

                if( ValidateFullProof( k, plot.PlotFile().PlotId(), fullProofXs, outF7 ) )
                    success = f7 == outF7;
                else
                    success = false;
            }
            else
            {
                success = false;
                Log( "Park %llu proof fetch failed for f7[%llu] local(%llu) = %llu ( 0x%016llx ) ", 
                   c3ParkIdx, f7Idx, e, f7, f7 );
            }

            if( !success )
            {
                proofFailCount++;
            }
        }

        const double elapsed = TimerEnd( timer );
        Log( "%10llu..%-10llu ( %3.2lf%% ) C3 Park Validated in %2.2lf seconds | Proofs Failed: %llu", 
                c3ParkIdx, c3ParkEnd-1, 
                (double)(c3ParkIdx-startC3Park) / c3ParkCount * 100, elapsed,
                proofFailCount );
    }

    // All done
    this->failCount = proofFailCount;
}

// #TODO: Support K>32
//-----------------------------------------------------------
void GetProofForChallenge( const char* plotPath, const char* challengeHex )
{
    FatalIf( !plotPath || !*plotPath, "Invalid plot path." );
    FatalIf( !challengeHex || !*challengeHex, "Invalid challenge." );

    const size_t lenChallenge = strlen( challengeHex );
    FatalIf( lenChallenge != 64, "Invalid challenge, should be 32 bytes." );

    uint64 challenge[4] = {};
    HexStrToBytes( challengeHex, lenChallenge, (byte*)challenge, 32 );

    FilePlot plot;
    FatalIf( !plot.Open( plotPath ), "Failed to open plot at %s.", plotPath );
    FatalIf( plot.K() != 32, "Only k32 plots are supported." );

    // Read F7 value
    CPBitReader f7Reader( (byte*)challenge, sizeof( challenge ) * 8 );
    const uint32 f7 = (uint32)f7Reader.Read64( 32 );

    // Find this f7 in the plot file
    PlotReader reader( plot );

    uint64 _indices[64] = {};
    Span<uint64> indices( _indices, sizeof( _indices ) / sizeof( uint64 ) );    // #TODO: Should simply return the start index and count

    auto matches = reader.GetP7IndicesForF7( f7, indices );
    FatalIf( matches.Length() == 0, "Could not find f7 %llu in plot", f7 );

    uint64 fullProofXs[PROOF_X_COUNT];
    uint64 proof   [32]  = {};
    char   proofStr[513] = {};
    uint64 p7Entries[kEntriesPerPark] = {};

    int64 prevP7Park = -1;

    for( uint64 i = 0; i < matches.Length(); i++ )
    {
        const uint64 p7Index = matches[i];
        const uint64 p7Park  = p7Index / kEntriesPerPark;
        
        // uint64 o = reader.GetFullProofForF7Index( matches[i], proof );
        if( (int64)p7Park != prevP7Park )
        {
            FatalIf( !reader.ReadP7Entries( p7Park, p7Entries ), "Failed to read P7 %llu.", p7Park );
        }

        prevP7Park = (int64)p7Park;

        const uint64 localP7Index = p7Index - p7Park * kEntriesPerPark;
        const uint64 t6Index      = p7Entries[localP7Index];

        const bool gotProof = FetchProof<true>( reader, t6Index, fullProofXs );
        
        if( gotProof )
        {
            ReorderProof( reader, fullProofXs );

            BitWriter writer( proof, sizeof( proof ) * 8 );

            for( uint32 j = 0; j < PROOF_X_COUNT; j++ )
                writer.Write64BE( fullProofXs[j], 32 );

            for( uint32 j = 0; j < PROOF_X_COUNT/2; j++ )
                proof[j] = Swap64( proof[j] );

            size_t encoded;
            BytesToHexStr( (byte*)proof, sizeof( proof ), proofStr, sizeof( proofStr ), encoded );
            // Log::Line( "[%llu] : %s", i, proofStr );
            Log::Line( proofStr );
        }
    }
}

//-----------------------------------------------------------
UnpackedK32Plot UnpackedK32Plot::Load( IPlotFile** plotFile, ThreadPool& pool, uint32 threadCount )
{
    ASSERT( plotFile );
    const uint32 k = plotFile[0]->K();
    FatalIf( k != 32, "Only k=32 plots are supported for unpacked validation." );

    threadCount = threadCount == 0 ? pool.ThreadCount() : threadCount;

    UnpackedK32Plot plot;

    PlotReader* readers = bbcalloc<PlotReader>( threadCount );
    for( uint32 i = 0; i < threadCount; i++ )
        new ( (void*)&readers[i] ) PlotReader( *plotFile[i] );


    PlotReader& plotReader = readers[0];

    // uint64 c1EntryCount = 0;
    // FatalIf( !plotReader.GetActualC1EntryCount( c1EntryCount ), "Failed to obtain C1 entry count." );
    
    uint64 f7Count = plotReader.GetMaxF7EntryCount(); FatalIf( f7Count < 1, "No F7s found." );

    // Load F7s
    {
        Log::Line( "Unpacking f7 values..." );
        uint32* f7 = bbcvirtallocboundednuma<uint32>( f7Count );

        std::atomic<uint64> sharedF7Count = 0;

        AnonMTJob::Run( pool, threadCount, [&]( AnonMTJob* self ) {

            PlotReader& reader = readers[self->_jobId];

            const uint64 plotParkCount = reader.GetC3ParkCount();

            uint64 parkCount, parkOffset, parkEnd;
            GetThreadOffsets( self, plotParkCount, parkCount, parkOffset, parkEnd );

            uint64  f7Buffer[kCheckpoint1Interval];

            uint32* f7Start  = f7 + parkOffset * kCheckpoint1Interval;
            uint32* f7Writer = f7Start;

            for( uint64 i = parkOffset; i < parkEnd; i++ )
            {
                const int64 entryCount = reader.ReadC3Park( i, f7Buffer );

                FatalIf( entryCount == 0, "Empty C3 park @ %llu.", i );

                if( entryCount > 0 )
                {
                    for( int64 e = 0; e < entryCount; e++ )
                        f7Writer[e] = (uint32)f7Buffer[e];
                    
                    f7Writer += entryCount;
                }

                if( entryCount < kCheckpoint1Interval )
                {
                    if( self->IsLastThread() )
                    {
                        // Short-circuit as soon as we find a partial park in the last thread
                        break;
                    }
                    else
                        Fatal( "[%u/%u] C3 park %llu is not full and it is not the last park.", self->_jobId, self->_jobCount, i );
                }
            }

            sharedF7Count += (uint64)(uintptr_t)(f7Writer - f7Start);
        });

        f7Count        = sharedF7Count;
        plot.f7.length = f7Count;
        plot.f7.values = f7;

        Log::Line( "Actual C3 Parks : %llu", CDiv( f7Count, kCheckpoint1Interval ) );
    }
    
    // Read Park 7
    Log::Line( "Reding park 7..." );
    uint64* p7Indices = nullptr;
    {
        const uint64 park7Count = CDiv( f7Count, kEntriesPerPark );

        #if _DEBUG
            const size_t p7Size             = plotReader.PlotFile().TableSize( PlotTable::Table7 );
            const size_t parkSize           = CalculatePark7Size( plotReader.PlotFile().K() );
            const size_t potentialParkCount = p7Size / parkSize;
            ASSERT( potentialParkCount >= park7Count );
        #endif
     
        p7Indices = bbcvirtallocboundednuma<uint64>( park7Count * kEntriesPerPark );

        AnonMTJob::Run( pool, threadCount, [=]( AnonMTJob* self ) {
            
            PlotReader& reader = readers[self->_jobId];

            uint64 parkCount, parkOffset, parkEnd;
            GetThreadOffsets( self, park7Count, parkCount, parkOffset, parkEnd );

            uint64* p7Writer = p7Indices + parkOffset * kEntriesPerPark;

            for( uint64 i = parkOffset; i < parkEnd; i++ )
            {
                FatalIf( !reader.ReadP7Entries( i, p7Writer ), "Failed to read park 7 %llu.", i );
                p7Writer += kEntriesPerPark;
            }
        });
    }

    
    auto LoadBackPtrTable = [&]( const TableId table ) {

        Log::Line( "Loading table %u", table+1 );
    
        const uint64 plotParkCount = plotReader.GetTableParkCount( (PlotTable)table );

        Span<Pair> backPointers = bbcvirtallocboundednuma_span<Pair>( plotParkCount * kEntriesPerPark );
        uint64     missingParks   = 0;
        uint64     missingEntries = 0;


        AnonMTJob::Run( pool, threadCount, [&]( AnonMTJob* self ) {
        
            PlotReader& reader = readers[self->_jobId];
            
            uint64 parkCount, parkOffset, parkEnd;
            GetThreadOffsets( self, plotParkCount, parkCount, parkOffset, parkEnd );
            
            uint64 parkEntryCount;
            uint128 linePoints[kEntriesPerPark];

            Span<Pair> tableWriter = backPointers.Slice( parkOffset * kEntriesPerPark, parkCount * kEntriesPerPark );

            for( uint64 i = parkOffset; i < parkEnd; i++ )
            {
                if( !reader.ReadLPPark( table, i, linePoints, parkEntryCount ) )
                {
                    // If its the last thread loading the park, there may be empty space
                    // after the actual parks end, so these are allowed, but we stop processing parks after this.
                    if( self->IsLastThread() )
                    {
                        missingParks = parkEnd - i;
                        break;
                    }
                    else
                        FatalErrorMsg( "Failed to read Table %u park %llu", table+1, i );
                }

                // Since we only support in-ram validation for k <= 32, we can do 64-bit LP reading
                for( uint64 e = 0; e < parkEntryCount; e++ )
                {
                    const BackPtr bp = LinePointToSquare64( (uint64)linePoints[e] );
                    Pair pair;
                    pair.left  = (uint32)bp.x;
                    pair.right = (uint32)bp.y;
                    tableWriter[e] = pair;
                }

                if( parkEntryCount < kEntriesPerPark )
                {
                    // We only allow incomplete parks at the end, so stop processing parks as soon as we encounter one
                    if( self->IsLastThread() && i + 1 == parkEnd )
                    {
                        missingEntries = kEntriesPerPark - parkEntryCount;
                        break;
                    }
                    else
                        FatalErrorMsg( "Encountered a non-full park for table %u at index %llu. These are unsupported", table+1, i );
                }

                tableWriter = tableWriter.Slice( kEntriesPerPark );
            }
        });
            
        const uint64 tableEntryCount = ( plotParkCount * kEntriesPerPark ) - ( missingParks * kEntriesPerPark + missingEntries ); 
        return backPointers.Slice( 0, tableEntryCount );
    };
    
    /// Load T6 first so we can sort them on P7
    Log::Line( "Loading back pointer tables..." );
    auto t6 = LoadBackPtrTable( TableId::Table6 );
    ASSERT( t6.Length() == f7Count );
    {
        // Span<Pair> sortedT6 = bbcvirtallocboundednuma_span<Pair>( t6.Length() );

        AnonMTJob::Run( pool, threadCount, [=]( AnonMTJob* self ) {
                    
            uint64 count, offset, end;
            GetThreadOffsets( self, f7Count, count, offset, end );

            static_assert( sizeof( uint64 ) == sizeof( Pair ) );
            const Span<uint64> reader( p7Indices + offset, count );
                  Span<Pair>   writer( (Pair*)p7Indices + offset, count );

            for( uint64 i = 0; i < count; i++ )
                writer[i] = t6[reader[i]];
        });
        
        plot.tables[(int)TableId::Table6] = Span<Pair>( (Pair*)p7Indices, f7Count );
        bbvirtfreebounded( t6.values );
    }

    // Read the rest of the entries
    plot.tables[(int)TableId::Table5] = LoadBackPtrTable( TableId::Table5 );
    plot.tables[(int)TableId::Table4] = LoadBackPtrTable( TableId::Table4 );
    plot.tables[(int)TableId::Table3] = LoadBackPtrTable( TableId::Table3 );
    plot.tables[(int)TableId::Table2] = LoadBackPtrTable( TableId::Table2 );
    plot.tables[(int)TableId::Table1] = LoadBackPtrTable( TableId::Table1 );
    
    Log::Line( "Decompressed plot into memory." );
    return plot;
}

//-----------------------------------------------------------
bool UnpackedK32Plot::FetchProof( const uint64 index, uint64 fullProofXs[PROOF_X_COUNT] )
{
    if( index >= this->f7.Length() )
        return false;

    uint64 lpIndices[2][PROOF_X_COUNT];

    uint64* lpIdxSrc = lpIndices[0];
    uint64* lpIdxDst = lpIndices[1];

    *lpIdxSrc = index;

    uint32 lookupCount = 1;
    for( TableId table = TableId::Table6; table >= TableId::Table1; table-- )
    {
        ASSERT( lookupCount <= 32 );

        const Span<Pair> plotTable = this->Table( table );

        for( uint32 i = 0, dst = 0; i < lookupCount; i++, dst += 2 )
        {
            const uint64 idx = lpIdxSrc[i];

            if( idx >= plotTable.Length() )
                return false;

            const Pair ptr = plotTable[idx];
            lpIdxDst[dst+0] = ptr.right;
            lpIdxDst[dst+1] = ptr.left;
        }

        lookupCount <<= 1;
        std::swap( lpIdxSrc, lpIdxDst );
    }

    // Full proof x's will be at the src ptr
    memcpy( fullProofXs, lpIdxSrc, sizeof( uint64 ) * PROOF_X_COUNT );
    return true;
}

//-----------------------------------------------------------
template<bool Use64BitLpToSquare>
bool FetchProof( PlotReader& plot, uint64 t6LPIndex, uint64 fullProofXs[PROOF_X_COUNT] )
{
    uint64 lpIndices[2][PROOF_X_COUNT];
    // memset( lpIndices, 0, sizeof( lpIndices ) );

    uint64* lpIdxSrc = lpIndices[0];
    uint64* lpIdxDst = lpIndices[1];

    *lpIdxSrc = t6LPIndex;

    // Fetch line points to back pointers going through all our tables
    // from 6 to 1, grabbing all of the x's that make up a proof.
    uint32 lookupCount = 1;

    for( TableId table = TableId::Table6; table >= TableId::Table1; table-- )
    {
        ASSERT( lookupCount <= 32 );

        for( uint32 i = 0, dst = 0; i < lookupCount; i++, dst += 2 )
        {
            const uint64 idx = lpIdxSrc[i];

            uint128 lp = 0;
            if( !plot.ReadLP( table, idx, lp ) )
                return false;

            BackPtr ptr;
            if constexpr ( Use64BitLpToSquare )
                ptr = LinePointToSquare64( (uint64)lp );
            else
                ptr = LinePointToSquare( lp );

            lpIdxDst[dst+0] = ptr.y;
            lpIdxDst[dst+1] = ptr.x;
        }

        lookupCount <<= 1;

        std::swap( lpIdxSrc, lpIdxDst );
        // memset( lpIdxDst, 0, sizeof( uint64 ) * PROOF_X_COUNT );
    }

    // Full proof x's will be at the src ptr
    memcpy( fullProofXs, lpIdxSrc, sizeof( uint64 ) * PROOF_X_COUNT );
    return true;
}

//-----------------------------------------------------------
bool ValidateFullProof( const uint32 k, const byte plotId[BB_PLOT_ID_LEN], uint64 fullProofXs[PROOF_X_COUNT], uint64& outF7 )
{
    uint64   fx  [PROOF_X_COUNT];
    MetaBits meta[PROOF_X_COUNT];

    // Convert these x's to f1 values
    {
        const uint32 xShift = k - kExtraBits;
        
        // Prepare ChaCha key
        byte key[32] = { 1 };
        memcpy( key + 1, plotId, 31 );

        chacha8_ctx chacha;
        chacha8_keysetup( &chacha, key, 256, NULL );

        // Enough to hold 2 cha-cha blocks since a value my span over 2 blocks
        byte blocks[kF1BlockSize*2];

        for( uint32 i = 0; i < PROOF_X_COUNT; i++ )
        {
            const uint64 x        = fullProofXs[i];
            const uint64 blockIdx = x * k / kF1BlockSizeBits; 

            chacha8_get_keystream( &chacha, blockIdx, 2, blocks );

            // Get the starting and end locations of y in bits relative to our block
            const uint64 bitStart = x * k - blockIdx * kF1BlockSizeBits;

            CPBitReader hashBits( blocks, sizeof( blocks ) * 8 );
            hashBits.Seek( bitStart );

            // uint64 y = SliceUInt64FromBits( blocks, bitStart, k ); // #TODO: Figure out what's wrong with this method.
            uint64 y = hashBits.Read64( k );
            y = ( y << kExtraBits ) | ( x >> xShift );

            fx  [i] = y;
            meta[i] = MetaBits( x, k );
        }
    }

    // Forward propagate f1 values to get the final f7
    uint32 iterCount = PROOF_X_COUNT;
    for( TableId table = TableId::Table2; table <= TableId::Table7; table++, iterCount >>= 1)
    {
        for( uint32 i = 0, dst = 0; i < iterCount; i+= 2, dst++ )
        {
            uint64 y0 = fx[i+0];
            uint64 y1 = fx[i+1];

            const MetaBits* lMeta = &meta[i+0];
            const MetaBits* rMeta = &meta[i+1];

            if( y0 > y1 ) 
            {
                std::swap( y0, y1 );
                std::swap( lMeta, rMeta );
            }

            // Must be on the same group
            if( !FxMatch( y0, y1 ) )
                return false;

            // FxGen
            uint64 outY;
            MetaBits outMeta;
            FxGen( table, k, y0, *lMeta, *rMeta, outY, outMeta );

            fx  [dst] = outY;
            meta[dst] = outMeta;
        }
    }

    outF7 = fx[0] >> kExtraBits;

    return true;
}


// #TODO: Avoid code duplication here? At least for f1
//-----------------------------------------------------------
void ReorderProof( PlotReader& plot, uint64 fullProofXs[PROOF_X_COUNT] )
{
    const uint32 k = plot.PlotFile().K();

    uint64   fx  [PROOF_X_COUNT];
    MetaBits meta[PROOF_X_COUNT];

    uint64  xtmp[PROOF_X_COUNT];
    uint64* xs = fullProofXs;

    // Convert these x's to f1 values
    GetProofF1( k, plot.PlotFile().PlotId(), fullProofXs, fx );
    for( uint32 i = 0; i < PROOF_X_COUNT; i++ )
        meta[i] = MetaBits( xs[i], k );

    // Forward propagate f1 values to get the final f7
    uint32 iterCount = PROOF_X_COUNT;
    for( TableId table = TableId::Table2; table <= TableId::Table7; table++, iterCount >>= 1)
    {
        for( uint32 i = 0, dst = 0; i < iterCount; i+= 2, dst++ )
        {
            uint64 y0 = fx[i+0];
            uint64 y1 = fx[i+1];

            const MetaBits* lMeta = &meta[i+0];
            const MetaBits* rMeta = &meta[i+1];

            if( y0 > y1 ) 
            {
                std::swap( y0, y1 );
                std::swap( lMeta, rMeta );

                // Swap X's so far that have generated this y
                const uint32 count = 1u << ((int)table-1);
                uint64* x = xs + i * count;
                bbmemcpy_t( xtmp   , x      , count );
                bbmemcpy_t( x      , x+count, count );
                bbmemcpy_t( x+count, xtmp   , count );
            }

            // FxGen
            uint64 outY;
            MetaBits outMeta;
            FxGen( table, k, y0, *lMeta, *rMeta, outY, outMeta );

            fx  [dst] = outY;
            meta[dst] = outMeta;
        }
    }
}

//-----------------------------------------------------------
void GetProofF1( uint32 k, const byte plotId[BB_PLOT_ID_LEN], uint64 fullProofXs[PROOF_X_COUNT], uint64 fx[PROOF_X_COUNT] )
{
    const uint32 xShift = k - kExtraBits;
        
    // Prepare ChaCha key
    byte key[32] = { 1 };
    memcpy( key + 1, plotId, 31 );

    chacha8_ctx chacha;
    chacha8_keysetup( &chacha, key, 256, NULL );

    // Enough to hold 2 cha-cha blocks since a value my span over 2 blocks
    byte blocks[kF1BlockSize*2];

    for( uint32 i = 0; i < PROOF_X_COUNT; i++ )
    {
        const uint64 x        = fullProofXs[i];
        const uint64 blockIdx = x * k / kF1BlockSizeBits; 

        chacha8_get_keystream( &chacha, blockIdx, 2, blocks );

        // Get the starting and end locations of y in bits relative to our block
        const uint64 bitStart = x * k - blockIdx * kF1BlockSizeBits;

        CPBitReader hashBits( blocks, sizeof( blocks ) * 8 );
        hashBits.Seek( bitStart );

        // uint64 y = SliceUInt64FromBits( blocks, bitStart, k ); // #TODO: Figure out what's wrong with this method.
        uint64 y = hashBits.Read64( k );
        y = ( y << kExtraBits ) | ( x >> xShift );

        fx[i] = y;
    }
}

//-----------------------------------------------------------
bool FxMatch( uint64 yL, uint64 yR )
{
    const uint64 groupL = yL / kBC;
    const uint64 groupR = yR / kBC;

    if( groupR - groupL != 1 )
        return false;

    // Groups are adjacent, check if the y values actually match
    const uint16 parity = groupL & 1;

    const uint64 groupLRangeStart = groupL * kBC;
    const uint64 groupRRangeStart = groupR * kBC;

    const uint64 localLY = yL - groupLRangeStart;
    const uint64 localRY = yR - groupRRangeStart;
    
    for( int iK = 0; iK < kExtraBitsPow; iK++ )
    {
        const uint64 targetR = L_targets[parity][localLY][iK];
        
        if( targetR == localRY )
            return true;
    } 

    return false;
}

//-----------------------------------------------------------
void FxGen( const TableId table, const uint32 k, 
            const uint64 y, const MetaBits& metaL, const MetaBits& metaR,
            uint64& outY, MetaBits& outMeta )
{
    FxBits input( y, k + kExtraBits );

    if( table < TableId::Table4 )
    {
        outMeta = metaL + metaR;
        input += outMeta;
    }
    else
    {
        input += metaL;
        input += metaR;
    }

    byte inputBytes[64];
    byte hashBytes [32];

    input.ToBytes( inputBytes );

    blake3_hasher hasher;
    blake3_hasher_init    ( &hasher );
    blake3_hasher_update  ( &hasher, inputBytes, input.LengthBytes() );
    blake3_hasher_finalize( &hasher, hashBytes, sizeof( hashBytes ) );

    outY = BytesToUInt64( hashBytes ) >> ( 64 - (k + kExtraBits) );

    if( table >= TableId::Table4 && table < TableId::Table7 )
    {
        size_t multiplier = 0;
        switch( table )
        {
            case TableId::Table4: multiplier = TableMetaOut<TableId::Table4>::Multiplier; break;
            case TableId::Table5: multiplier = TableMetaOut<TableId::Table5>::Multiplier; break;
            case TableId::Table6: multiplier = TableMetaOut<TableId::Table6>::Multiplier; break;
            default: 
                ASSERT( 0 );
                break;
        }

        const uint32 metaBits  = (uint32)( k * multiplier );
        const uint32 yBits     = k + kExtraBits;
        const uint32 startByte = yBits / 8 ;
        const uint32 startBit  = yBits - startByte * 8;

        outMeta = MetaBits( hashBytes + startByte, metaBits, startBit );
    }
}

//-----------------------------------------------------------
/// Convertes 8 bytes to uint64 and endian-swaps it.
/// This takes any byte alignment, so that bytes does
/// not have to be aligned to 64-bit boundary.
/// This is for compatibility for how chiapos extracts
/// bytes into integers.
//-----------------------------------------------------------
inline uint64 BytesToUInt64( const byte bytes[8] )
{
    uint64 tmp;
    memcpy( &tmp, bytes, sizeof( uint64 ) );
    return Swap64( tmp );
}

//-----------------------------------------------------------
// Treats bytes as a set of 64-bit big-endian fields,
// from which it will extract a whole 64-bit value
// at the given bit offset. 
// The result may be truncated if the requested number of 
// bits + the number of bits overflows the 64-bit field.
// That is, if the local bit offset in the target bit field
// + the bitCount is greater than 64.
// This function is for compatibility with the way chiapos
// slices bits off of binary byte blobs.
//-----------------------------------------------------------
inline uint64 SliceUInt64FromBits( const byte* bytes, uint32 bitOffset, uint32 bitCount )
{
    ASSERT( bitCount <= 64 );
     
    // #TODO: This is wrong, it's not treating the bytes as 64-bit fields.
    //        So that we may have swapped at the wrong position.
    //        In fact we might have fields that span 2 64-bit values.
    //        So we need to split it into 2, and do 2 swaps.
    const uint64 startByte = bitOffset / 8;
    bytes += startByte;

    // Convert bit offset to be local to the uint64 field
    bitOffset -= ( bitOffset >> 6 ) * 64; // bitOffset >> 6 == bitOffset / 64

    uint64 field = BytesToUInt64( bytes );
    
    field <<= bitOffset;     // Start bits from the MSBits
    field >>= 64 - bitCount; // Take the MSbits

    return field;
}


//-----------------------------------------------------------
void TVLog( const uint32 id, const char* msg, va_list args )
{
    _logLock.lock();
    fprintf( stdout, "[%3u] ", id );
    vfprintf( stdout, msg, args );
    putc( '\n', stdout );
    _logLock.unlock();
}

//-----------------------------------------------------------
// void TLog( const uint32 id, const char* msg, ... )
// {
//     va_list args;
//     va_start( args, msg );
//     TVLog( id, msg, args );
//     va_end( args );
// }

#pragma GCC diagnostic pop

