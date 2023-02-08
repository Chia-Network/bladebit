#include "TestUtil.h"
#include "harvesting/GreenReaper.h"
#include "tools/PlotReader.h"
#include "plotmem/LPGen.h"

// #TODO: Move these to the GreenReaper api
#include "util/BitView.h"

bool GetProofForChallenge( PlotReader& reader, const char* challengeHex, uint64 fullProofXs[GR_POST_PROOF_X_COUNT] );
bool GetProofForChallenge( PlotReader& reader, const uint32 f7, uint64 fullProofXs[GR_POST_PROOF_X_COUNT] );

//-----------------------------------------------------------
TEST_CASE( "compressed-plot-proof", "[sandbox][plots]" )
{
    FilePlot filePlot;
    ENSURE( filePlot.Open( "/home/harold/plot/ref/plot-k32-2022-10-04-22-13-c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835.plot" ) );
    ENSURE( filePlot.K() == 32 );

    PlotReader reader( filePlot );

    const uint32 f7 = 7;
    
    uint64 fullProofXs    [GR_POST_PROOF_X_COUNT];
    uint32 compressedProof[GR_POST_PROOF_CMP_X_COUNT] = {};
    
    ENSURE( GetProofForChallenge( reader, f7, fullProofXs ) );

    const uint32 entryBitCount = bbclamp<uint32>( GetEnvU32( "bb_entry_bits", 16 ), 10, 16 );
    const uint32 shift         = 32 - entryBitCount;

    GreenReaperConfig cfg = {};
    cfg.threadCount = bbclamp<uint32>( GetEnvU32( "bb_thread_count", 1 ), 1, SysHost::GetLogicalCPUCount() );

    // Compress the full proof x's to n bits
    GRCompressedProofRequest req = {};
    
    Log::WriteLine( "Proof X's:" );

    for( uint32 i = 0; i < GR_POST_PROOF_X_COUNT; i+=2 )
    {
        uint64 x1 = fullProofXs[i];
        uint64 x2 = fullProofXs[i+1];

        Log::WriteLine( "[%2u]: %-10u ( 0x%08x )", i  , (uint32)x1, (uint32)x1 );
        Log::WriteLine( "[%2u]: %-10u ( 0x%08x )", i+1, (uint32)x2, (uint32)x2 );
        
        x1 >>= shift;
        x2 >>= shift;
        ENSURE( x1 < (1ull << entryBitCount) );
        ENSURE( x2 < (1ull << entryBitCount) );

        // Convert to 31-bit linepoint
        const uint32 x12 = (uint32)SquareToLinePoint( x2, x1 );
        compressedProof[i>>1] = x12;
    }

    GreenReaperContext* cx = nullptr;
    cx = grCreateContext( &cfg );
    ENSURE( cx );

    req.compressionLevel = 17 - entryBitCount;
    // req.entryBitCount = entryBitCount;
    // req.expectedF7    = f7;
    // req.plotId        = filePlot.PlotId();

    // Warm start
    memcpy( req.compressedProof, compressedProof, sizeof( compressedProof ) );
    grFetchProofForChallenge( cx, &req );
    memcpy( req.compressedProof, compressedProof, sizeof( compressedProof ) );
    
    const auto timer = TimerBegin();
    auto result = grFetchProofForChallenge( cx, &req );
    const double elapsed = TimerEnd( timer );

    // const bool isSame = memcmp( req.fullProof, fullProofXs, sizeof( fullProofXs ) ) == 0;
    // #TODO: Need it place it in proof ordering for this to work.
    // for( uint32 i = 0; i < GR_POST_PROOF_X_COUNT; i++ )
    // {
    //     ASSERT( req.fullProof[i] == fullProofXs[i] );
    // }

    Log::Line( "Completed %u-bit proof fetch %s in %.2lf seconds using %u thread(s).",
        entryBitCount,
        result == GRProofResult_OK ? "successfully" : "with failure",
        elapsed, cfg.threadCount );

    grDestroyContext( cx );
    cx = nullptr;
}


//-----------------------------------------------------------
bool GetProofForChallenge( PlotReader& reader, const char* challengeHex, uint64 fullProofXs[GR_POST_PROOF_X_COUNT] )
{
    FatalIf( !challengeHex || !*challengeHex, "Invalid challenge." );

    const size_t lenChallenge = strlen( challengeHex );
    FatalIf( lenChallenge != 64, "Invalid challenge, should be 32 bytes." );

    // #TODO: Use HexStrToBytesSafe
    uint64 challenge[4] = {};
    HexStrToBytes( challengeHex, lenChallenge, (byte*)challenge, 32 );
    
    CPBitReader f7Reader( (byte*)challenge, sizeof( challenge ) * 8 );
    const uint64 f7 = f7Reader.Read64( 32 );

    return GetProofForChallenge( reader, (uint32)f7, fullProofXs );
}

//-----------------------------------------------------------
bool GetProofForChallenge( PlotReader& reader, const uint32 f7, uint64 fullProofXs[GR_POST_PROOF_X_COUNT] )
{
    uint64 _indices[64] = {};
    Span<uint64> indices( _indices, sizeof( _indices ) / sizeof( uint64 ) );

    auto matches = reader.GetP7IndicesForF7( f7, indices );

    // uint64 fullProofXs[GR_POST_PROOF_X_COUNT];
    // uint64 proof   [32]  = {};
    // char   proofStr[513] = {};
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

        // if( compressed )
        //     gotProof = FetchC16Proof( reader, t6Index, fullProofXs );
        // else
        //     gotProof = FetchProof<true>( reader, t6Index, fullProofXs );
        
        if( reader.FetchProof( t6Index, fullProofXs ) )
        {
            // #TODO: reorder it
            return true;


            // ReorderProof( reader, fullProofXs );

            // BitWriter writer( proof, sizeof( proof ) * 8 );

            // for( uint32 j = 0; j < PROOF_X_COUNT; j++ )
            //     writer.Write64BE( fullProofXs[j], 32 );

            // for( uint32 j = 0; j < PROOF_X_COUNT/2; j++ )
            //     proof[j] = Swap64( proof[j] );

            // size_t encoded;
            // BytesToHexStr( (byte*)proof, sizeof( proof ), proofStr, sizeof( proofStr ), encoded );
            // Log::Line( "[%llu] : %s", i, proofStr );
            // Log::Line( proofStr );
        }

        return false;
    }

    return false;
}

