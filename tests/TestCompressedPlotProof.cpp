#include "TestUtil.h"
#include "harvesting/GreenReaper.h"
#include "tools/PlotReader.h"
#include "plotmem/LPGen.h"
#include "BLS.h"

// #TODO: Move these to the GreenReaper api
#include "util/BitView.h"

bool GetProofForChallenge( PlotReader& reader, const char* challengeHex, uint64 fullProofXs[GR_POST_PROOF_X_COUNT] );
bool GetProofForChallenge( PlotReader& reader, const uint32 f7, uint64 fullProofXs[GR_POST_PROOF_X_COUNT] );
static void Sha256( byte outHash[32], const byte* bytes, size_t length );

//-----------------------------------------------------------
TEST_CASE( "compressed-plot-proof", "[sandbox][plots]" )
{
    FilePlot filePlot;
    ENSURE( filePlot.Open( "/home/harold/plot/ref/plot-k32-2022-10-04-22-13-c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835.plot" ) );
    ENSURE( filePlot.K() == 32 );

    PlotReader reader( filePlot );

    const uint32 f7 = 7;
    
    uint64 fullProofXs    [GR_POST_PROOF_X_COUNT];
    uint64 compressedProof[GR_POST_PROOF_CMP_X_COUNT] = {};
    
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
        result == GRResult_OK ? "successfully" : "with failure",
        elapsed, cfg.threadCount );

    grDestroyContext( cx );
    cx = nullptr;
}

//-----------------------------------------------------------
TEST_CASE( "compressed-plot-qualities", "[sandbox][plots]" )
{
    // const char*  plotPath         = GetEnv( "bb_plot_path", "/home/harold/plot/tmp/plot-k32-c01-2023-02-13-22-21-c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835.plot" );
    // const char* plotPath = "/home/harold/plot/tmp/plot-k32-2023-02-09-21-15-c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835.plot";

    // const char* plotPath = "/home/harold/plot/tmp/plot-k32-c01-2023-02-13-22-21-c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835.plot";
    // const char* plotPath = "/home/harold/plot/tmp/plot-k32-c02-2023-02-14-21-19-c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835.plot";
    // const char* plotPath = "/home/harold/plot/tmp/plot-k32-c03-2023-02-14-21-31-c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835.plot";
    // const char* plotPath = "/home/harold/plot/tmp/plot-k32-c04-2023-02-08-01-33-c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835.plot";
    // const char* plotPath = "/home/harold/plot/tmp/plot-k32-c05-2023-02-14-21-35-c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835.plot";
    // const char* plotPath = "/home/harold/plot/tmp/plot-k32-c06-2023-02-14-21-43-c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835.plot";
    const char* plotPath = "/home/harold/plot/tmp/plot-k32-c07-2023-02-08-17-35-c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835.plot";
    
    // const char* plotPath = "/home/harold/plot/tmp/plot-k32-c09-2023-02-14-21-22-c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835.plot";
                 plotPath    = GetEnv( "bb_plot", plotPath );
    const uint32 iterations  = GetEnvU32( "bb_iterations", 1 );
    const uint32 f7Start     = GetEnvU32( "bb_f7", 7 );
    const uint32 threadCount = bbclamp<uint32>( GetEnvU32( "bb_thread_count", 16 ), 1, SysHost::GetLogicalCPUCount() );

    Log::Line( "Iterations  : %u", iterations );
    Log::Line( "Starting F7 : %u", f7Start );
    Log::Line( "Threads     : %u", threadCount );

    FilePlot filePlot;
    ENSURE( filePlot.Open( plotPath ) );
    ENSURE( filePlot.K() == 32 );
    ENSURE( filePlot.CompressionLevel() > 0 );

    const uint32 k = filePlot.K();

    // Read F7 from challenge
    uint64 f7 = f7Start;
    byte challenge[32] = {};
    {
        const char* challengeStr = "00000000ff04b8ee9355068689bd558eafe07cc7af47ad1574b074fc34d6913a";
        HexStrToBytes( challengeStr, sizeof( challenge )*2, challenge, sizeof( challenge ) );

        // CPBitReader f7Reader( (byte*)challenge, sizeof( challenge ) * 8 );
        // f7 = f7Reader.Read64( filePlot.K() );
    }

    // Init GR context & qualities tree
    GreenReaperContext* gr = nullptr;
    {
        GreenReaperConfig cfg = {};
        cfg.threadCount = threadCount;

        gr = grCreateContext( &cfg );
        ENSURE( gr );
    }

    const uint32 last5Bits = (uint32)challenge[31] & 0x1f;

    PlotReader reader( filePlot );

    uint64 _indices[64] = {};
    Span<uint64> indices( _indices, sizeof( _indices ) / sizeof( uint64 ) );

    uint64 p7Entries[kEntriesPerPark] = {};

    uint64 nFound = 0;

    for( uint32 c = 0; c < iterations; c++ )
    {
        uint64 p7IndexBase = 0;
        const uint64 matchCount  = reader.GetP7IndicesForF7( f7, p7IndexBase );

        Log::Line( "[%-2u] F7: %llu", c, f7 );
        

            // FatalIf( matches.Length() < 1, "F7 has no proofs." );
        if( matchCount < 1 )
        {
            // Log::Line( " No proofs." );
            f7++;
            continue;
        }

        nFound += matchCount;

        // Embed f7 into challenge as BE
        const size_t f7Size = CDiv( k, 8 );
        for( size_t i = 0; i < f7Size; i++ )
            challenge[i] = (byte)(f7 >> ((f7Size - i - 1) * 8));

        f7++;

        const bool needBothLeaves = filePlot.CompressionLevel() >= 6 ;

        for( uint64 i = 0; i < matchCount; i++ )
        {
            const uint64 p7Index = p7IndexBase + i;
            const uint64 p7Park  = p7Index / kEntriesPerPark;
            FatalIf( !reader.ReadP7Entries( p7Park, p7Entries ), "Failed to read P7 %llu.", p7Park );   

            const uint64 localP7Index = p7Index - p7Park * kEntriesPerPark;
                  uint64 lpIndex      = p7Entries[localP7Index];
                  uint64 altIndex     = 0;

            // Init root node and set as current node
            // GRQualitiesTreeInit( &qTree );
            // auto& root = *GRQualitiesTreeAddNode( &qTree, nullptr );
            // GRQualitiesNode* node = &root;

            // Go from Table 6 to the final table to following only 1 path, and generating the graph
            const TableId finalTable = TableId::Table2; // #TODO: Support other tables

            for( TableId table = TableId::Table6; table > finalTable; table-- )
            {
                // Read line point
                uint128 lp = 0;
                FatalIf( !reader.ReadLP( table, lpIndex, lp ), "Failed to read line point for table %u", table+1 );
                const BackPtr ptr = LinePointToSquare( lp );

                ASSERT( ptr.x >= ptr.y );

                const bool isTableBitSet = ((last5Bits >> ((uint32)table-1)) & 1) == 1;

                if( !isTableBitSet )
                {
                    lpIndex = ptr.y;
                    // node->nextIndex = ptr.x;    // Store the alternate path
                    altIndex = ptr.x;
                }
                else
                {
                    lpIndex = ptr.x;
                    // node->nextIndex = ptr.y;    // Store the alternate path
                    altIndex = ptr.y;
                }

                if( table -1 == finalTable )
                    break;

                // GRQualitiesNode* child = GRQualitiesTreeAddNode( &qTree, node );
                // ENSURE( child );
                // node = child;
            }

            // Read compressed table line point (contains partial x's)
            {
                // Read both back pointers, depending on compression level
                uint128 xLP0, xLP1;
                FatalIf( !reader.ReadLP( finalTable, lpIndex, xLP0 ), "Failed to read line point from compressed table" );

                if( needBothLeaves )
                    FatalIf( !reader.ReadLP( finalTable, altIndex, xLP1 ), "Failed to read line point from compressed table" );
                // const BackPtr xs1 = LinePointToSquare( xLP );

                // FatalIf( !reader.ReadLP( finalTable, node->nextIndex, xLP ), "Failed to read line point from compressed table" );
                // const BackPtr xs2 = LinePointToSquare( xLP );

                // Unsupported proof (should be dropped)
                // if( (xs1.x == 0 || xs1.y == 0) || (xs2.x == 0 || xs2.y == 0) )
                // {
                //     ASSERT( 0 );
                // }

                // Now decompress the X's
                GRCompressedQualitiesRequest req = {};
                req.plotId                = filePlot.PlotId();
                req.compressionLevel      = filePlot.CompressionLevel();
                req.challenge             = challenge;
                req.xLinePoints[0].hi     = (uint64)(xLP0 >> 64);
                req.xLinePoints[0].lo     = (uint64)xLP0;

                if( needBothLeaves )
                {
                    req.xLinePoints[1].hi = (uint64)(xLP1 >> 64);
                    req.xLinePoints[1].lo = (uint64)xLP1;
                }

                const auto r = grGetFetchQualitiesXPair( gr, &req );
                ENSURE( r != GRResult_Failed );
                ENSURE( r != GRResult_OutOfMemory );

                if( r == GRResult_NoProof )
                {
                    Log::Line( " [%-2u] Dropped.", i );
                    nFound --;
                    continue;
                }

                byte hash[32] = {};
                {
                    const size_t HASH_SIZE_MAX = 32 + CDiv( 2*50, 8 );

                    const size_t hashSize = 32 + CDiv( 2*k, 8 );
                    byte hashInput[HASH_SIZE_MAX] = {};

                    memcpy( hashInput, challenge, 32 );

                    Bits<HASH_SIZE_MAX-32> hashBits;
                    hashBits.Write( req.x2, k );
                    hashBits.Write( req.x1, k );

                    hashBits.ToBytes( hashInput+32 );

                    Sha256( hash, hashInput, hashSize );
                }

                char hashStr[65] = {};
                size_t _;
                BytesToHexStr( hash, 32, hashStr, sizeof( hashStr), _ );

                Log::Line( " [%-2u] 0x%s", i, hashStr );
            }
        }
    }

    Log::Line( "" );
    Log::Line( "Found %llu / %u proofs ( %.2lf%% )", nFound, iterations, nFound / (double)iterations * 100 );

    // Cleanup
    grDestroyContext( gr );
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
    uint64 p7BaseIndex = 0;
    const uint64 matchCount = reader.GetP7IndicesForF7( f7, p7BaseIndex );

    // uint64 fullProofXs[GR_POST_PROOF_X_COUNT];
    // uint64 proof   [32]  = {};
    // char   proofStr[513] = {};
    uint64 p7Entries[kEntriesPerPark] = {};

    int64 prevP7Park = -1;

    for( uint64 i = 0; i < matchCount; i++ )
    {
        const uint64 p7Index = p7BaseIndex + i;
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

        if( reader.FetchProof( t6Index, fullProofXs ) == ProofFetchResult::OK )
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

void Sha256( byte outHash[32], const byte* bytes, const size_t length )
{
    bls::Util::Hash256( outHash, bytes, length );
}