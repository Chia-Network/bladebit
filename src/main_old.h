#include <thread>
#include <cstdlib>
#include <string>

#include "Version.h"
#include "util/Util.h"
#include "util/Log.h"
#include "SysHost.h"
#include "plotmem/MemPlotter.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"

#pragma warning( push )

extern "C" {
    #include "bech32/segwit_addr.h"
}

#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma warning( disable : 6287  )
#pragma warning( disable : 4267  )
#pragma warning( disable : 26495 )
#include "bls.hpp"
#include "elements.hpp"
#include "schemes.hpp"
#include "util.hpp"
#pragma GCC diagnostic pop
#pragma warning( pop )

#define PLOT_FILE_PREFIX_LEN (sizeof("plot-k32-2021-08-05-18-55-")-1)
#define PLOT_FILE_FMT_LEN (sizeof( "/plot-k32-2021-08-05-18-55-77a011fc20f0003c3adcc739b615041ae56351a22b690fd854ccb6726e5f43b7.plot.tmp" ))

/// Internal Data Structures
struct Config
{
    uint            threads            = 0;
    uint            plotCount          = 1;
    bool            warmStart          = false;
    bool            disableNuma        = false;
    bool            disableCpuAffinity = false;

    bls::G1Element  farmerPublicKey;
    bls::G1Element* poolPublicKey      = nullptr;
    
    ByteSpan*       contractPuzzleHash = nullptr;
    const char*     outputFolder       = nullptr;

    int             maxFailCount       = 100;

    const char*     plotId             = nullptr;
    const char*     plotMemo           = nullptr;
    bool            showMemo           = false;
};

/// Internal Functions
void            ParseCommandLine( int argc, const char* argv[], Config& cfg );
bool            HexPKeyToG1Element( const char* hexKey, bls::G1Element& pkey );

ByteSpan        DecodePuzzleHash( const char* poolContractAddress );
void            GeneratePlotIdAndMemo( Config& cfg, byte plotId[32], byte plotMemo[48+48+32], uint16& outMemoSize );
bls::PrivateKey MasterSkToLocalSK( bls::PrivateKey& sk );
bls::G1Element  GeneratePlotPublicKey( const bls::G1Element& localPk, bls::G1Element& farmerPk, const bool includeTaproot );

std::vector<uint8_t> BytesConcat( std::vector<uint8_t> a, std::vector<uint8_t> b, std::vector<uint8_t> c );

void PrintSysInfo();
void GetPlotIdBytes( const std::string& plotId, byte outBytes[32] );
void PrintUsage();

#if _DEBUG
    std::string          HexToString( const byte* bytes, size_t length );
    std::vector<uint8_t> HexStringToBytes( const char* hexStr );
    std::vector<uint8_t> HexStringToBytes( const std::string& hexStr );
    void PrintPK( const bls::G1Element&  key );
    void PrintSK( const bls::PrivateKey& key );
#endif

//-----------------------------------------------------------
const char* USAGE = "bladebit [<OPTIONS>] [<out_dir>]\n"
R"(
<out_dir>: Output directory in which to output the plots.
           This directory must exist.

OPTIONS:

 -h, --help           : Shows this message and exits.

 -t, --threads        : Maximum number of threads to use.
                        For best performance, use all available threads (default behavior).
                        Values below 2 are not recommended.
 
 -n, --count          : Number of plots to create. Default = 1.

 -f, --farmer-key     : Farmer public key, specified in hexadecimal format.
                        *REQUIRED*

 -p, --pool-key       : Pool public key, specified in hexadecimal format.
                        Either a pool public key or a pool contract address must be specified.

 -c, --pool-contract  : Pool contract address, specified in hexadecimal format.
                        Address where the pool reward will be sent to.
                        Only used if pool public key is not specified.

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
)";


//-----------------------------------------------------------
int main( int argc, const char* argv[] )
{
    // Install a crash handler to dump our stack traces
    SysHost::InstallCrashHandler();

    // Parse command line info
    Config cfg;
    ParseCommandLine( argc-1, argv+1, cfg );

    // Create the plot output path
    size_t outputFolderLen = strlen( cfg.outputFolder );
    
    char* plotOutPath = new char[outputFolderLen + PLOT_FILE_FMT_LEN];

    if( outputFolderLen )
    {
        memcpy( plotOutPath, cfg.outputFolder, outputFolderLen );

        // Add a trailing slash, if we need one.
        if( plotOutPath[outputFolderLen-1] != '/' )
            plotOutPath[outputFolderLen++] = '/';
    }

    // Begin plotting
    PlotRequest req;
    ZeroMem( &req );

    // #TODO: Don't let this config to permanently remain on the stack
    MemPlotConfig plotCfg;
    plotCfg.threadCount   = cfg.threads;
    plotCfg.noNUMA        = cfg.disableNuma;
    plotCfg.noCPUAffinity = cfg.disableCpuAffinity;
    plotCfg.warmStart     = cfg.warmStart;

    MemPlotter plotter( plotCfg );

    byte   plotId[32];
    byte   memo  [48+48+32];
    uint16 memoSize;
    char   plotIdStr[65] = { 0 };

    int failCount = 0;
    for( uint i = 0; i < cfg.plotCount; i++ )
    {
        // Generate a new plot id
        GeneratePlotIdAndMemo( cfg, plotId, memo, memoSize );

        // Apply debug plot id and/or memo
        if( cfg.plotId )
            HexStrToBytes( cfg.plotId, 64, plotId, 32 );

        if( cfg.plotMemo )
        {
            const size_t memoLen = strlen( cfg.plotMemo );
            HexStrToBytes( cfg.plotMemo, memoLen, memo, memoLen/2 );
        }
        
        // Convert plot id to string
        {
            size_t numEncoded = 0;
            BytesToHexStr( plotId, sizeof( plotId), plotIdStr, sizeof( plotIdStr ), numEncoded );

            ASSERT( numEncoded == 32 );
            plotIdStr[64] = 0;
        }

        // Set the output path
        {
            time_t     now = time( nullptr  );
            struct tm* t   = localtime( &now ); ASSERT( t );
            
            const size_t r = strftime( plotOutPath + outputFolderLen, PLOT_FILE_FMT_LEN, "plot-k32-%Y-%m-%d-%H-%M-", t );
            if( r != PLOT_FILE_PREFIX_LEN )
                Fatal( "Failed to generate plot file." );

            memcpy( plotOutPath + outputFolderLen + PLOT_FILE_PREFIX_LEN     , plotIdStr, 64 );
            memcpy( plotOutPath + outputFolderLen + PLOT_FILE_PREFIX_LEN + 64, ".plot.tmp", sizeof( ".plot.tmp" ) );
        }

        Log::Line( "Generating plot %d / %d: %s", i+1, cfg.plotCount, plotIdStr );
        if( cfg.showMemo )
        {
            char memoStr[(48+48+32)*2 + 1];

            size_t numEncoded = 0;
            BytesToHexStr( memo, memoSize, memoStr, sizeof( memoStr ) - 1, numEncoded );
            memoStr[numEncoded*2] = 0;

            Log::Line( "Plot Memo: %s", memoStr );
        }
        Log::Line( "" );

        // Prepare the request
        req.outPath     = plotOutPath;
        req.plotId      = plotId;
        req.memo        = memo;
        req.memoSize    = memoSize;
        req.IsFinalPlot = i+1 == cfg.plotCount;

        // Plot it
        if( !plotter.Run( req ) )
        {
            Log::Error( "Error: Plot %s failed... Trying next plot.", plotIdStr );
            if( cfg.maxFailCount > 0 && ++failCount >= cfg.maxFailCount )
            {
                // #TODO: Wait for pending plot writes to disk
                Fatal( "Maximum number of plot failures reached. Exiting." );
            }
        }

        Log::Line( "" );
    }
    
    Log::Flush();
    return 0;
}

//-----------------------------------------------------------
void ParseCommandLine( int argc, const char* argv[], Config& cfg )
{
    #define check( a ) (strcmp( a, arg ) == 0)
    int i;
    const char* arg = nullptr;

    auto value = [&](){

        if( ++i >= argc )
            Fatal( "Expected a value for parameter '%s'", arg );

        return argv[i];
    };

    auto ivalue = [&]() {

        const char* val = value();
        int64 v = 0;
        
        int r = sscanf( val, "%lld", &v );
        if( r != 1 )
            Fatal( "Invalid value for argument '%s'.", arg );

        return v;
    };

    auto uvalue = [&]() {
        
        int64 v = ivalue();
        if( v < 0 || v > 0xFFFFFFFF )
            Fatal( "Invalid value for argument '%s'. Value '%ld' is out of range.", arg, v );

        return (uint32)v;
    };

    const char* farmerPublicKey     = nullptr;
    const char* poolPublicKey       = nullptr;
    const char* poolContractAddress = nullptr;

    for( i = 0; i < argc; i++ )
    {
        arg = argv[i];

        if( check( "-h" ) || check( "--help") )
        {
            PrintUsage();
            exit( 0 );
        }
        else if( check( "-t" ) || check( "--threads") )
        {
            cfg.threads = uvalue();
            
            if( cfg.threads == 1 )
                Log::Line( "Warning: Only 1 thread was specified. Sub-optimal performance expected." );
        }
        else if( check( "-n" ) || check( "--count" ) )
        {
            cfg.plotCount = uvalue();
            if( cfg.plotCount < 1 )
            {
                Log::Line( "Warning: Invalid plot count specified. Setting it to 1." );
                cfg.plotCount = 1;
            }
        }
        else if( check( "-f" ) || check( "--farmer-key" ) )
        {
            farmerPublicKey = value();
        }
        else if( check( "-p" ) || check( "--pool-key" ) )
        {
            poolPublicKey = value();
        }
        else if( check( "-c" ) || check( "--pool-contract" ) )
        {
            poolContractAddress = value();
        }
        else if( check( "-w" ) || check( "--warm-start" ) )
        {
            cfg.warmStart = true;
        }
        else if( check( "-i" ) || check( "--plot-id" ) )
        {
            cfg.plotId = value();

            size_t len = strlen( cfg.plotId );
            if( len < 64 && len != 66 )
                Fatal( "Invalid plot id." );
            
            if( len == 66 )
            {
                if( cfg.plotId[0] == '0' && cfg.plotId[1] == 'x' )
                    cfg.plotId += 2;
                else
                    Fatal( "Invalid plot id." );
            }
        }
        else if( check( "--memo" ) )
        {
            cfg.plotMemo = value();

            size_t len = strlen( cfg.plotMemo );
            if( len > 2 && cfg.plotMemo[0] == '0' && cfg.plotMemo[1] == 'x' )
            {
                cfg.plotMemo += 2;
                len -= 2;
            }
            
            if( len/2 != (48 + 48 + 32) && len != (32 + 48 + 32) )
                Fatal( "Invalid plot memo." );
        }
        else if( check( "--show-memo" ) )
        {
            cfg.showMemo = true;
        }
        else if( check( "-m" ) || check( "--no-numa" ) )
        {
            cfg.disableNuma = true;
        }
        else if( check( "--no-cpu-affinity" ) )
        {
            cfg.disableCpuAffinity = true;
        }
        else if( check( "-v" ) || check( "--verbose" ) )
        {
            Log::SetVerbose( true );
        }
        else if( check( "--memory" ) )
        {
            // #TODO: Get this value from Memplotter
            const size_t requiredMem  = 416ull GB;
            const size_t availableMem = SysHost::GetAvailableSystemMemory();
            const size_t totalMem     = SysHost::GetTotalSystemMemory();

            Log::Line( "required : %llu", requiredMem  );
            Log::Line( "total    : %llu", totalMem     );
            Log::Line( "available: %llu", availableMem );

            exit( 0 );
        }
        else if( check( "--memory-json" ) )
        {
            // #TODO: Get this value from Memplotter
            const size_t requiredMem  = 416ull GB;
            const size_t availableMem = SysHost::GetAvailableSystemMemory();
            const size_t totalMem     = SysHost::GetTotalSystemMemory();

            Log::Line( "{ \"required\": %llu, \"total\": %llu, \"available\": %llu }",
                         requiredMem, totalMem, availableMem );

            exit( 0 );
        }
        else if( check( "--version" ) )
        {
            Log::Line( BLADEBIT_VERSION_STR );
            exit( 0 );
        }
        else
        {
            if( i+1 < argc )
            {
                Fatal( "Unexpected argument '%s'.", arg );
                exit( 1 );
            }

            cfg.outputFolder = arg;
        }
    }
    #undef check


    if( farmerPublicKey )
    {
        if( !HexPKeyToG1Element( farmerPublicKey, cfg.farmerPublicKey ) )
            Fatal( "Failed to parse farmer public key '%s'.", farmerPublicKey );
        
        // Remove 0x prefix for printing
        if( farmerPublicKey[0] == '0' && farmerPublicKey[1] == 'x' )
            farmerPublicKey += 2;
    }
    else
        Fatal( "A farmer public key is required. Please specify a farmer public key." );

    if( poolPublicKey )
    {
        if( poolContractAddress )
            Log::Write( "Warning: Pool contract address will not be used. A pool public key was specified." );
        
        // cfg.poolPublicKey = new bls::G1Element()
        bls::G1Element poolPubG1;
        if( !HexPKeyToG1Element( poolPublicKey, poolPubG1 ) )
            Fatal( "Error: Failed to parse pool public key '%s'.", poolPublicKey );

        cfg.poolPublicKey = new bls::G1Element( std::move( poolPubG1 ) );

        // Remove 0x prefix for printing
        if( poolPublicKey[0] == '0' && poolPublicKey[1] == 'x' )
            poolPublicKey += 2;
    }
    else if( poolContractAddress )
    {
        cfg.contractPuzzleHash = new ByteSpan( std::move( DecodePuzzleHash( poolContractAddress ) ) );
    }
    else
        Fatal( "Error: Either a pool public key or a pool contract address must be specified." );


    const uint threadCount = SysHost::GetLogicalCPUCount();

    if( cfg.threads == 0 )
        cfg.threads = threadCount;
    else if( cfg.threads > threadCount )
    {
        Log::Write( "Warning: Lowering thread count from %d to %d, the native maximum.", 
                    cfg.threads, threadCount );

        cfg.threads = threadCount;
    }
    
    if( cfg.plotCount < 1 )
        cfg.plotCount = 1;

    if( cfg.outputFolder == nullptr )
    {
        Log::Line( "Warning: No output folder specified. Using current directory." );
        cfg.outputFolder = "";
    }

    Log::Line( "Creating %d plots:", cfg.plotCount );
    
    if( cfg.outputFolder )
        Log::Line( " Output path           : %s", cfg.outputFolder );
    else
        Log::Line( " Output path           : Current directory." );

    Log::Line( " Thread count          : %d", cfg.threads );
    Log::Line( " Warm start enabled    : %s", cfg.warmStart ? "true" : "false" );


    Log::Line( " Farmer public key     : %s", farmerPublicKey );

    if( poolPublicKey )
        Log::Line( " Pool public key       : %s", poolPublicKey   );
    else if( poolContractAddress )
        Log::Line( " Pool contract address : %s", poolContractAddress );
    
    Log::Line( "" );
}

//-----------------------------------------------------------
void GeneratePlotIdAndMemo( Config& cfg, byte plotId[32], byte plotMemo[48+48+32], uint16& outMemoSize )
{
    bls::G1Element& farmerPK = cfg.farmerPublicKey;
    bls::G1Element* poolPK   = cfg.poolPublicKey;

    // Generate random master secret key
    byte seed[32];
    SysHost::Random( seed, sizeof( seed ) );

    bls::PrivateKey sk      = bls::AugSchemeMPL().KeyGen( bls::Bytes( seed, sizeof( seed ) ) );
    bls::G1Element  localPk = std::move( MasterSkToLocalSK( sk ) ).GetG1Element();

    // #See: chia-blockchain create_plots.py
    //       The plot public key is the combination of the harvester and farmer keys
    //       New plots will also include a taproot of the keys, for extensibility
    const bool includeTaproot = cfg.contractPuzzleHash != nullptr;
    
    bls::G1Element plotPublicKey = std::move( GeneratePlotPublicKey( localPk, farmerPK, includeTaproot ) );
    
    std::vector<uint8_t> farmerPkBytes = farmerPK.Serialize();
    std::vector<uint8_t> localSkBytes  = sk.Serialize();

    // The plot id is based on the harvester, farmer, and pool keys
    if( !includeTaproot )
    {
        std::vector<uint8_t> bytes = poolPK->Serialize();
        
        // Gen plot id
        auto plotPkBytes = plotPublicKey.Serialize();
        bytes.insert( bytes.end(), plotPkBytes.begin(), plotPkBytes.end() );

        bls::Util::Hash256( plotId, bytes.data(), bytes.size() );

        // Gen memo
        auto memoBytes = BytesConcat( poolPK->Serialize(), farmerPkBytes, localSkBytes );

        const size_t poolMemoSize = 48 + 48 + 32;
        ASSERT( memoBytes.size() == poolMemoSize );

        memcpy( plotMemo, memoBytes.data(), poolMemoSize );
        outMemoSize = (uint16)poolMemoSize;
    }
    else
    {
        // Create a pool plot with a contract puzzle hash
        ASSERT( cfg.contractPuzzleHash );

        const auto& ph = *cfg.contractPuzzleHash;
        std::vector<uint8_t> phBytes( (uint8_t*)ph.values, (uint8_t*)ph.values + ph.length );
        
        // Gen plot id
        std::vector<uint8_t> plotIdBytes = phBytes;
        auto plotPkBytes = plotPublicKey.Serialize();

        plotIdBytes.insert( plotIdBytes.end(), plotPkBytes.begin(), plotPkBytes.end() );
        bls::Util::Hash256( plotId, plotIdBytes.data(), plotIdBytes.size() );

        // Gen memo
        auto memoBytes = BytesConcat( phBytes, farmerPkBytes, localSkBytes );

        const size_t phMemoSize = 32 + 48 + 32;
        ASSERT( memoBytes.size() == phMemoSize );

        memcpy( plotMemo, memoBytes.data(), phMemoSize );
        outMemoSize = (uint16)phMemoSize;
    }
}

//-----------------------------------------------------------
bls::PrivateKey MasterSkToLocalSK( bls::PrivateKey& sk )
{
    // #SEE: chia-blockchain: derive-keys.py
    // EIP 2334 bls key derivation
    // https://eips.ethereum.org/EIPS/eip-2334
    // 12381 = bls spec number
    // 8444  = Chia blockchain number and port number
    // 0, 1, 2, 3, 4, 5, 6 farmer, pool, wallet, local, backup key, singleton, pooling authentication key numbers

    const uint32 blsSpecNum         = 12381;
    const uint32 chiaBlockchainPort = 8444; 
    const uint32 localIdx           = 3;

    bls::PrivateKey ssk = bls::AugSchemeMPL().DeriveChildSk( sk, blsSpecNum );
    ssk = bls::AugSchemeMPL().DeriveChildSk( ssk, chiaBlockchainPort );
    ssk = bls::AugSchemeMPL().DeriveChildSk( ssk, localIdx );
    ssk = bls::AugSchemeMPL().DeriveChildSk( ssk, 0        );

    return ssk;
}

//-----------------------------------------------------------
bls::G1Element GeneratePlotPublicKey( const bls::G1Element& localPk, bls::G1Element& farmerPk, const bool includeTaproot )
{
    bls::G1Element plotPublicKey;

    if( includeTaproot )
    {
        std::vector<uint8_t> taprootMsg = (localPk + farmerPk).Serialize();
        taprootMsg = BytesConcat( taprootMsg, localPk.Serialize(), farmerPk.Serialize() );
        
        byte tapRootHash[32];
        bls::Util::Hash256( tapRootHash, taprootMsg.data(), taprootMsg.size() );

        bls::PrivateKey taprootSk = bls::AugSchemeMPL().KeyGen( bls::Bytes( tapRootHash, sizeof( tapRootHash ) ) );
        
        plotPublicKey = localPk + farmerPk + taprootSk.GetG1Element();
    }
    else
    {
        plotPublicKey = localPk + farmerPk;
    }

    return plotPublicKey;
}

//-----------------------------------------------------------
ByteSpan DecodePuzzleHash( const char* poolContractAddress )
{
    ASSERT( poolContractAddress );

    size_t length = strlen( poolContractAddress );

    if( length < 9 )
        Fatal( "Error: Invalid pool contract address '%s'.", poolContractAddress );

    char* hrp  = (char*)malloc( length - 6 );
    byte* data = (byte*)malloc( length - 8 );

    size_t dataLength = 0;
    bech32_encoding encoding = bech32_decode( hrp, data, &dataLength, poolContractAddress );
    if( encoding == BECH32_ENCODING_NONE )
        Fatal( "Error: Failed to decode contract address '%s'.", poolContractAddress );

    ASSERT( dataLength > 0 );
    free( hrp );

    // See: convertbits in bech32m.py
    // This extends fields from 5 bits to 8 bits
    byte* decoded = (byte*)malloc( length - 8 );

    const uint fromBits = 5;
    const uint toBits   = 8;

    uint acc     = 0;
    uint bits    = 0;
    uint maxv    = (1 << toBits) - 1;
    uint maxAcc  = (1 << (fromBits + toBits - 1)) - 1;
    uint bitsLen = 0;

    for( size_t i = 0; i < dataLength; i++ )
    {
        uint value = data[i];

        if( value < 0 || (value >> fromBits) )
            Fatal( "Error: Invalid pool contract address '%s'. Could not decode bits.", poolContractAddress );

        acc = ((acc << fromBits) | value) & maxAcc;
        bits += fromBits;
        
        while( bits >= toBits )
        {
            ASSERT( bitsLen < length-8 );
            bits -= toBits;
            decoded[bitsLen++] = (acc >> bits) & maxv;
        }
    }

    if( bits >= fromBits || ((acc << (toBits - bits)) & maxv) )
        Fatal( "Error: Invalid pool contract address bits '%s'.", poolContractAddress );

    free( data );

    return ByteSpan( decoded, bitsLen );
}

//-----------------------------------------------------------
bool HexPKeyToG1Element( const char* hexKey, bls::G1Element& pkey )
{
    ASSERT( hexKey );
    
    size_t length = strlen( hexKey );

    if( length < bls::G1Element::SIZE*2 )
        return false;

    if( hexKey[0] == '0' && hexKey[1] == 'x' )
    {
        hexKey += 2;
        length -= 2;
    }

    if( length != bls::G1Element::SIZE*2 )
        return false;

    byte g1Buffer[bls::G1Element::SIZE];
    HexStrToBytes( hexKey, length, g1Buffer, sizeof( g1Buffer ) );

    bls::Bytes g1Bytes( g1Buffer, sizeof( g1Buffer ) );

    pkey = bls::G1Element::FromBytes( g1Bytes );
    
    return pkey.IsValid();
}

//-----------------------------------------------------------
void GetPlotIdBytes( const std::string& plotId, byte outBytes[32] )
{
    const char* pId = plotId.c_str();
    if( plotId.length() == 66 )
    {
        ASSERT( pId[0] == '0' && pId[1] == 'x' );
        pId += 2;
    }

    HexStrToBytes( pId, 64, outBytes, 32 );
}


//-----------------------------------------------------------
inline std::vector<uint8_t> BytesConcat( std::vector<uint8_t> a, std::vector<uint8_t> b, std::vector<uint8_t> c )
{
    a.insert( a.end(), b.begin(), b.end() );
    a.insert( a.end(), c.begin(), c.end() );
    return a;
}

//-----------------------------------------------------------
void PrintUsage()
{
    fputs( USAGE, stderr );
    fflush( stderr );
}

#if _DEBUG
    //-----------------------------------------------------------
    std::string HexToString( const byte* bytes, size_t length )
    {
        ASSERT( length );

        const size_t slen = length * 2 + 1;
        char* buffer      = (char*)malloc( slen );
        memset( buffer, 0, slen );
        
        size_t numEncoded;
        BytesToHexStr( bytes, length, buffer, slen, numEncoded );

        std::string str( buffer );
        free( buffer );

        return str;
    }

    //-----------------------------------------------------------
    std::vector<uint8_t> HexStringToBytes( const char* hexStr )
    {
        const size_t len  = strlen( hexStr );

        byte* buffer = (byte*)malloc( len / 2 );

        HexStrToBytes( hexStr, len, buffer, len / 2 );
        std::vector<uint8_t> ret( buffer, buffer + len / 2 );
        
        free( buffer );
        return ret;
    }

    //-----------------------------------------------------------
    std::vector<uint8_t> HexStringToBytes( const std::string& hexStr )
    {
        return HexStringToBytes( hexStr.c_str() );
    }

    //-----------------------------------------------------------
    void PrintPK( const bls::G1Element& key )
    {
        std::vector<uint8_t> bytes = key.Serialize();
        Log::Line( "%s", HexToString( (byte*)bytes.data(), bytes.size() ).c_str() );
    }

    //-----------------------------------------------------------
    void PrintSK( const bls::PrivateKey& key )
    {
        std::vector<uint8_t> bytes = key.Serialize();
        Log::Line( "%s", HexToString( (byte*)bytes.data(), bytes.size() ).c_str() );
    }
#endif
