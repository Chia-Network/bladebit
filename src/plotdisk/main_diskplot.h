#include <stdlib.h>
#include "DiskPlotter.h"
#include "SysHost.h"
#include "io/FileStream.h"
#include "util/Log.h"
#include "plotting/PlotTools.h"

// TEST
#include "plotdisk/jobs/IOJob.h"
#include "threading/ThreadPool.h"
#include <sys/resource.h>

// TEST:
void TestDiskBackPointers();
void TestLookupMaps();

struct GlobalConfig
{

};


//-----------------------------------------------------------
void ParseConfig( int argc, const char* argv[], GlobalConfig& gConfig, DiskPlotter::Config& cfg );

size_t ParseSize( const char* arg, const char* sizeText );
size_t ValidateTmpPathAndGetBlockSize( DiskPlotter::Config& cfg );

//-----------------------------------------------------------
int WriteTest( int argc, const char* argv[] )
{
    uint         threadCount  = 4;
    const size_t bufferSize   = 64ull GB;
    // const uint   bufferCount  = 4;
    const char*  filePath     = "/mnt/p5510a/test.tmp";

    // if( argc > 0 )
    // {
    //     int r = sscanf( argv[0], "%lu", &threadCount );
    //     FatalIf( r != 1, "Invalid value for threadCount" );
    // }

    Log::Line( "Threads    : %u", threadCount );
    Log::Line( "Buffer Size: %llu GiB", bufferSize BtoGB );
    
    ThreadPool pool( threadCount );
    FileStream* files        = new FileStream[threadCount];
    byte**      blockBuffers = new byte*[threadCount];

    for( uint i = 0; i < threadCount; i++ )
    {
        FatalIf( !files[i].Open( filePath, FileMode::OpenOrCreate, FileAccess::ReadWrite, 
                  FileFlags::LargeFile | FileFlags::NoBuffering ), "Failed to open file." );

        blockBuffers[i] = (byte*)SysHost::VirtualAlloc( files[i].BlockSize() );
    }
    Log::Line( "Allocating buffer..." );
    byte* buffer = (byte*)SysHost::VirtualAlloc( bufferSize, true ); 

    Log::Line( "Writing..." );
    int error = 0;
    double elapsed = IOJob::WriteWithThreads( 
        threadCount, pool, files, buffer, bufferSize, blockBuffers, files[0].BlockSize(), error );

    if( error )
        Fatal( "Error when writing: %d (0x%x)", error, error );

    const double writeRate = bufferSize / elapsed;

    Log::Line( "Wrote %.2lf GiB in %.4lf seconds: %.2lf MiB/s (%.2lf MB/s)", 
               (double)bufferSize BtoGB, elapsed, writeRate BtoMB, writeRate / 1000 / 1000 );

    FatalIf( !files[0].Seek( 0, SeekOrigin::Begin ), 
            "Failed to seek file with error %d.", files[0].GetError() );
    
    Log::Line( "Reading..." );
    auto readTimer = TimerBegin();
    IOJob::ReadFromFile( files[0], buffer, bufferSize, blockBuffers[0], files[0].BlockSize(), error );
    elapsed = TimerEnd( readTimer );

    FatalIf( error, "Error reading: %d, (0x%x)", error, error );

    const double readRate = bufferSize / elapsed;
    Log::Line( "Read %.2lf GiB in %.4lf seconds: %.2lf MiB/s (%.2lf MB/s)", 
               (double)bufferSize BtoGB, elapsed, readRate BtoMB, readRate / 1000 / 1000 );

    return 0;
}

//-----------------------------------------------------------
int main( int argc, const char* argv[] )
{
    // Install a crash handler to dump our stack traces
    SysHost::InstallCrashHandler();

    struct rlimit limit;
    getrlimit( RLIMIT_NOFILE, &limit );
    Log::Line( "%u / %u", limit.rlim_cur, limit.rlim_max );
    
    limit.rlim_cur = limit.rlim_max;
    setrlimit( RLIMIT_NOFILE, &limit );

    #if _DEBUG
        Log::Line( "DEBUG: ON" );
    #else
        Log::Line( "DEBUG: OFF" );
    #endif

    #if _NDEBUG
        Log::Line( "NDEBUG: ON" );
    #else
        Log::Line( "NDEBUG: OFF" );
    #endif

    // TestDiskBackPointers(); return 0;
    // TestLookupMaps(); return 0;

    argc--;
    argv++;
    // return WriteTest( argc, argv );

    DiskPlotter::Config plotCfg;
    GlobalConfig gCfg;

    ParseConfig( argc, argv, gCfg, plotCfg );

    DiskPlotter plotter( plotCfg );

    byte*   plotId       = new byte[BB_PLOT_ID_LEN];
    byte*   plotMemo     = new byte[BB_PLOT_MEMO_MAX_SIZE];
    char*   plotFileName = new char[BB_PLOT_FILE_LEN_TMP];
    uint16  plotMemoSize = 0;

    DiskPlotter::PlotRequest req;
    req.plotFileName = plotFileName;
    req.plotId       = plotId;
    req.plotMemo     = plotMemo;
    // #TODO: Generate plot id & memo


    // TEST
    // #TODO: Remove
    {
        const char refPlotId  [] = "c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835";
        const char refPlotMemo[] = "80a836a74b077cabaca7a76d1c3c9f269f7f3a8f2fa196a65ee8953eb81274eb8b7328d474982617af5a0fe71b47e9b8ade0cc43610ce7540ab96a524d0ab17f5df7866ef13d1221a7203e5d10ad2a4ae37f7b73f6cdfd6ddf4122e8a1c2f8ef01b7bf8a22a9ac82a003e07b551c851ea683839f3e1beb8ac9ede57d2c020669";

        memset( plotId  , 0, BB_PLOT_ID_LEN );
        memset( plotMemo, 0, BB_PLOT_MEMO_MAX_SIZE );

        HexStrToBytes( refPlotId  , sizeof( refPlotId   )-1, plotId  , BB_PLOT_ID_LEN );
        HexStrToBytes( refPlotMemo, sizeof( refPlotMemo )-1, plotMemo, BB_PLOT_MEMO_MAX_SIZE );
        
        req.plotMemoSize = 128;
    }

    PlotTools::GenPlotFileName( plotId, plotFileName );
    plotter.Plot( req );

    exit( 0 );
}

//-----------------------------------------------------------
void ParseConfig( int argc, const char* argv[], GlobalConfig& gConfig, DiskPlotter::Config& cfg )
{
    #define check( a ) (strcmp( a, arg ) == 0)
    int i;
    const char* arg = nullptr;

    auto value = [&](){

        FatalIf( ++i >= argc, "Expected a value for parameter '%s'", arg );
        return argv[i];
    };

    auto ivalue = [&]() {

        const char* val = value();
        int64 v = 0;
        
        int r = sscanf( val, "%lld", &v );
        FatalIf( r != 1, "Invalid int64 value for argument '%s'.", arg );

        return v;
    };

    auto uvalue = [&]() {
        
        const char* val = value();
        uint64 v = 0;

        int r = sscanf( val, "%llu", &v );
        FatalIf( r != 1, "Invalid uint64 value for argument '%s'.", arg );

        return v;
    };


     // Set defaults
    const size_t f1DefaultWriteInterval    = 128ull MB;
    size_t       fxDefaultWriteInterval    = 64ull  MB;
    size_t       matchDefaultWriteInterval = 64ull  MB;
    const uint   minBufferCount            = 3;

    // Parse fx and match per-table
    auto checkFx = [&]( const char* a ) {
        
        const size_t minSize = sizeof( "--fx" ) - 1;
        const size_t len     = strlen( a );

        if( len >= minSize && memcmp( "--fx", a, minSize ) == 0 )
        {
            if( len == minSize )
            {
                // Set the default value
                fxDefaultWriteInterval = ParseSize( "--fx", value() );
                return true;
            }
            else
            {
                // Set the value for a table (--fx2, --fx4...)
                const char intChar = a[minSize];
                
                // Expect an integer from 2-7 (inclusive) to come immediately after --fx
                if( intChar >= '2' && intChar <= '7' )
                {
                    const int tableId = (int)intChar - '0';
                    cfg.writeIntervals[tableId].fxGen = ParseSize( "--fn", value() );
                    
                    return true;
                }
            }
        }

        return false;
    };

    auto checkMatch = [&]( const char* a ) {
        
        const size_t minSize = sizeof( "--match" ) - 1;
        const size_t len     = strlen( a );

        if( len >= minSize && memcmp( "--match", a, minSize ) == 0 )
        {
            if( len == minSize )
            {
                // Set the default value
                matchDefaultWriteInterval = uvalue();
                return true;
            }
            else
            {
                // Set the value for a table (--fx2, --fx4...)
                const char intChar = a[minSize];
                
                // Expect an integer from 2-7 (inclusive) to come immediately after --fx
                if( intChar >= '2' && intChar <= '7' )
                {
                    const int tableId = (int)intChar - '0';
                    cfg.writeIntervals[tableId].matching = uvalue();
                    
                    return true;
                }
            }
        }

        return false;
    };
   
    // Set some defaults
    ZeroMem( &cfg );
    cfg.workHeapSize = BB_DP_MIN_RAM_SIZE;  // #TODO: I don't think we need this anymore.
                                            //        We only need a variable size for the write intervals.
                                            //        This size will be fixed.

    cfg.ioBufferCount = minBufferCount;


    // Start parsing
    for( i = 0; i < argc; i++ )
    {
        arg = argv[i];

        if( check( "--f1" ) )
        {
            cfg.writeIntervals[0].fxGen = ParseSize( arg, value() );
        }
        else if( checkFx( arg ) || checkMatch( arg ) )
        {
            continue;
        }
        else if( check( "-t" ) || check( "--threads" ) )
        {
            cfg.workThreadCount = (uint)uvalue();
        }
        else if( check( "-b" ) || check( "--buffer-count" ) )
        {
            cfg.ioBufferCount = (uint)uvalue();
        }
        // else if( check( "-d" ) || check( "--direct-io" ) )
        // {
        //     cfg.enableDirectIO = true;
        // }
        // else if( check( "--io-threads" ) )
        // {
        //     cfg.ioThreadCount = (uint)uvalue();
        // }
        // else if( check( "-h" ) || check( "--heap" ) )
        // {
        //     ParseSize( arg, value() );
        // }
        else if( check( "--temp" ) )
        {
            cfg.tmpPath = value();
        }
        else if( i == argc - 1 )
        {
            cfg.tmpPath = arg;
        }
        else
        {
            Fatal( "Error: Unexpected argument '%s'.", arg );
        }
    }

    
    // Validate some parameters
    const size_t diskBlockSize = ValidateTmpPathAndGetBlockSize( cfg );

    const size_t minBucketSize = BB_DP_MAX_ENTRIES_PER_BUCKET * sizeof( uint32 );

    size_t maxWriteInterval = 0;

    for( TableId table = TableId::Table1; table < TableId::_Count; table++ )
    {
        auto& writeInterval = cfg.writeIntervals[(int)table];

        if( writeInterval.fxGen == 0 )
            writeInterval.fxGen = table == TableId::Table1 ? f1DefaultWriteInterval : fxDefaultWriteInterval;

        if( writeInterval.matching == 0 )
            writeInterval.matching = matchDefaultWriteInterval;

        // Ensure the intervals are <= than the minimum write size of a bucket (table 7)
        // and >= disk block size of the temporary directory.
        FatalIf( writeInterval.fxGen    > minBucketSize, "f%d write interval must be less or equal than %llu bytes.", (int)table+1, minBucketSize );
        FatalIf( writeInterval.matching > minBucketSize, "Table %d match write interval must be less or equal than %llu bytes.", (int)table+1, minBucketSize );
        FatalIf( writeInterval.fxGen    < diskBlockSize, "f%d write interval must be greater or equal than the tmp directory block size of %llu bytes.", (int)table+1, diskBlockSize );
        FatalIf( writeInterval.matching < diskBlockSize, "Table %d match write interval must be greater or equal than the tmp directory block size of %llu bytes.", (int)table + 1, minBucketSize );

        // Round up the size to the block size
        writeInterval.fxGen    = RoundUpToNextBoundaryT( writeInterval.fxGen   , diskBlockSize );
        writeInterval.matching = RoundUpToNextBoundaryT( writeInterval.matching, diskBlockSize );

        maxWriteInterval = std::max( maxWriteInterval, writeInterval.fxGen    );
        maxWriteInterval = std::max( maxWriteInterval, writeInterval.matching );
    }

    cfg.ioBufferSize = maxWriteInterval;

    FatalIf( cfg.ioBufferCount < 3, "IO buffer (write interval buffers) cont must be 3 or more." );
    

    const uint sysLogicalCoreCount = SysHost::GetLogicalCPUCount();

    if( cfg.workThreadCount == 0 )
        cfg.workThreadCount = sysLogicalCoreCount;
    else if( cfg.workThreadCount > sysLogicalCoreCount )
    {
        Log::Line( "Warning: Limiting work threads to %u, which is the system's logical CPU count.", sysLogicalCoreCount );
        cfg.workThreadCount = sysLogicalCoreCount;
    }

    if( cfg.ioThreadCount == 0 )
        cfg.ioThreadCount = 1;        // #TODO: figure out a reasonable default. Probably 1 or 2 for current consumer NVMes running on PCIe3...
    else if( cfg.ioThreadCount > sysLogicalCoreCount )
    {
        Log::Line( "Warning: Limiting disk queue threads to %u, which is the system's logical CPU count.", sysLogicalCoreCount );
        cfg.ioThreadCount = sysLogicalCoreCount;
    }
}

//-----------------------------------------------------------
size_t ParseSize( const char* arg, const char* sizeText )
{
    const size_t len = strlen( sizeText );
    const char* end = sizeText + len;

    const char* suffix = sizeText;

#ifdef _WIN32
    #define StriCmp _stricmp
#else
    #define StriCmp strcasecmp
#endif

    // Try to find a suffix:
    //  Find the first character that's not a digit
    do
    {
        const char c = *suffix;
        if( c < '0' || c > '9' )
            break;
    }
    while( ++suffix < end );

    // Apply multiplier depending on the suffix
    size_t multiplier = 1;

    const size_t suffixLength = end - suffix;
    if( suffixLength > 0 )
    {
        if( StriCmp( "GB", suffix ) == 0 || StriCmp( "G", suffix ) == 0 )
            multiplier = 1ull GB;
        else if( StriCmp( "MB", suffix ) == 0 || StriCmp( "M", suffix ) == 0 )
            multiplier = 1ull MB;
        else if( StriCmp( "KB", suffix ) == 0 || StriCmp( "K", suffix ) == 0 )
            multiplier = 1ull KB;
        else
            Fatal( "Invalid suffix '%s' for argument '%s'", suffix, arg );
    }

    const size_t MAX_DIGITS = 19;
    char digits[MAX_DIGITS + 1];

    const size_t digitsLength = suffix - sizeText;
    FatalIf( digitsLength < 1 || digitsLength > MAX_DIGITS, "Invalid parameters value for argument '%s'.", arg );

    // Read digits
    size_t size = 0;

    memcpy( digits, sizeText, digitsLength );
    digits[digitsLength] = 0;

    FatalIf( sscanf( digits, "%llu", &size ) != 1, 
             "Invalid parameters value for argument '%s'.", arg );

    const size_t resolvedSize = size * multiplier;
    
    // Check for overflow
    FatalIf( resolvedSize < size, "Size overflowed for argument '%s'.", arg );;

    return resolvedSize;
}

//-----------------------------------------------------------
size_t ValidateTmpPathAndGetBlockSize( DiskPlotter::Config& cfg )
{
    FatalIf( cfg.tmpPath == nullptr, "No temporary path specified." );

    size_t pathLen = strlen( cfg.tmpPath );
    FatalIf( pathLen < 1, "Invalid temporary path." );

    char* tmpPath = bbmalloc<char>( pathLen + 2 );
    memcpy( tmpPath, cfg.tmpPath, pathLen );

    if( cfg.tmpPath[pathLen - 1] != '/'
    #ifdef _WIN32
        && cfg.tmpPath[pathLen - 1] != '\\'
    #endif
        )
    {
        tmpPath[pathLen++] = '/';
    }

    tmpPath[pathLen] = (char)0;
    cfg.tmpPath = tmpPath;

    // Open a file in the temp dir to obtain the block size
    uint64 randNum = 0;
    SysHost::Random( (byte*)&randNum, sizeof( randNum ) );

    char* randFileName = bbmalloc<char>( pathLen + 32 );
    
    int r = snprintf( randFileName, pathLen + 32, "%s.%llx.blk", tmpPath, randNum );
    FatalIf( r < 1, "Unexpected error validating temp directory." );

    FileStream tmpFile;

    if( !tmpFile.Open( randFileName, FileMode::Create, FileAccess::ReadWrite ) )
    {
        int err = tmpFile.GetError();
        Fatal( "Failed to open a file in the temp directory with error %d (0x%x).", err, err );
    }
    

    cfg.expectedTmpDirBlockSize = tmpFile.BlockSize();

    remove( randFileName );
    free( randFileName );

    return cfg.expectedTmpDirBlockSize;
}

