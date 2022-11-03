#include "util/CliParser.h"
#include "util/Log.h"
#include "util/Util.h"
#include "io/HybridStream.h"
#include "plotdisk/jobs/IOJob.h"
#include "plotting/GlobalPlotConfig.h"
#include "util/jobs/MemJobs.h"

void IOTestPrintUsage();

static const size_t FILE_NAME_SIZE = 32;

static void GetTmpFileName( char fileName[FILE_NAME_SIZE] );
static void InitPages( ThreadPool& pool, const uint32 threadCount, void* mem, const size_t size );


//-----------------------------------------------------------
void IOTestMain( GlobalPlotConfig& gCfg, CliParser& cli )
{
    size_t      writeSize    = 4ull GB;
    const char* testDir      = nullptr;
    bool        noDirectIO   = false;
    uint32      passCount    = 1;
    size_t      memReserve   = 0;
    double      passDelaySec = 0.0;

    while( cli.HasArgs() )
    {
        if( cli.ReadSize( writeSize, "-s", "--size" ) )
        {
            FatalIf( writeSize < 1, "Write size must be > 0." );
        }
        else if( cli.ReadSwitch( noDirectIO, "-d", "--no-direct-io" ) )
            continue;
        else if( cli.ReadSize( memReserve, "-m", "--memory" ) )
            continue;
        else if( cli.ReadU32( passCount, "-p", "--passes" ) )
        {
            if( passCount < 1 ) passCount = 1;
            continue;
        }
        else if( cli.ReadF64( passDelaySec, "--delay" ) )
        {
            if( passDelaySec < 0.0 )
                passDelaySec = 0;
            continue;
        }
        else if( cli.ArgConsume( "-h", "--help" ) )
        {
            IOTestPrintUsage();
            exit( 0 );
        }
        else if( cli.IsLastArg() )
        {
            testDir = cli.ArgConsume();
        }
        else
        {
            Fatal( "Unexpected argument '%s'.", cli.Arg() );
        }
    }

    FatalIf( testDir == nullptr || testDir[0] == 0, 
        "Expected an output directory as the last argument." );


    char* filePath    = nullptr;
    char* fileNamePtr = nullptr;
    {
        size_t dirLen = strlen( testDir );

        filePath = new char[dirLen + FILE_NAME_SIZE + sizeof( ".tmp" ) ];
        if( dirLen > 0 )
        {
            memcpy( filePath, testDir, dirLen );
            if( filePath[dirLen-1] != '/' && filePath[dirLen-1] != '\\' )
                filePath[dirLen++] = '/';
        }

        fileNamePtr = filePath + dirLen;
        *fileNamePtr = 0;
        memcpy( fileNamePtr + FILE_NAME_SIZE, ".tmp", sizeof( ".tmp" ) );
    }
    ASSERT( filePath );

    FileFlags flags = FileFlags::LargeFile;
    if( !noDirectIO )
        flags |= FileFlags::NoBuffering;

    const uint32 maxThreads  = SysHost::GetLogicalCPUCount();
    const uint32 threadCount = gCfg.threadCount == 0 ? 1 : std::min( gCfg.threadCount, maxThreads );

    ThreadPool pool( threadCount, ThreadPool::Mode::Fixed, gCfg.disableCpuAffinity );

    Log::Line( "Size   : %.2lf MiB", (double)writeSize BtoMB );
    Log::Line( "Cache  : %.2lf MiB", (double)memReserve BtoMB );
    Log::Line( "Threads: %u",        threadCount );
    Log::Line( "Passes : %u",        passCount   );
    Log::Line( "Performing test with file %s", filePath );

    IStream** files = new IStream*[threadCount];

    byte* cache = nullptr;
    if( memReserve > 0 )
    {
        Log::Line( "Reserving memory cache..." );
        
        const size_t pageSize     = SysHost::GetPageSize();
        const size_t reserveAlloc = RoundUpToNextBoundaryT( memReserve, pageSize ) + pageSize * 2;

        cache = bbvirtalloc<byte>( reserveAlloc );
        InitPages( pool, threadCount, cache, reserveAlloc );

        // Add some guard pages
        SysHost::VirtualProtect( cache, pageSize, VProtect::NoAccess );
        SysHost::VirtualProtect( cache + reserveAlloc - pageSize, pageSize, VProtect::NoAccess );
        cache += pageSize;
    }

    // Open main file and view to the files
    {
        // Generate a random file name
        char fileName[FILE_NAME_SIZE];
        GetTmpFileName( fileName );
        memcpy( fileNamePtr, fileName, FILE_NAME_SIZE );

        if( memReserve > 0 )
        {
            auto* memFile = new HybridStream();
            FatalIf( !memFile->Open( cache, memReserve, filePath, FileMode::Create, FileAccess::ReadWrite, flags ), 
                "Failed to open temporary test file at path '%s'.", testDir );

            files[0] = memFile;
        }
        else
        {
            auto* diskFile = new FileStream();
            FatalIf( !diskFile->Open( filePath, FileMode::Create, FileAccess::ReadWrite, flags ), 
                "Failed to open temporary test file at path '%s'.", testDir );

            files[0] = diskFile;
        }

        // Open other views into the same file, if we have multiple-threads
        for( uint32 i = 1; i < threadCount; i++ )
        {
            if( memReserve > 0 )
            {
                auto* memFile = new HybridStream();
                FatalIf( !memFile->Open( cache, memReserve, filePath, FileMode::Open, FileAccess::ReadWrite, flags ), 
                    "Failed to open temporary test file at path '%s'.", testDir );

                files[i] = memFile;
            }
            else
            {
                auto* diskFile = new FileStream();
                FatalIf( !diskFile->Open( filePath, FileMode::Open, FileAccess::ReadWrite, flags ), 
                    "Failed to open temporary test file at path '%s' with error: %d.", testDir, diskFile->GetError() );

                files[i] = diskFile;
            }
        }
    }
    
    // Begin test
    const size_t fsBlockSize = files[0]->BlockSize();
    FatalIf( fsBlockSize < 1, "Invalid file system block size of 0." );

    const size_t totalWriteSize = RoundUpToNextBoundaryT( writeSize, fsBlockSize );
    const double sizeMB = (double)totalWriteSize BtoMB;

    // Allocate data
    Log::Line( "Allocating buffer..." );
    byte* buffer = bbvirtalloc<byte>( totalWriteSize );
    ASSERT( (uintptr_t)buffer / (uintptr_t)fsBlockSize * (uintptr_t)fsBlockSize == (uintptr_t)buffer );
    
    byte** blocks = new byte*[threadCount];
    for( uint32 i = 0; i < threadCount; i++ )
        blocks[i] = bbvirtalloc<byte>( fsBlockSize );


    InitPages( pool, threadCount, buffer, totalWriteSize );

    for( uint32 pass = 0; pass < passCount; pass++ )
    {
        if( passCount > 1 )
        {
            Log::Line( "[Pass %u/%u]", pass+1, passCount );
        }

        // Write
        Log::Line( "" );
        Log::Line( "Writing..." );
        int  err   = 0;
        auto timer = TimerBegin();
        FatalIf( !IOJob::MTWrite( pool, threadCount, files, buffer, totalWriteSize, (void**)blocks, fsBlockSize, err ),
            "Failed to write with error %d (0x%x).", err, err );
        auto elapsed = TimerEnd( timer );

        Log::Line( "Wrote %.2lf MiB in %.2lf seconds @ %.2lf MiB/s (%.2lf GiB/s) or %2.lf MB/s (%.2lf GB/s).", 
                    sizeMB, elapsed, totalWriteSize / elapsed BtoMB, totalWriteSize / elapsed BtoGB,
                    totalWriteSize / elapsed / 1000000.0, totalWriteSize / elapsed / 1000000000.0 );

        // Read
        Log::Line( "" );
        Log::Line( "Reading..." );

        for( uint32 i = 0; i < threadCount; i++ )
        {
            IStream* file = files[i];
            if( !file->Seek( 0, SeekOrigin::Begin ) )
            {
                err = file->GetError();
                Fatal( "Seek failed on test file with error%d (0x%x).", err, err );
            }
        }

        err   = 0;
        timer = TimerBegin();
        FatalIf( !IOJob::MTRead( pool, threadCount, files, buffer, totalWriteSize, (void**)blocks, fsBlockSize, err ),
            "Failed to read with error %d (0x%x)", err, err );
        elapsed = TimerEnd( timer );

        Log::Line( "Read %.2lf MiB in %.2lf seconds @ %.2lf MiB/s (%.2lf GiB/s) or %2.lf MB/s (%.2lf GB/s).", 
                    sizeMB, elapsed, totalWriteSize / elapsed BtoMB, totalWriteSize / elapsed BtoGB,
                    totalWriteSize / elapsed / 1000000.0, totalWriteSize / elapsed / 1000000000.0 );

        if( pass+1 < passCount )
        {
            for( uint32 i = 0; i < threadCount; i++ )
            {
                IStream* file = files[i];
                if( !file->Seek( 0, SeekOrigin::Begin ) )
                {
                    err = file->GetError();
                    Fatal( "Seek failed on test file with error%d (0x%x).", err, err );
                }
            }

            if( passDelaySec > 0 )
                Thread::Sleep( (long)( passDelaySec * 1000.0 ) );
        }
    }

    // Cleanup files
    for( uint32 i = 0; i < threadCount; i++ )
        delete files[i];

    remove( filePath );
}

//-----------------------------------------------------------
void GetTmpFileName( char fileName[FILE_NAME_SIZE] )
{
    ASSERT( fileName );

    byte random[FILE_NAME_SIZE/2];
    SysHost::Random( random, sizeof( random ) );

    size_t numEncoded = 0;
    BytesToHexStr( random, sizeof( random ), fileName, FILE_NAME_SIZE, numEncoded );
    ASSERT( numEncoded == sizeof( random ) );
}

//-----------------------------------------------------------
void InitPages( ThreadPool& pool, const uint32 threadCount, void* mem, const size_t size )
{
    FaultMemoryPages::RunJob( pool, threadCount, mem, size );
}


//-----------------------------------------------------------
static const char* USAGE = R"(iotest [OPTIONS] <test_dir>

Performs read/write test on the specified disk path.

[OPTIONS]
 -s, --size <size>  : Size to write. Default is 4GB.
                      Ex: 512MB 1GB 16GB

 -d, --no-direct-io : Disable direct IO, which enables OS IO buffering.

 -m, --memory <size>: Reserve memory size to use as IO cache for the file.
 
 -p, --passes <n>   : The number of passes to perform. By default it is 1.
 
 --delay <secs>     : Time (in seconds) to wait between passes.

 -h, --help         : Print this help message and exit.
)";

//-----------------------------------------------------------
void IOTestPrintUsage()
{
    Log::Line( USAGE );
}


