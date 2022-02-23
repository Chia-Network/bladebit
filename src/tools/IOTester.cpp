#include "util/CliParser.h"
#include "util/Log.h"
#include "util/Util.h"
#include "io/HybridStream.h"
#include "plotdisk/jobs/IOJob.h"

void IOTestPrintUsage();

static void InitPages( void* mem, size_t size );


//-----------------------------------------------------------
void IOTestMain( CliParser& cli )
{
    size_t      writeSize  = 4ull GB;
    const char* testDir    = nullptr;
    bool        noDirectIO = false;
    size_t      memReserve = 0;

    while( cli.HasArgs() )
    {
        if( cli.ReadValue( writeSize, "-s", "--size" ) )
        {
            FatalIf( writeSize < 1, "Write size must be > 0." );
        }
        else if( cli.ReadSwitch( noDirectIO, "-d", "--no-direct-io" ) )
            continue;
        else if( cli.ReadValue( memReserve, "-m", "--memory" ) )
            continue;
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

    char* filePath = nullptr;
    {
        char fileName[32+1] = { 0 };
        byte random[16];
        SysHost::Random( random, sizeof( random ) );

        size_t numEncoded = 0;
        BytesToHexStr( random, sizeof( random ), fileName, sizeof( fileName ) - 1, numEncoded );

        size_t dirLen = strlen( testDir );

        filePath = new char[dirLen + sizeof( fileName ) + 1];
        if( dirLen > 0 )
        {
            memcpy( filePath, testDir, dirLen );
            if( filePath[dirLen-1] != '/' && filePath[dirLen-1] != '\\' )
                filePath[dirLen++] = '/';
        }

        memcpy( filePath + dirLen, fileName, sizeof( fileName ) );
    }
    ASSERT( filePath );    

    FileFlags flags = FileFlags::LargeFile;
    if( !noDirectIO )
        flags |= FileFlags::NoBuffering;

    Log::Line( "Performing test with file %s", filePath );
    IStream* file = nullptr;

    // FileStream file;
    if( memReserve > 0 )
    {
        Log::Line( "Reserving memory cache..." );
        byte* cache = bbvirtalloc<byte>( memReserve );
        InitPages( cache, memReserve );

        auto* memFile = new HybridStream();
        FatalIf( !memFile->Open( cache, memReserve, filePath, FileMode::Create, FileAccess::ReadWrite, flags ), 
            "Failed to open temporary test file at path '%s'.", testDir );

        file = memFile;
    }
    else
    {
        auto* diskFile = new FileStream();
        FatalIf( !diskFile->Open( filePath, FileMode::Create, FileAccess::ReadWrite, flags ), 
            "Failed to open temporary test file at path '%s'.", testDir );

        file = diskFile;
    }
    
    // Begin test
    const size_t fsBlockSize = file->BlockSize();
    FatalIf( fsBlockSize < 1, "Invalid file system block size of 0." );

    const size_t totalWriteSize = RoundUpToNextBoundaryT( writeSize, fsBlockSize );

    const double sizeMB = (double)totalWriteSize BtoMB;

    // Allocate data
    Log::Line( "Allocating buffer..." );
    byte* buffer = bbvirtalloc<byte>( totalWriteSize );
    byte* block  = bbvirtalloc<byte>( fsBlockSize    );

    InitPages( buffer, totalWriteSize );
    

    // Write
    Log::Line( "" );
    Log::Line( "Writing..." );
    int err = 0;
    auto timer = TimerBegin();
    IOWriteJob::WriteToFile( *file, buffer, totalWriteSize, block, fsBlockSize, err );
    FatalIf( err, "Failed to write with error %d (0x%x).", err, err );
    auto elapsed = TimerEnd( timer );

    Log::Line( "Wrote %.2lf MiB in %.2lf seconds @ %.2lf MiB/s .", 
                sizeMB, elapsed, totalWriteSize / elapsed BtoMB );

    // Read
    Log::Line( "" );
    Log::Line( "Reading..." );
    if( !file->Seek( 0, SeekOrigin::Begin ) )
    {
        err = file->GetError();
        Fatal( "Seek failed on test file with error%d (0x%x).", err, err );
    }

    err = 0;
    timer = TimerBegin();
    IOWriteJob::ReadFromFile( *file, buffer, totalWriteSize, block, fsBlockSize, err );
    FatalIf( err, "Failed to read with error %d (0x%x)", err, err );
    elapsed = TimerEnd( timer );

    Log::Line( "Read %.2lf MiB in %.2lf seconds @ %.2lf MiB/s .", 
                sizeMB, elapsed, totalWriteSize / elapsed BtoMB );

    delete file;
    remove( filePath );
}

//-----------------------------------------------------------
void InitPages( void* mem, size_t size )
{
    // #TODO: Use job from Memplot to do this mult-threaded
    const size_t pageSize = SysHost::GetPageSize();
          byte* page = (byte*)mem;
    const byte* end  = page + size;

    while( page < end )
    {
        *page = 0;
        page += pageSize;
    }
}


//-----------------------------------------------------------
static const char* USAGE = R"(
iotest [ARGUMENTS] <test_dir>

 -s, --size <size>  : Size to write. Default is 4GB.
                      Ex: 512MB 1GB 16GB

 -d, --no-direct-io : Disable direct IO, which enables OS IO buffering.

 -m, --memory <size>: Reserve memory size to use as IO cache for the file.
 
 -h, --help         : Print this help message and exit.
)";

//-----------------------------------------------------------
void IOTestPrintUsage()
{
    Log::Line( USAGE );
}


