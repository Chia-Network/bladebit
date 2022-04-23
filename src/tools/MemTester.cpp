#include "threading/ThreadPool.h"
#include "threading/MTJob.h"
#include "plotting/GlobalPlotConfig.h"
#include "util/Util.h"
#include "util/CliParser.h"
#include "util/Log.h"
#include "util/jobs/MemJobs.h"

void MemTestPrintUsage();

//-----------------------------------------------------------
void MemTestMain( GlobalPlotConfig& gCfg, CliParser& cli )
{
    size_t memSize  = 16ull MB;
    uint32 passCount = 1;

    while( cli.HasArgs() )
    {
        if( cli.ReadSize( memSize, "-s", "--size" ) )
        {
            FatalIf( memSize < 1, "Memory size must be > 0." );
        }
        else if( cli.ReadU32( passCount, "-p", "--passes" ) )
        {
            if( passCount < 1 ) passCount = 1;
            continue;
        }
        else if( cli.ArgConsume( "-h", "--help" ) )
        {
            MemTestPrintUsage();
            exit( 0 );
        }
        else
        {
            Fatal( "Unexpected argument '%s'.", cli.Arg() );
        }
    }

    const uint32 maxThreads  = SysHost::GetLogicalCPUCount();
    const uint32 threadCount = gCfg.threadCount == 0 ? 1 : std::min( gCfg.threadCount, maxThreads );

    ThreadPool pool( threadCount, ThreadPool::Mode::Fixed, gCfg.disableCpuAffinity );

    Log::Line( "Size   : %.2lf MiB", (double)memSize BtoMB );
    Log::Line( "Threads: %u",        threadCount );
    Log::Line( "Passes : %u",        passCount   );

    Log::Line( "Allocating buffer..." );
    byte* src = bbvirtalloc<byte>( memSize );
    byte* dst = bbvirtalloc<byte>( memSize );

    FaultMemoryPages::RunJob( pool, threadCount, src, memSize );
    FaultMemoryPages::RunJob( pool, threadCount, dst, memSize );

    const double sizeMB = (double)memSize BtoMB;

    Log::Line( "Starting Test" );
    Log::Line( "" );
    for( uint32 pass = 0; pass < passCount; pass++ )
    {
        if( passCount > 1 )
        {
            Log::Line( "[Pass %u/%u]", pass+1, passCount );
        }

        auto timer = TimerBegin();
        MemCpyMT::Copy( dst, src, memSize, pool, threadCount );
        auto elapsed = TimerEnd( timer );

        Log::Line( "Copied %.2lf MiB in %.2lf seconds @ %.2lf MiB/s (%.2lf GiB/s) or %2.lf MB/s (%.2lf GB/s).", 
                    sizeMB, elapsed, memSize / elapsed BtoMB, memSize / elapsed BtoGB,
                    memSize / elapsed / 1000000.0, memSize / elapsed / 1000000000.0 );
        Log::Line( "" );
    }


    exit( 0 );
}

//-----------------------------------------------------------
static const char* USAGE = R"(memtest [OPTIONS]

Performs a memory copy operation.
Specify -t <n> in the global options to set the thread count.

[OPTIONS]
 -s, --size <size>  : Size of memory to copy.
                      Ex: 512MB 1GB 4GB

 -p, --passes <n>   : The number of passes to perform. By default it is 1.

 -h, --help         : Print this help message and exit.
)";

//-----------------------------------------------------------
void MemTestPrintUsage()
{
    Log::Line( USAGE );
}