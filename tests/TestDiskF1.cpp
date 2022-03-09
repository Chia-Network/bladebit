#include "TestUtil.h"
#include "plotdisk/DiskF1.h"
#include "util/jobs/MemJobs.h"
#include "plotting/PlotTools.h"
#include "util/StackAllocator.h"

#define WORK_TMP_PATH "/mnt/p5510a/disk_tmp/"

//-----------------------------------------------------------
TEST_CASE( "F1Disk", "[f1]" )
{
    SysHost::InstallCrashHandler();
    const uint32 numBuckets = 128;

    DiskPlotContext context = { 0 };

    context.numBuckets    = numBuckets;
    context.cacheSize     = 64ull GB;
    context.cache         = bbvirtallocbounded<byte>( context.cacheSize );
    context.f1ThreadCount = 32;
    context.heapSize      = 8ull GB;
    context.heapBuffer    = bbvirtallocbounded<byte>( context.heapSize );

    ThreadPool pool( SysHost::GetLogicalCPUCount() );
    DiskBufferQueue ioQueue( WORK_TMP_PATH, context.heapBuffer, context.heapSize, 1, false );
    
    {
        FileSetInitData fdata = {
            .cache     = context.cache,
            .cacheSize = context.cacheSize
        };
        ioQueue.InitFileSet( FileId::FX0, "fx", context.numBuckets,
                             FileSetOptions::Cachable | FileSetOptions::DirectIO, &fdata );
    }
    
    const size_t fsBlockSize = ioQueue.BlockSize( FileId::FX0 );
    context.t1FsBlocks    = bbvirtallocbounded<byte>( fsBlockSize * numBuckets );

    context.threadPool = &pool;
    context.ioQueue    = &ioQueue;

    Log::Line( "Initializing memory" );
    FaultMemoryPages::RunJob( pool, pool.ThreadCount(), context.cache     , context.cacheSize );
    FaultMemoryPages::RunJob( pool, pool.ThreadCount(), context.heapBuffer, context.heapSize  );
    FaultMemoryPages::RunJob( pool, pool.ThreadCount(), context.t1FsBlocks, fsBlockSize * numBuckets );

    Log::Line( "Testing F1" );
    const char plotIdHex[] = "7a709594087cca18cffa37be61bdecf9b6b465de91acb06ecb6dbe0f4a536f73";
    byte plotId[BB_PLOT_ID_LEN];
    HexStrToBytes( plotIdHex, sizeof( plotIdHex )-1, plotId, sizeof( plotId ) );
    context.plotId = plotId;
    

    Log::Line( "Heap  Size: %.2lf GiB", (double)context.heapSize BtoGB );
    Log::Line( "Cache Size: %.2lf GiB", (double)context.cacheSize BtoGB );
    Log::Line( "Generating f1 with %u buckets and %u threads", context.numBuckets, context.f1ThreadCount );
    const auto timer = TimerBegin();
    DiskF1<numBuckets> f1( context, FileId::FX0 );
    f1.GenF1();

    auto elapsed = TimerEnd( timer );
    Log::Line( "Finished f1 in %.2lf seconds. IO wait time: %.2lf seconds.",
        elapsed, ioQueue.IOBufferWaitTime() );
}

