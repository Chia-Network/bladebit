#include "TestUtil.h"
#include "plotdisk/DiskF1.h"
#include "util/jobs/MemJobs.h"
#include "plotdisk/jobs/IOJob.h"
#include "plotting/PlotTools.h"
#include "util/StackAllocator.h"
#include "algorithm/RadixSort.h"

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
    // const char plotIdHex[] = "7a709594087cca18cffa37be61bdecf9b6b465de91acb06ecb6dbe0f4a536f73";
    const char plotIdHex[] = "c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835";
    
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

    // Ensure tese y values are correct
    ioQueue.SeekBucket( FileId::FX0, 0, SeekOrigin::Begin );

    Log::Line( "Loading reference Y table." );
    uint64 refCount = 1ull << _K;
    uint64* referenceY = LoadReferenceTable<uint64>( "/mnt/p5510a/reference/t1.y.tmp", refCount );
    ASSERT( refCount == 1ull << _K );

    Log::Line( "Loading our Y values" );
    uint64* yEntries = bbcvirtallocbounded<uint64>( refCount );
    {
        const uint64 maxEntriesPerBucket =  (uint64)( ( (1ull << _K ) / numBuckets ) * BB_DP_XTRA_ENTRIES_PER_BUCKET );

        uint64* yBucket = bbcvirtallocbounded<uint64>( maxEntriesPerBucket );

        const uint32 yBits     = 31;
        const uint64 yMask     = ( 1ull << yBits ) - 1;
        const size_t entrySize = _K + yBits;

        Fence fence;
        uint64* yReader = yEntries;        

        uint64 totalEntries = 0;
        for( uint64 bucket = 0; bucket < numBuckets; bucket++ )
            totalEntries += context.bucketCounts[0][bucket];

        ASSERT( totalEntries = refCount );

        for( uint64 bucket = 0; bucket < numBuckets; bucket++ )
        {
            Log::Line( " Bucket %u", bucket );
            const uint64 bucketEntries = context.bucketCounts[0][bucket];

            const size_t readBits  = entrySize * bucketEntries;
            const size_t readBytes = RoundUpToNextBoundary( readBits / 8, sizeof( uint64 ) );

            ioQueue.ReadFile( FileId::FX0, bucket, yBucket, readBytes );
            ioQueue.SignalFence( fence, bucket+1 );
            ioQueue.CommitCommands();

            fence.Wait( bucket+1 );
    
            // Unpack values and add bucket part
            // const uint64 bucketMask = bucket << yBits;
            const uint64 mask = ( 1ull << ( _K + kExtraBits ) ) - 1;
            BitReader reader(  yBucket, readBits );
            // for( uint64 i = 0; i < bucketEntries; i++ )
            //     yReader[i] = bucketMask | ( reader.ReadBits64( entrySize ) & yMask );
            for( uint64 i = 0; i < bucketEntries; i++ )
            {
                yReader[i] = reader.ReadBits64( entrySize ) & mask;
                ASSERT( yReader[i] != 0 );
            }

            // Sort Y
            RadixSort256::Sort<BB_DP_MAX_JOBS, uint64>( pool, yReader, yBucket, bucketEntries );

            yReader += bucketEntries;
        }

        bbvirtfreebounded( yBucket );
    }

    Log::Line( "Validating entries" );
    for( uint64 i = 0; i < refCount; i++ )
    {
        ASSERT( referenceY[i] == yEntries[i] );
    }

    Log::Line( "Success" );
}

