#include "TestUtil.h"
#include "plotdisk/DiskPlotter.h"
#include "plotdisk/DiskF1.h"
#include "plotdisk/DiskFp.h"
#include "util/jobs/MemJobs.h"
#include "plotdisk/jobs/IOJob.h"
#include "plotting/PlotTools.h"
#include "util/StackAllocator.h"
#include "algorithm/RadixSort.h"

#define WORK_TMP_PATH "/mnt/p5510a/disk_tmp/"

template<uint32 numBuckets>
void LoadF1Buckets( DiskPlotContext& context, uint64* yEntries, const uint64* referenceY, const uint64 refCount );

//-----------------------------------------------------------
TEST_CASE( "F1Disk", "[f1]" )
{
    SysHost::InstallCrashHandler();

    const uint32 numBuckets = 128;
    DiskPlotContext context = { 0 };

    // Allocate based on 128 buckets, which has the largest allocation size
    const size_t heapSize = DiskPlotter::GetRequiredSizeForBuckets( 128, WORK_TMP_PATH, WORK_TMP_PATH );

    context.numBuckets    = numBuckets;
    context.heapSize      = heapSize;
    context.heapBuffer    = bbvirtallocbounded<byte>( context.heapSize );
    context.cacheSize     = 64ull GB;
    context.cache         = bbvirtallocbounded<byte>( context.cacheSize );
    context.f1ThreadCount = 32;

    ThreadPool pool( SysHost::GetLogicalCPUCount() );
    context.threadPool    = &pool;
    

    Log::Line( "Initializing memory" );
    FaultMemoryPages::RunJob( pool, pool.ThreadCount(), context.cache     , context.cacheSize );
    FaultMemoryPages::RunJob( pool, pool.ThreadCount(), context.heapBuffer, context.heapSize  );

    Log::Line( "Heap  Size: %.2lf GiB", (double)context.heapSize BtoGB );
    Log::Line( "Cache Size: %.2lf GiB", (double)context.cacheSize BtoGB );
    Log::Line( "Testing f1 with threads", context.f1ThreadCount );

    uint64* yEntries = nullptr;
    const uint64* referenceY = nullptr;
    uint64 refCount = 1ull << _K;

    // Run F1
    // const char plotIdHex[] = "7a709594087cca18cffa37be61bdecf9b6b465de91acb06ecb6dbe0f4a536f73";
    const char plotIdHex[] = "c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835";
    
    byte plotId[BB_PLOT_ID_LEN];
    HexStrToBytes( plotIdHex, sizeof( plotIdHex )-1, plotId, sizeof( plotId ) );
    context.plotId = plotId;

    const uint32 buckets[] = { 128, 256, 512, 1024 };
    for( uint32 i = 0; i < sizeof( buckets ) / sizeof( buckets[0] ); i++ )
    {
        uint32 b = buckets[i];
        context.numBuckets = b;

        Log::Line( "[Buckets %u]", b );
        Log::Line( " Generating f1" );
        

        DiskBufferQueue* ioQueue = new DiskBufferQueue( WORK_TMP_PATH, context.heapBuffer, context.heapSize, 1, false );
        context.ioQueue = ioQueue;

        {
            FileSetInitData fdata = {
                .cache     = context.cache,
                .cacheSize = context.cacheSize
            };

            ioQueue->InitFileSet( FileId::FX0, "fx", b,
                                  FileSetOptions::Cachable | FileSetOptions::DirectIO, &fdata );

            context.tmp1BlockSize = ioQueue->BlockSize( FileId::FX0 );
            context.tmp2BlockSize = ioQueue->BlockSize( FileId::FX0 );
        }

        const auto timer = TimerBegin();
        
        switch( b )
        {
            case 128 : { DiskF1<128 > f1( context, FileId::FX0 ); f1.GenF1(); } break;
            case 256 : { DiskF1<256 > f1( context, FileId::FX0 ); f1.GenF1(); } break;
            case 512 : { DiskF1<512 > f1( context, FileId::FX0 ); f1.GenF1(); } break;
            case 1024: { DiskF1<1024> f1( context, FileId::FX0 ); f1.GenF1(); } break;
        
            default:
                ENSURE( 0 );
                break;
        }
        
        auto elapsed = TimerEnd( timer );
        Log::Line( "Finished f1 in %.2lf seconds. IO wait time: %.2lf seconds.",
            elapsed, ioQueue->IOBufferWaitTime() );

        if( referenceY == nullptr )
        {
            Log::Line( "Loading reference Y table." );
            referenceY = LoadReferenceTable<uint64>( "/mnt/p5510a/reference/t1.y.tmp", refCount );
            ASSERT( refCount == 1ull << _K );

            yEntries = bbcvirtallocbounded<uint64>( refCount );
        }

        // Ensure our y values are correct
        switch( b )
        {
            case 128 : LoadF1Buckets<128 >( context, yEntries, referenceY, refCount ); break;
            case 256 : LoadF1Buckets<256 >( context, yEntries, referenceY, refCount ); break;
            case 512 : LoadF1Buckets<512 >( context, yEntries, referenceY, refCount ); break;
            case 1024: LoadF1Buckets<1024>( context, yEntries, referenceY, refCount ); break;
        
            default:
                ENSURE( 0 );
                break;
        }

        Log::Line( " Success" );

        // Done
        memset( context.bucketCounts[0], 0, sizeof( context.bucketCounts[0] ) );
    }
}

//-----------------------------------------------------------
template<uint32 numBuckets>
void LoadF1Buckets( DiskPlotContext& context, uint64* yEntries, const uint64* referenceY, const uint64 refCount )
{
    Log::Line( " Loading our y buckets" );
    
    const uint64 maxEntriesPerBucket = (uint64)( ( (1ull << _K ) / numBuckets ) * BB_DP_XTRA_ENTRIES_PER_BUCKET );

    uint64* yBuckets[2];
    yBuckets[0] = bbcvirtallocbounded<uint64>( maxEntriesPerBucket );
    yBuckets[1] = bbcvirtallocbounded<uint64>( maxEntriesPerBucket );

    using Info = DiskPlotInfo<TableId::Table1, numBuckets>;
    
    
    const uint32 yBits     = Info::YBitSize;
    const uint64 yMask     = ( 1ull << yBits ) - 1;
    const size_t entrySize = Info::EntrySizePackedBits;

    Fence fence;
    uint64* yReader = yEntries;

    uint64 totalEntries = 0;
    for( uint64 bucket = 0; bucket < numBuckets; bucket++ )
        totalEntries += context.bucketCounts[0][bucket];

    ASSERT( totalEntries = refCount )
    
    ThreadPool&      pool    = *context.threadPool;
    DiskBufferQueue& ioQueue = *context.ioQueue;
    ioQueue.SeekBucket( FileId::FX0, 0, SeekOrigin::Begin );

    auto LoadBucket = [&]( const uint32 bucket ) {

        const uint64 bucketEntries = context.bucketCounts[0][bucket];
        const size_t readBits      = entrySize * bucketEntries;
        const size_t readBytes     = RoundUpToNextBoundary( readBits / 8, sizeof( uint64 ) );

        uint64* buffer = yBuckets[bucket % 2];

        ioQueue.ReadFile( FileId::FX0, bucket, buffer, readBytes );
        ioQueue.SignalFence( fence, bucket + 1 );
        ioQueue.CommitCommands();
    };

    LoadBucket( 0 );

    for( uint64 bucket = 0; bucket < numBuckets; bucket++ )
    {
        Log::Line( "  Validating %u", bucket );
        
        if( bucket + 1 < numBuckets )
            LoadBucket( bucket + 1 );

        const uint64 bucketEntries = context.bucketCounts[0][bucket];
        const size_t readBits      = entrySize * bucketEntries;

        fence.Wait( bucket+1 );
        uint64* yBucket =  yBuckets[bucket % 2];

        // Unpack values and add bucket part
        const uint64 bucketMask = bucket << yBits;
        
        AnonMTJob::Run( pool, [=]( AnonMTJob* self ) {
            
            uint64 count, offset, end;
            GetThreadOffsets( self, bucketEntries, count, offset, end );
            
            BitReader reader( yBucket, readBits, offset * entrySize );
            for( uint64 i = offset; i < end; i++ )
                yReader[i] = bucketMask | ( reader.ReadBits64( entrySize ) & yMask );
        });
        // BitReader reader(  yBucket, readBits );
        // for( uint64 i = 0; i < bucketEntries; i++ )
        //     yReader[i] = bucketMask | ( reader.ReadBits64( entrySize ) & yMask );

        // const uint64 mask = ( 1ull << ( _K + kExtraBits ) ) - 1;
        // for( uint64 i = 0; i < bucketEntries; i++ )
        // {
        //     yReader[i] = reader.ReadBits64( entrySize ) & mask;
        //     ASSERT( yReader[i] != 0 );
        // }

        // Sort Y
        RadixSort256::Sort<BB_DP_MAX_JOBS, uint64, 5>( pool, yReader, yBucket, bucketEntries );

    //  Log::Line( " Validating entries" );
        for( uint64 i = 0; i < bucketEntries; i++ )
        {
            ENSURE( referenceY[i] == yReader[i] );
        }

        yReader += bucketEntries;
        referenceY += bucketEntries;
    }

    bbvirtfreebounded( yBuckets[0] );
    bbvirtfreebounded( yBuckets[1] );
    
}