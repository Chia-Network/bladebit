#include "catch2/catch_test_macros.hpp"
#include "util/jobs/MemJobs.h"
#include "util/BitView.h"
#include "threading/ThreadPool.h"
#include "plotting/PlotTypes.h"
#include "plotting/PlotTools.h"
#include "plotting/PackEntries.h"
#include "plotdisk/DiskBufferQueue.h"
#include "plotdisk/jobs/F1GenBucketized.h"
#include "plotdisk/IOTransforms.h"
#include "util/Util.h"
#include "util/Log.h"

class F1PackTransform : public IIOTransform
{
public:
    size_t bitsPerEntry;

    //-----------------------------------------------------------
    inline void Write( TransformData& data ) override
    {
        const uint32  numBuckets   = data.numBuckets;
              uint32* bucketCounts = data.bucketSizes;
              uint64* output       = (uint64*)data.buffer;
        const uint64* input        = output;

        for( uint32 i = 0; i < numBuckets; i++ )
        {
            const uint64 entryCount = bucketCounts[i];
            *output = 0;    // Uncessesary
            
            EntryPacker::SerializeEntries( input, output, (int64)entryCount, 0, bitsPerEntry );

            const uint64 fieldAlignedCount = CDiv( entryCount * bitsPerEntry, 64 );

            output += fieldAlignedCount;
            input  += entryCount;
            ASSERT( output <= input );

            // Update the write size
            bucketCounts[i] = (uint32)( fieldAlignedCount * sizeof( uint64 ) );
        }
    }
};

TEST_CASE( "F1GenBucketized", "[f1]" )
{
    const uint32 k           = _K;
    const uint32 threadCount = 24;//SysHost::GetLogicalCPUCount();
    const size_t cacheSize   = 64ull GB;
    const size_t heapSize    = 2ull GB;
    const char*  tmpDir      = "/mnt/p5510a/disk_tmp/";

    ThreadPool pool( threadCount );
    
    void* cache = bbvirtalloc( cacheSize );
    void* heap  = bbvirtalloc( heapSize  );

    Log::Line( "Allocating..." );
    FaultMemoryPages::RunJob( pool, threadCount, cache, cacheSize );
    FaultMemoryPages::RunJob( pool, threadCount, heap , heapSize  );

    const FileId fileId = FileId::FX0;
  
    const char* plotIdHex = "7a709594087cca18cffa37be61bdecf9b6b465de91acb06ecb6dbe0f4a536f73";
    byte plotId[BB_PLOT_ID_LEN];
    HexStrToBytes( plotIdHex, sizeof( plotIdHex )-1, plotId, sizeof( plotId ) );

    uint32 bucketCounts[BB_DP_MAX_BUCKET_COUNT] = { 0 };

    const uint32 i = 256;
    // for( uint32 i = 64; i <= 1024; i <<= 1 )
    {
        Log::Line( "[%-3u Buckets]", i );

        
        F1PackTransform packTransform;
        packTransform.bitsPerEntry = F1GenBucketized::GetEntrySizeBits( i, k );

        // Let it leak, no destruction mechanism yet
        auto* queue = new DiskBufferQueue( tmpDir, (byte*)heap, heapSize, 1, false );
    
        {
            FileSetInitData data = {
                .cache     = cache,
                .cacheSize = cacheSize
            };
            queue->InitFileSet( fileId, "fx", i, FileSetOptions::Cachable, &data );
            queue->SetTransform( fileId, packTransform );
        }

        const auto timer = TimerBegin();
        F1GenBucketized::GenerateF1Disk( plotId, pool, threadCount, *queue, bucketCounts, i, fileId );
        const auto elapsed = TimerEnd( timer );
        Log::Line( " Generated f1 in %.2lf seconds. Wait time: %.2lf", elapsed, queue->IOBufferWaitTime() );
        Log::Line( "" );

        queue->DeleteBucket( fileId );
        queue->CommitCommands();
    }
}
