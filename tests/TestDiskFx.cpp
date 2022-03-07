#include "TestUtil.h"
#include "plotdisk/DiskFp.h"

#define TMP_PATH "/mnt/p5510a/disk_tmp/"

template<TableId table>
struct TableTest
{
    //-----------------------------------------------------------
    inline void Run( ThreadPool& pool )
    {
        const uint32 numBuckets = 128;
        using Info = DiskPlotInfo<table,numBuckets>;
        using Fp   = DiskFp<table,numBuckets>;
        
        // const size_t bits = DiskFp<table,numBuckets>::EntryInSizePackedBits;
        const size_t bits           = Info::EntrySizePackedBits;
        const size_t bufferFullSize = 0;

        const int64 maxEntries = Info::MaxBucketEntries;
        Log::Line( "Max Bucket Entries: %lld", maxEntries );
        // Log::Line( "Bits: %llu", bits );
        // auto* ioQueue = new DiskBufferQueue( TMP_PATH, 

    }
};

//-----------------------------------------------------------
TEST_CASE( "FxDisk", "[fx]" )
{
    Log::Line( "Hello Fx." );
    ThreadPool pool( SysHost::GetLogicalCPUCount() );

    TableTest<TableId::Table2> test;
    test.Run( pool );
}

