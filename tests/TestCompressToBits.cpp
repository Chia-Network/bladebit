#include "catch2/catch_test_macros.hpp"
#include "pos/chacha8.h"
#include "util/Util.h"
#include "util/Log.h"
#include "plotting/PlotTools.h"
#include "plotdisk/DiskPlotConfig.h"
#include "util/BitView.h"
#include "util/jobs/MemJobs.h"
#include "threading/ThreadPool.h"


static void SerializeF1Entries( const uint64* entries, uint64* bits   , const int64 entryCount, const uint64 entryOffset, const size_t bitsPerEntry );
static void DerializeF1Entries( const uint64* bits,    uint64* entries, const int64 entryCount, const uint64 entryOffset, const size_t bitsPerEntry );


struct F1PackEntryJob : MTJob<F1PackEntryJob>
{
    const uint64* input;
    uint64*       output;
    int64         entryCount;
    uint64        offset;
    size_t        bitsPerEntry;
    bool          unpack;

    static void Serialize( ThreadPool& pool, uint32 threadCount, 
        const uint64* input,
        uint64*       output,
        int64         entryCount,
        size_t        bitsPerEntry );

    static void Deserialize( ThreadPool& pool, uint32 threadCount, 
        const uint64* input,
        uint64*       output,
        int64         entryCount,
        size_t        bitsPerEntry );

    void Run() override;

private:
    static void DoJob( ThreadPool& pool, uint32 threadCount, 
        const uint64* input,
        uint64*       output,
        int64         entryCount,
        size_t        bitsPerEntry,
        bool          unpack );
};

void TestF1WithBuckets( uint32 numBuckets, ThreadPool& pool, uint32 threadCount );

//-----------------------------------------------------------
TEST_CASE( "F1CompressToBits", "[bit_serialization]" )
{
    const uint32 threadCount = SysHost::GetLogicalCPUCount();
    ThreadPool pool( threadCount );

    for( uint32 i = 64; i <= 1024; i <<= 1 )
    {
        Log::Line( "[%-3u Buckets]", i );
        TestF1WithBuckets( i, pool, threadCount );
        Log::Line( "" );
    }
}


//-----------------------------------------------------------
void TestF1WithBuckets( uint32 numBuckets, ThreadPool& pool, uint32 threadCount )
{
    const char* plotIdHex = "7a709594087cca18cffa37be61bdecf9b6b465de91acb06ecb6dbe0f4a536f73";
    byte plotId[BB_PLOT_ID_LEN];
    HexStrToBytes( plotIdHex, sizeof( plotIdHex )-1, plotId, sizeof( plotId ) );

    const uint32 k = _K;
    static_assert( k == 32, "Only k32 supported for now." );
    const uint32 kMinusKExtraBits  = _K - kExtraBits;

    byte key[32] = { 1 };
    memcpy( key + 1, plotId, 31 );
    chacha8_ctx chacha;
    chacha8_keysetup( &chacha, key, 256, NULL );

    const uint64 blockCount       = ( 1ull << k ) / numBuckets;
    const uint64 entriesPerBlock  = kF1BlockSize / (k / 8ull);
    const int64  entryCount       = (int64)( entriesPerBlock * blockCount );
    const uint32 bitsSaved        = (uint32)log2( numBuckets ) - kExtraBits;
    const size_t bitsPerEntry     = k * 2 - bitsSaved;
    const size_t writeBufferSize  = RoundUpToNextBoundaryT( (uint64)entryCount * bitsPerEntry, 64ull*2 ) / 8;

    Log::Line( " Entries %llu  Blocks: %llu", entryCount, blockCount );
    Log::Line( "Allocating..." );

    const size_t entriesAllocSize = (size_t)entryCount * sizeof( uint64 );
    uint32* blocks          = bbvirtalloc<uint32>( blockCount * kF1BlockSize );
    uint64* entries         = bbvirtalloc<uint64>( entriesAllocSize );
    uint64* packedEntries   = bbvirtalloc<uint64>( entriesAllocSize );
    uint64* unpackedEntries = bbvirtalloc<uint64>( entriesAllocSize );

    FaultMemoryPages::RunJob( pool, threadCount, entries        , entriesAllocSize );
    FaultMemoryPages::RunJob( pool, threadCount, packedEntries  , entriesAllocSize );
    FaultMemoryPages::RunJob( pool, threadCount, unpackedEntries, entriesAllocSize );

    Log::Line( "ChaCha..." );
    auto timer = TimerBegin();
    uint64 x = 0;
    chacha8_get_keystream( &chacha, x, (uint32)blockCount, (uint8_t*)blocks );
    double elapsed = TimerEnd( timer );
    Log::Line( "ChaCha generated %llu entries in %.2lf seconds.", entryCount, elapsed );

    // Prepare entries
    // uint32 bucketCounts[BB_DP_MAX_BUCKET_COUNT] = { 0 };

    // Don't care about buckets for now, just test bit compression time elapsed
    timer = TimerBegin();
    for( int64 i = 0; i < entryCount; i++ )
    {
        uint64 y  = Swap32( blocks[i] );
        uint64 xx = x + i;

        entries[i] = (y << kExtraBits) | ( xx >> kMinusKExtraBits );
    }
    elapsed = TimerEnd( timer );
    Log::Line( "Prepared %llu entries in %.2lf seconds.", entryCount, elapsed );

    // Now compress to bits
    timer = TimerBegin();
    // SerializeF1Entries( entries, packedEntries, entryCount, 0, bitsPerEntry );
    F1PackEntryJob::Serialize( pool, threadCount, entries, packedEntries, entryCount, bitsPerEntry );
    elapsed = TimerEnd( timer );
    Log::Line( "Serialized %llu entries in %.2lf seconds.", entryCount, elapsed );

    // Deserialize
    timer = TimerBegin();
    // DerializeF1Entries( packedEntries, unpackedEntries, entryCount, 0, bitsPerEntry );
    F1PackEntryJob::Deserialize( pool, threadCount, packedEntries, unpackedEntries, entryCount, bitsPerEntry );
    elapsed = TimerEnd( timer );
    Log::Line( "Deserialized %llu entries in %.2lf seconds.", entryCount, elapsed );

    Log::Write( "Checking equal... " );
    REQUIRE( memcmp( entries, unpackedEntries, sizeof( uint64 ) * entryCount ) == 0 );
    Log::Line( "OK!" );

    bbvirtfree( blocks          );
    bbvirtfree( entries         );
    bbvirtfree( packedEntries   );
    bbvirtfree( unpackedEntries );
}

//-----------------------------------------------------------
void SerializeF1Entries( const uint64* entries, uint64* bits, const int64 entryCount, const uint64 entryOffset, const size_t bitsPerEntry )
{
    const uint64 bitOffset = entryOffset * bitsPerEntry;
    entries += entryOffset;

    BitWriter writer( bits, (uint64)entryCount * bitsPerEntry, bitOffset );

    for( int64 i = 0; i < entryCount; i++ )
        writer.Write( entries[i], bitsPerEntry );
}

//-----------------------------------------------------------
void DerializeF1Entries( const uint64* bits, uint64* entries, const int64 entryCount, const uint64 entryOffset, const size_t bitsPerEntry )
{
    const uint64 bitOffset      = entryOffset * bitsPerEntry;
    // const uint64 fieldOffset    = bitOffset / 64;
    // const uint64 fieldBitOffset = bitOffset - fieldOffset * 64;
    // const size_t capacity       = RoundUpToNextBoundary( entryCount * bitsPerEntry, 64 );

    entries += entryOffset;
    // bits    += fieldOffset;

    // BitReader reader( bits, capacity, fieldBitOffset );
    BitReader reader( bits, entryCount * bitsPerEntry, bitOffset );

    for( int64 i = 0; i < entryCount; i++ )
        entries[i] = reader.ReadBits64( bitsPerEntry );
}

//-----------------------------------------------------------
void F1PackEntryJob::Run()
{
    ASSERT( entryCount >= 2 );

    if( this->unpack )
    {
        DerializeF1Entries( input, output, entryCount, offset, bitsPerEntry );
    }
    else
    {
        // 2 passes to ensure 2 threads don't write to the same field at the same time
        SerializeF1Entries( input, output, entryCount - 2, offset, bitsPerEntry );
        this->SyncThreads();
        SerializeF1Entries( input, output, 2, offset + entryCount - 2, bitsPerEntry );
    }
}


//-----------------------------------------------------------
void F1PackEntryJob::Serialize( ThreadPool& pool, uint32 threadCount, 
        const uint64* input,
        uint64*       output,
        int64         entryCount,
        size_t        bitsPerEntry )
{
    DoJob( pool, threadCount, input, output, entryCount, bitsPerEntry, false );
}

//-----------------------------------------------------------
void F1PackEntryJob::Deserialize( ThreadPool& pool, uint32 threadCount, 
        const uint64* input,
        uint64*       output,
        int64         entryCount,
        size_t        bitsPerEntry )
{
    DoJob( pool, threadCount, input, output, entryCount, bitsPerEntry, true );
}

//-----------------------------------------------------------
void F1PackEntryJob::DoJob( ThreadPool& pool, uint32 threadCount, 
        const uint64* input,
        uint64*       output,
        int64         entryCount,
        size_t        bitsPerEntry,
        bool          unpack )
{
    ASSERT( threadCount <= pool.ThreadCount() );
    MTJobRunner<F1PackEntryJob> jobs( pool );

    const uint64 entriesPerThread = entryCount / threadCount;
    ASSERT( entriesPerThread >= 2 );

    for( uint64 i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];
        job.input        = input;
        job.output       = output;
        job.entryCount   = entriesPerThread;
        job.offset       = i * entriesPerThread;
        job.bitsPerEntry = bitsPerEntry;
        job.unpack       = unpack;
    }

    const uint64 trailingEntries = entryCount - entriesPerThread * threadCount;
    jobs[threadCount-1].entryCount += trailingEntries;

    jobs.Run( threadCount );
}

