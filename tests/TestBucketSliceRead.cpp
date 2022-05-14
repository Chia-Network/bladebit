#include "TestUtil.h"
#include "io/BucketStream.h"
#include "io/HybridStream.h"
#include "util/Util.h"
#include "threading/MTJob.h"
#include "pos/chacha8.h"
#include "ChiaConsts.h"
#include "plotdisk/DiskBufferQueue.h"

const uint32 k          = 24;
const uint32 entryCount = 1ull << k;

Span<uint32> entriesSrc;
Span<uint32> entriesWrite;
Span<uint32> entriesRead ;

const FileId     fileId  = FileId::FX0;
DiskBufferQueue* ioQueue = nullptr;
Fence fence;


//-----------------------------------------------------------
template<uint32 _numBuckets>
void RunForBuckets( ThreadPool& pool, const uint32 threadCount )
{
    Log::Line( "Testing buckets: %u", _numBuckets );

    fence.Reset( 0 );

    using Job = AnonPrefixSumJob<uint32>;
    Job::Run( pool, threadCount, [=]( Job* self ) {

        const uint32 entriesPerBucket = entryCount / _numBuckets;
        const uint32 bucketBits       = bblog2( _numBuckets );
        const uint32 bucketShift      = k - bucketBits;

        uint32 count, offset, end;
        GetThreadOffsets( self, entriesPerBucket, count, offset, end );

        // Distribute to buckets
        for( uint32 i = offset; end < entriesPerBucket; end++ )
        {
            const uint32 v = entriesSrc[i];
            entriesWrite[v >> bucketShift] = v;
        }

        uint32 counts            [_numBuckets] = {};
        uint32 pfxSum            [_numBuckets];
        uint32 totalCounts       [_numBuckets];
        uint32 offsets           [_numBuckets] = {};
        uint32 alignedWriteCounts[_numBuckets];

        memcpy( offsets, _offsets, sizeof( offsets ) );

        self->CalculateBlockAlignedPrefixSum( _numBuckets, fileId, counts, pfxSum, totalCounts, offsets, alignedWriteCounts );

        // Write to disk
        if( self->BeginLockBlock() )
        {
            ioQueue->WriteBucketElements( fileId, entriesWrite.Ptr(), sizeof( uint32 ), alignedWriteCounts );
            ioQueue->SeekBucket( fileId, 0, SeekOrigin::Begin );
            ioQueue->CommitCommands();
            ioQueue->SignalFence( fence );

            // Wait for the writes to finish
            fence.Wait();

            // Read from disk

        }
        self->EndLockBlock();

        // Save offsets for the next run?
        // memcpy( _offsets, offsets, sizeof( offsets ) );
    });
}

//-----------------------------------------------------------
TEST_CASE( "bucket-slice-write", "[unit-core]" )
{
    SysHost::InstallCrashHandler();

    const uint32 entriesPerBucket = entryCount / 64;

    entriesSrc   = bbcvirtallocboundednuma_span<uint32>( entriesPerBucket );
    entriesWrite = bbcvirtallocboundednuma_span<uint32>( entriesPerBucket );
    entriesRead  = bbcvirtallocboundednuma_span<uint32>( entriesPerBucket );

    // for( uint32 i = 0; i < entriesPerBucket; i++ )
    //     entriesTmp[i] = i;

    byte key[32] = { 1 };
    {
        auto bytes = HexStringToBytes( "c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835" );
        memcpy( key+1, bytes.data() + 1, 31 );
    }

    Log::Line( "ChaCha gen..." );
    const uint32 blockCount = ( entriesPerBucket * sizeof( uint32 ) ) / kF1BlockSize;

    chacha8_ctx chacha;
    chacha8_keysetup( &chacha, key, 256, nullptr );
    chacha8_get_keystream( &chacha, 0, blockCount, (byte*)entriesSrc.Ptr() );

    const uint32 maxThreads = SysHost::GetLogicalCPUCount();
    ThreadPool pool( maxThreads );

    RunForBuckets<64>( pool, maxThreads );
}