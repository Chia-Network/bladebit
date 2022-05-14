#include "TestUtil.h"
#include "io/BucketStream.h"
#include "io/HybridStream.h"
#include "util/Util.h"
#include "threading/MTJob.h"
#include "pos/chacha8.h"
#include "ChiaConsts.h"
#include "plotdisk/DiskBufferQueue.h"
#include "algorithm/RadixSort.h"

const uint32 k          = 24;
const uint32 entryCount = 1ull << k;

Span<uint32> entriesRef;
Span<uint32> entriesTmp;
Span<uint32> entriesBuffer;
// Span<uint32> entriesRead ;

const char*      tmpDir  = "/home/harold/plot/tmp";
const FileId     fileId  = FileId::FX0;
DiskBufferQueue* ioQueue = nullptr;
Fence fence;

using Job = AnonPrefixSumJob<uint32>;

auto seed = HexStringToBytes( "c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835" );
        

//-----------------------------------------------------------
void GenerateRandomEntries( ThreadPool& pool )
{
    ASSERT( seed.size() == 32 );

    entriesRef = bbcvirtallocboundednuma_span<uint32>( entryCount );
    entriesTmp = bbcvirtallocboundednuma_span<uint32>( entryCount );

    Job::Run( pool, [=]( Job* self ){

        const uint32 entriesPerBlock = kF1BlockSize / sizeof( uint32 );
        const uint32 blockCount      = entryCount / entriesPerBlock;
        ASSERT( blockCount * entriesPerBlock == entryCount );
        
        uint32 count, offset, end;
        GetThreadOffsets( self, blockCount, count, offset, end );

        byte key[32] = { 1 };
        memcpy( key+1, seed.data()+1, 31 );

        chacha8_ctx chacha;
        chacha8_keysetup( &chacha, key, 256, nullptr );
        chacha8_get_keystream( &chacha, offset, count, (byte*)entriesRef.Ptr() + offset * kF1BlockSize );
        
        const uint32 mask = ( 1u << k ) - 1;
        offset *= entriesPerBlock;
        end    *= entriesPerBlock;
        for( uint32 i = offset; i < end; i++ )
            entriesRef[i] &= mask;
    });
}

//-----------------------------------------------------------
template<uint32 _numBuckets>
void RunForBuckets( ThreadPool& pool, const uint32 threadCount )
{
    Log::Line( "Testing buckets: %u", _numBuckets );
    
    // Fence* fence = &_fence;
    fence.Reset( 0 );
    fence.Signal();

    Job::Run( pool, threadCount, [=, &fence]( Job* self ) {

        const uint32 entriesPerBucket = entryCount / _numBuckets;
        const uint32 bucketBits       = bblog2( _numBuckets );
        const uint32 bucketShift      = k - bucketBits;
        const size_t blockSize        = ioQueue->BlockSize( fileId );
        
        Span<uint32> srcEntries = entriesRef.Slice();

        uint32 count, offset, end;
        GetThreadOffsets( self, entriesPerBucket, count, offset, end );

        uint32 pfxSum            [_numBuckets];
        uint32 totalCounts       [_numBuckets];
        uint32 offsets           [_numBuckets] = {};
        uint32 alignedWriteCounts[_numBuckets];

        for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
        {
            uint32 counts[_numBuckets] = {};

            // Count entries / bucket
            for( uint32 i = offset; i < end; i++ )
                counts[srcEntries[i] >> bucketShift]++;

            // Prefix sum
            self->CalculateBlockAlignedPrefixSum<uint32>( _numBuckets, blockSize, counts, pfxSum, totalCounts, offsets, alignedWriteCounts );

            // Distribute entries to buckets
            for( uint32 i = offset; i < end; i++ )
            {
                const uint32 e   = srcEntries[i];
                const uint32 dst = --pfxSum[e >> bucketShift];

                entriesBuffer[dst] = e;
            }

            // Write to disk
            if( self->BeginLockBlock() )
            {
                ioQueue->WriteBucketElements( fileId, entriesBuffer.Ptr(), sizeof( uint32 ), totalCounts );
                ioQueue->SignalFence( fence );
                ioQueue->CommitCommands();
                fence.Wait();
            }
            self->EndLockBlock();

            // Write next bucket slices
            srcEntries = srcEntries.Slice( entriesPerBucket );
        }
    });

    // Sort the reference entries
    RadixSort256::Sort<BB_DP_MAX_JOBS>( pool, entriesRef.Ptr(), entriesTmp.Ptr(), entriesRef.Length() );

    ioQueue->SeekBucket( fileId, 0, SeekOrigin::Begin );
    ioQueue->CommitCommands();
    fence.Reset();

    // Read back entries and validate them
    for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
    {
        auto readBuffer = entriesBuffer.Slice();

        ioQueue->ReadBucketElementsT( fileId, readBuffer );
        ioQueue->SignalFence( fence );
        ioQueue->CommitCommands();
        fence.Wait();

        // RadixSort256::Sort<BB_DP_MAX_JOBS>( pool, entriesRef.Ptr(), entriesTmp.Ptr(), entriesRef.Length() );
    }
}

//-----------------------------------------------------------
TEST_CASE( "bucket-slice-write", "[unit-core]" )
{
    SysHost::InstallCrashHandler();

    const uint32 maxThreads = SysHost::GetLogicalCPUCount();
    ThreadPool pool( maxThreads );

    const FileSetOptions flags = FileSetOptions::DirectIO | FileSetOptions::Cachable | FileSetOptions::Interleaved;
    FileSetInitData opts;
    opts.cacheSize = (size_t)( sizeof( uint32 ) * entryCount * 1.25);
    opts.cache     = bbvirtallocboundednuma( opts.cacheSize );

    byte dummyHeap = 1;
    ioQueue = new DiskBufferQueue( tmpDir, tmpDir, tmpDir, &dummyHeap, dummyHeap, 1 );
    ioQueue->InitFileSet( fileId, "test-slices", 64, flags, &opts );

    // Allocate buffers
    const uint32 minBuckets = 64;
    const uint32 blockSize              = (uint32)ioQueue->BlockSize( fileId );
    const uint32 entriesPerBucket       = entryCount / minBuckets;
    const uint32 maxEntriesPerSlice     = ((uint32)(entriesPerBucket / minBuckets) * BB_DP_ENTRY_SLICE_MULTIPLIER);
    const uint32 entriesPerSliceAligned  = RoundUpToNextBoundaryT( maxEntriesPerSlice, blockSize ) + blockSize / sizeof( uint32 ); // Need an extra block for when we offset the entries
    const uint32 entriesPerBucketAligned = entriesPerSliceAligned * minBuckets;                                                    // in subsequent slices

    entriesBuffer = bbcvirtallocboundednuma_span<uint32>( (size_t)( entriesPerBucketAligned ) );
    // entriesRead  = bbcvirtallocboundednuma_span<uint32>( (size_t)( entriesPerBucketAligned ) );

    Log::Line( "ChaCha gen..." );
    GenerateRandomEntries( pool );

    RunForBuckets<64>( pool, maxThreads );
}