#include "DiskPlotPhase1.h"
#include "pos/chacha8.h"
#include "Util.h"
#include "util/Log.h"

// Test
#include "io/FileStream.h"
#include "SysHost.h"


struct GenF1Job : MTJob
{
    const byte* key;

    uint32  blockCount;
    uint32  entryCount;
    uint32  x;
};

//-----------------------------------------------------------
DiskPlotPhase1::DiskPlotPhase1( DiskPlotContext& cx )
    : _cx( cx )
{

}

//-----------------------------------------------------------
void DiskPlotPhase1::Run()
{
    // Test file
    FileStream file;
    if( !file.Open( "E:/bbtest.data", FileMode::Create, FileAccess::Write, FileFlags::LargeFile | FileFlags::NoBuffering ) )
        Fatal( "Failed to open file." );

    const size_t blockSize = file.BlockSize();

    byte key[32] = {
        22, 24, 11, 3, 1, 15, 11, 6, 23, 22,
        22, 24, 11, 3, 1, 15, 11, 6, 23, 22,
        22, 24, 11, 3, 1, 15, 11, 6, 23, 22, 5, 28
    };

    const uint   chachaBlockSize = 64;

    const uint   k = 33;
    const uint64 entryCount = 1ull << k;
    const uint   blockCount = (uint)( entryCount / chachaBlockSize );
    
    SysHost::SetCurrentThreadAffinityCpuId( 0 );

    uint32* buffer;

    {
        Log::Line( "Allocating buffer..." );
        auto timer = TimerBegin();

        buffer = (uint32*)SysHost::VirtualAlloc( entryCount * sizeof( uint32 ), true );
        FatalIf( !buffer, "Failed to allocate buffer." );

        double elapsed = TimerEnd( timer );
        Log::Line( "Finished in %.2lf seconds.", elapsed );
    }

    {
        chacha8_ctx chacha;
        ZeroMem( &chacha );

        Log::Line( "Generating ChaCha..." );
        auto timer = TimerBegin();

        chacha8_keysetup( &chacha, key, 256, NULL );
        chacha8_get_keystream( &chacha, 0, blockCount, (byte*)buffer );

        double elapsed = TimerEnd( timer );
        Log::Line( "Finished in %.2lf seconds.", elapsed );
    }

    {
        Log::Line( "Started writing to file..." );

        const size_t sizeWrite = entryCount * sizeof( uint );
        const size_t blockSize = file.BlockSize();
        
        size_t blocksToWrite = sizeWrite / blockSize;

        auto timer = TimerBegin();

        do
        {
            ssize_t written = file.Write( buffer, blocksToWrite * blockSize );
            FatalIf( written <= 0, "Failed to write to file." );

            size_t blocksWritten = (size_t)written / blockSize;
            ASSERT( blocksWritten <= blocksToWrite );

            blocksToWrite -= blocksWritten;
        } while( blocksToWrite > 0 );
        
        double elapsed = TimerEnd( timer );

        Log::Line( "Finished in %.2lf seconds.", elapsed );
    }
}

//-----------------------------------------------------------
void DiskPlotPhase1::GenF1()
{
    DiskPlotContext& cx   = _cx;
    ThreadPool&      pool = *cx.threadPool;

    const uint threadCount = pool.ThreadCount();

    // Prepare ChaCha key
    byte key[32] = { 1 };
    memcpy( key + 1, cx.plotId, 31 );

    // Prepare jobs
    const size_t chaChaBlockSize = kF1BlockSizeBits / 8;

    const uint64 entryCount      = 1ull << _K;
    const uint32 blockCount      = (uint32)(entryCount / chaChaBlockSize);
    const uint32 blocksPerBucket = blockCount / BB_DP_BUCKET_COUNT;
    const uint32 blocksPerThread = blocksPerBucket / threadCount;

//     const uint64 entriesPerThread = entryCount / threadCount;


    GenF1Job jobs[BB_DP_MAX_JOBS];

    for( uint i = 0; i < threadCount; i++ )
    {
        GenF1Job& job = jobs[i];
        job.jobId      = i;
        job.jobCount   = threadCount;
        job.key        = key;
        job.blockCount = blocksPerThread;
    }

    pool.RunJob( GenF1Thread, jobs, threadCount );
}

//-----------------------------------------------------------
void DiskPlotPhase1::GenF1Thread( GenF1Job* job )
{

}
