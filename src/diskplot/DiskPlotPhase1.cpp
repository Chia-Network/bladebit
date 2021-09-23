#include "DiskPlotPhase1.h"
#include "pos/chacha8.h"

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
