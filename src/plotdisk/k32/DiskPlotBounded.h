#pragma once
#include "plotdisk/DiskPlotContext.h"
#include "util/StackAllocator.h"

struct K32CrossBucketEntries;

// Bounded k32 disk plotter
class K32BoundedPhase1
{
public:
    K32BoundedPhase1( DiskPlotContext& context );
    ~K32BoundedPhase1();

    void Run();

    static size_t GetRequiredSize( const uint32 numBuckets, const size_t t1BlockSize, const size_t t2BlockSize, const uint32 threadCount );

private:

    template<uint32 _numBuckets>
    void RunWithBuckets();

    template<uint32 _numBuckets>
    void RunF1();

    template<TableId table, uint32 _numBuckets>
    void RunFx();

private:
    DiskPlotContext& _context;
    DiskBufferQueue& _ioQueue;

    StackAllocator _allocator;

#if BB_DP_FP_MATCH_X_BUCKET
    size_t                      _xBucketStackMarker = 0;
    Span<K32CrossBucketEntries> _crossBucketEntries[2];
#endif

};

