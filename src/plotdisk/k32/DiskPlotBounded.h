#pragma once
#include "plotdisk/DiskPlotContext.h"

// Bounded k32 disk plotter
class K32BoundedPhase1
{
public:
    K32BoundedPhase1( DiskPlotContext& context );
    ~K32BoundedPhase1();

    void Run();

    static size_t GetRequiredSize( const uint32 numBuckets, const size_t t1BlockSize, const size_t t2BlockSize );

private:

    template<uint32 _numBuckets>
    void RunWithBuckets();

    template<uint32 _numBuckets>
    void RunF1();

    template<uint32 _numBuckets>
    void RunFx( const TableId table );

private:
    DiskPlotContext& _context;
    DiskBufferQueue& _ioQueue;
};