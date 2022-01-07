#pragma once

#include "DiskPlotContext.h"

class DiskPlotPhase2
{
public:
    DiskPlotPhase2( DiskPlotContext& context );
    ~DiskPlotPhase2();

    void Run();

    void    MarkTable( TableId table, uint64* lTableMarks, uint64* rTableMarks );
    void    LoadNextBuckets( TableId table, uint32 bucket, uint64*& mapBuffer, Pairs& pairsBuffer );
    uint32* SortAndStripMap( uint64* map, uint32 entryCount );

private:
    DiskPlotContext& _context;

    // uint64* _markingBuffers[2];
    uint64* _tmpMap;
    Fence*  _bucketReadFence;
    Fence*  _mapWriteFence;

    byte*   _bucketBuffers[BB_DP_BUCKET_COUNT]; 
    // byte*  _pairsBuffers[BB_DP_BUCKET_COUNT]; 

    uint32 _bucketsLoaded;
};