#pragma once

#include "DiskPlotContext.h"

class DiskPlotPhase2
{
    struct PairAndMap
    {
        Pairs   pairs;
        uint64* map;
    };

public:
    DiskPlotPhase2( DiskPlotContext& context );
    ~DiskPlotPhase2();

    void Run();

    void    MarkTable( TableId table, uint64* lTableMarks, uint64* rTableMarks );
    void    LoadNextBuckets( TableId table, uint32 bucket, uint64*& mapBuffer, Pairs& pairsBuffer, uint32& outBucketEntryCount );
    uint32* SortAndStripMap( uint64* map, uint32 entryCount );

    inline const Phase3Data& GetPhase3Data() const { return _phase3Data; }

private:
    DiskPlotContext& _context;

    uint64* _tmpMap;
    Fence*  _bucketReadFence;
    Fence*  _mapWriteFence;

    PairAndMap  _bucketBuffers[BB_DP_BUCKET_COUNT]; 
    uint32      _bucketsLoaded;

    Phase3Data  _phase3Data;
};