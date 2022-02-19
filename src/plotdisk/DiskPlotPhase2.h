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
    uint32* UnpackMap( uint64* map, uint32 entryCount, const uint32 bucket );
    
    void    UnpackTableMap( TableId table );

    inline const Phase3Data& GetPhase3Data() const { return _phase3Data; }

private:
    DiskPlotContext& _context;

    uint32* _tmpMap;
    Fence*  _bucketReadFence;
    Fence*  _mapWriteFence;

    PairAndMap  _bucketBuffers[BB_DP_BUCKET_COUNT]; 
    uint32      _bucketsLoaded;

    Phase3Data  _phase3Data;
};