#pragma once

#include "DiskPlotContext.h"

class DiskPlotPhase3
{
public:
    DiskPlotPhase3( DiskPlotContext& context, const Phase3Data& phase3Data );
    ~DiskPlotPhase3();

    void Run();

private:
    void ProcessTable( const TableId rTable );

    void TableFirstStep( const TableId rTable );
    void BucketFirstStep( const TableId rTable, const uint32 bucket );

    void DistributeLinePoints( const TableId rTable, const uint32 entryCount, const uint64* linePoints, const uint32* rMap );

    void ProcessLTableMap( const uint32 bucket, const uint32 entryCount, const uint64* lMap, uint32* outSortedLMap );

    uint32 PointersToLinePoints( const uint32 entryCount, const uint64* markedEntries, 
                                 const uint32* lTable, const Pairs pairs,
                                 const uint32* rMapIn, uint32* rMapOut,
                                 uint64* outLinePoints );

    void LoadLMapBuckets(  const TableId rTable, const uint32 buckets );

    // void TableSecondStep( const TableId rTable );

private:
    DiskPlotContext& _context;
    
    Phase3Data _phase3Data;

    uint64* _markedEntries;             // Right table marked entries buffer
    uint64* _lMap       [2];
    Pairs   _rTablePairs[2];
    uint64* _rMap       [2];
    uint64* _tmpLMap;                   // Temporary L map buffer for stripping and sorting.
    uint64* _linePoints;                // Used to convert to line points/tmp buffer
    
    Fence   _rTableFence;
    Fence   _lTableFence;

    uint32  _lMapBucketsLoaded  = 0;

    uint32  _lYEntriesLoaded = 0;       // Keeps track of the "y" buckets, that is the original bucket lengths, based on Y sorted values
                                        // which generated tables in Phase 1. This is disjointed from the bucket lengths in the L map since those
                                        // are evenly distributed (all the buckets have the same length, except potentially the last one).

    uint64  _rTableOffset       = 0;
    uint32  _rTableBucket       = 0;

    uint64  _tableEntryCount[7];        // Count of each table, after prunning
};
