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



    // void TableSecondStep( const TableId rTable );

private:
    DiskPlotContext& _context;
    
    Phase3Data _phase3Data;

    uint64* _markedEntries;             // Right table marked entries buffer
    uint64* _lMap       [2];
    Pairs   _rTablePairs[2];
    uint64* _rMap       [2];
    uint64* _tmpRMap;                   // Temporary R map buffer for stripping and sorting.
    uint64* _linePoints;                // Used to convert to line points/tmp buffer
    
    Fence   _rTableFence;
    Fence   _lTableFence;

    uint64  _tableEntryCount[7];        // Count of each table, after prunning
};
