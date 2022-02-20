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

    uint32 PointersToLinePoints( TableId rTable, const uint64 entryOffset,
                                 const uint32 entryCount, const uint64* markedEntries, 
                                 const uint32* lTable, const Pairs pairs,
                                 const uint32* rMapIn, uint32* rMapOut,
                                 uint64* outLinePoints );

    

    void TableSecondStep( const TableId rTable );

    void WriteLPReverseLookup( const TableId rTable, const uint32* key,
                               const uint32 bucket, const uint32 entryCount,
                               const uint64 entryOffset );

    void WriteLinePointsToPark( TableId rTable, bool isLastBucket, const uint64* linePoints, uint32 bucketLength );


    void TableThirdStep( const TableId rTable );

    void WritePark7( uint32 bucket, uint32* t6Indices, uint32 entryCount );

    void DeleteFile( FileId fileId, uint32 bucket );
    void DeleteBucket( FileId fileId );

private:
    DiskPlotContext& _context;
    
    Phase3Data _phase3Data;

    uint64* _markedEntries;             // Right table marked entries buffer
    uint32* _lMap       [2];
    Pairs   _rTablePairs[2];
    uint32* _rMap       [2];
    uint32* _rPrunedMap;                // Temporary buffer for storing the pruned R map
    uint64* _linePoints;                // Used to convert to line points/tmp buffer
    
    Fence   _readFence;

    uint64  _rTableOffset;              // 


    uint64  _tablePrunedEntryCount[7];  // Count of each table, after prunning

    // Entry count for the current R table after it has been pruned
    uint64  _prunedEntryCount;

    // Entry count for each bucket of our current R table after
    // it has been converted to line points.
    uint32  _lpBucketCounts  [BB_DPP3_LP_BUCKET_COUNT];
    uint32  _lMapBucketCounts[BB_DP_BUCKET_COUNT];

    // Left over entries in a bucket (which is not the last bucket)
    // that did not fit into a full park, these need to be included in
    // the first park of the next bucket
    uint64  _bucketParkLeftOvers[kEntriesPerPark] = { 0 };
    uint32  _bucketParkLeftOversCount = 0;

    uint32  _park7LeftOvers[kEntriesPerPark] = { 0 };
    uint32  _park7LeftOversCount = 0;
};
