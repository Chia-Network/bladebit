#pragma once

#include "DiskPlotContext.h"
#include "plotdisk/BitBucketWriter.h"
#include "plotdisk/DiskPairReader.h"

class DiskPlotPhase3
{
public:
    DiskPlotPhase3( DiskPlotContext& context );
    ~DiskPlotPhase3();

    void Run();

    static size_t GetRequiredHeapSize( const uint32 numBuckets, const bool bounded, const size_t t1BlockSize, const size_t t2BlockSize );
private:
    
    template<uint32 _numBuckets>
    void RunBuckets();

    template<TableId rTable, uint32 _numBuckets, bool _bounded>
    void ProcessTable();

    template<TableId rTable>
    void ConvertToLinePoints( 
        const uint32 bucket, const int64 bucketLength, const uint32* leftEntries, 
        const void* rightMarkedEntries, const Pair* rightPairs, const uint64* rightMap );

    template<TableId rTable, uint32 _numBuckets>
    void WriteLinePointsToBuckets( const uint32 bucket,const int64 entryCount, const uint64* linePoints, const uint32* key );
    
    template<uint32 _numBuckets>
    void WritePark7( const uint64 inMapBucketCounts[_numBuckets+1] );

    // template<uint32 _numBuckets>
    // void WritePark7();

    // void TableFirstStep( const TableId rTable );
    // void BucketFirstStep( const TableId rTable, const uint32 bucket );

    // uint32 PointersToLinePoints( TableId rTable, const uint64 entryOffset,
    //                              const uint32 entryCount, const uint64* markedEntries, 
    //                              const uint32* lTable, const Pairs pairs,
    //                              const uint32* rMapIn, uint32* rMapOut,
    //                              uint64* outLinePoints );

    

    // void TableSecondStep( const TableId rTable );

    // void WriteLPReverseLookup( const TableId rTable, const uint32* key,
    //                            const uint32 bucket, const uint32 entryCount,
    //                            const uint64 entryOffset );

    // void WriteLinePointsToPark( TableId rTable, bool isLastBucket, const uint64* linePoints, uint32 bucketLength );


    // void TableThirdStep( const TableId rTable );

    // void WritePark7( uint32 bucket, uint32* t6Indices, uint32 entryCount );

    // void DeleteFile( FileId fileId, uint32 bucket );
    // void DeleteBucket( FileId fileId );

    void GetCacheSizes( size_t& outCacheSizeLP, size_t& outCacheSizeMap );

    template<uint32 _numBuckets>
    void GetCacheSizesForBuckets( size_t& outCacheSizeLP, size_t& outCacheSizeMap );
    
    template<uint32 _numBuckets, const bool _bounded>
    static size_t GetRequiredHeapSizeForBuckets( const size_t t1BlockSize, const size_t t2BlockSize );

private:
    DiskPlotContext& _context;
    DiskBufferQueue& _ioQueue;

    Fence _readFence;
    Fence _writeFence;
    Fence _stepFence;
    Fence _plotFence;

    Duration _ioWaitTime  = Duration::zero();
    
    FileId _mapReadId  = FileId::LP_MAP_0;
    FileId _mapWriteId = FileId::LP_MAP_1;

    // Read buffers
    // Pair*   _pairRead[2];
    // uint64* _rMapRead[2];
    // uint32* _lMapRead[2];

    // uint32*             _lMapBuffers[2];        // Only used for table 1. The rest use a map reader.
    // IP3LMapReader<uint32>* _lMap = nullptr;
    // uint32*                _rMap[2];

    // uint64  _lEntriesLoaded = 0;

    // Unpacked working buffers
    // Pair*   _rPrunedPairs;      // Temporary buffer for storing the pruned pairs and map
    // uint64* _rPrunedMap;                
    // uint64* _linePoints;        // Used to convert to line points/tmp buffer

    // uint64  _rTableOffset;              // 
    
    // BitBucketWriter<LP_BUCKET_COUNT> _lpWriter;
    // BitBucketWriter<LP_BUCKET_COUNT> _mapWriter;
    // uint64*                          _lpWriteBuffer [2];
    // uint32*                          _mapWriteBuffer[2];


    // Entry count for the current R table after it has been pruned
    // uint64  _prunedEntryCount;

    uint64  _tablePrunedEntryCount [7] = { 0 };                   // Count of each table, after prunning
    uint64  _lpPrunedBucketCounts  [BB_DP_MAX_BUCKET_COUNT+1];    // Entry count per bucket of our current R table after it has been converted to buckets.
    uint64  _lMapPrunedBucketCounts[BB_DP_MAX_BUCKET_COUNT+1];    // Entry count per bucket for the lMap table after it has been pruned by the previous table compression.

    // // Left over entries in a bucket (which is not the last bucket)
    // // that did not fit into a full park, these need to be included in
    // // the first park of the next bucket
    // uint64  _bucketParkLeftOvers[kEntriesPerPark] = { 0 };
    // uint32  _bucketParkLeftOversCount = 0;

    // uint32  _park7LeftOvers[kEntriesPerPark] = { 0 };
    // uint32  _park7LeftOversCount = 0;
};
