#pragma once
#include "DiskPlotContext.h"
#include "util/Log.h"
#include "ChiaConsts.h"


struct GroupInfo
{
    // When scanning groups
    uint32* groupBoundaries;
    uint32  groupCount;
    uint32  startIndex;

    // When matching
    uint32  entryCount;
    Pairs   pairs;
};

class DiskPlotPhase1
{   
public:
    DiskPlotPhase1( DiskPlotContext& cx );
    void Run();

private:
    void GenF1();
    
    template <uint32 _numBuckets>
    void GenF1Buckets();

    // Run forward propagations portion
    void ForwardPropagate();

    template<TableId table>
    void ForwardPropagateTable();

    template<TableId table, uint32 _numBuckets>
    void ForwardPropagateBuckets();

    // Write C tables
    void SortAndCompressTable7();

private:
    DiskPlotContext& _cx;
    DiskBufferQueue* _diskQueue;
};


struct ScanGroupJob : MTJob<ScanGroupJob>
{
    const uint* yBuffer;
    uint*       groupBoundaries;
    uint        bucketIdx;
    uint        startIndex;
    uint        endIndex;
    uint        maxGroups;
    uint        groupCount;

    void Run() override;
};

struct MatchJob : MTJob<MatchJob>
{
    const uint32* yBuffer;
    GroupInfo*    groupInfo;
    uint          bucketIdx;
    uint          maxPairCount;

    // Final destination contiguous pair buffer
    uint32*       copyLDst;
    uint16*       copyRDst;

    // Fence ensures the copy destination buffer is not currently in-use
    Fence*        copyFence;

    void Run() override;
};
