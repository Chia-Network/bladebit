#pragma once
#include "DiskPlotContext.h"
#include "util/Log.h"
#include "ChiaConsts.h"

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
    void WriteCTables();

    template<uint32 _numBuckets>
    void WriteCTablesBuckets();

private:
    DiskPlotContext& _cx;
    DiskBufferQueue* _diskQueue;

    Duration         _tableIOWaitTime;

    FileId           _fxIn  = FileId::FX0;
    FileId           _fxOut = FileId::FX1;
};
