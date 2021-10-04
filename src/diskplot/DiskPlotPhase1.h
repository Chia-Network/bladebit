#pragma once
#include "DiskPlotContext.h"
#include "plotshared/MTJob.h"
#include "DiskBufferQueue.h"

struct GenF1Job : MTJob<GenF1Job>
{
    const byte* key;
    
    byte*   buffer;;
    uint32  blockCount;
    uint32  chunkCount;
    uint32  x;
    uint32* counts;
    
    DiskBufferQueue* diskQueue;

    void Run() override;
};

class DiskPlotPhase1
{
public:
    DiskPlotPhase1( DiskPlotContext& cx );
    void Run();

private:
    void GenF1();

private:
    DiskPlotContext& _cx;
    DiskBufferQueue  _diskQueue;
};

