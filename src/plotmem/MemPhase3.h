#pragma once
#include "PlotContext.h"

class MemPhase3
{
    friend class MemPlotter;
public:

    MemPhase3( MemPlotContext& context );

    void Run();

private:
    template<bool IsTable6>
    uint64 ProcessTable( uint32* lEntries, uint64* lpBuffer,
                         Pair* rTable, const uint64 rTableCount, 
                         const byte* markedEntries, TableId tableId );

private:
    MemPlotContext& _context;
};