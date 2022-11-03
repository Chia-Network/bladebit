#pragma once
#include "PlotContext.h"


/**
 * Here we simply mark all tables's used entries,
 * without remapping them or anythnig.
 */ 
class MemPhase2
{
    friend class MemPlotter;
public:

    MemPhase2( MemPlotContext& context );

    void Run();


private:

    void ClearMarkingBuffers();

    template<bool HasRightTableMarkingBuffer>
    void MarkTable( const Pair* rightTable, uint64 rightEntryCount, const byte* rMarkedEntries, byte* lMarkingBuffer );

private:
    MemPlotContext& _context;
};