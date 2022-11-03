#pragma once

#include "plotdisk/DiskPlotContext.h"
#include "plotdisk/DiskPairReader.h"
#include "util/BitField.h"

class DiskPlotPhase2
{
public:
    DiskPlotPhase2( DiskPlotContext& context );
    ~DiskPlotPhase2();

    void Run();

private:
    template<uint32 _numBuckets, bool _bounded>
    void RunWithBuckets();

    template<uint32 _numBuckets, bool _bounded>
    void    MarkTable( const TableId rTable, DiskPairAndMapReader<_numBuckets, _bounded> reader, Pair* pairs, uint64* map, BitField lTableMarks, const BitField rTableMarks );

    template<TableId table, uint32 _numBuckets, bool _bounded>
    void    MarkTableBuckets( DiskPairAndMapReader<_numBuckets, _bounded> reader, Pair* pairs, uint64* map, BitField lTableMarks, const BitField rTableMarks );

private:
    DiskPlotContext& _context;
    Fence*           _bucketReadFence;
    Fence*           _mapWriteFence;
    size_t           _markingTableSize = 0;
    Duration         _ioTableWaitTime = Duration::zero();
};