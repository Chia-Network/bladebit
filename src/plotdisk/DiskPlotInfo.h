#pragma once
#include "ChiaConsts.h"
#include "plotting/Tables.h"
#include "plotdisk/DiskPlotConfig.h"

template<TableId table>
struct FpEntry {};

// Unpacked, 64-bit-aligned 
template<> struct FpEntry<TableId::Table1> { uint64 ykey; };              // x, y
template<> struct FpEntry<TableId::Table2> { uint64 ykey; uint64 meta; }; // meta, key, y
template<> struct FpEntry<TableId::Table3> { uint64 ykey; Meta4  meta; };
template<> struct FpEntry<TableId::Table4> { uint64 ykey; Meta4  meta; };
template<> struct FpEntry<TableId::Table5> { uint64 ykey; Meta3  meta; }; // It should be 20, but we use an extra 4 to round up to 64 bits.
template<> struct FpEntry<TableId::Table6> { uint64 ykey; uint64 meta; };
template<> struct FpEntry<TableId::Table7> { uint64 ykey; };

typedef FpEntry<TableId::Table1> T1Entry;
typedef FpEntry<TableId::Table2> T2Entry;
typedef FpEntry<TableId::Table3> T3Entry;
typedef FpEntry<TableId::Table4> T4Entry;
typedef FpEntry<TableId::Table5> T5Entry;
typedef FpEntry<TableId::Table6> T6Entry;
typedef FpEntry<TableId::Table7> T7Entry;

static_assert( sizeof( T1Entry ) == 8  );
static_assert( sizeof( T2Entry ) == 16 );
static_assert( sizeof( T3Entry ) == 24 );
static_assert( sizeof( T4Entry ) == 24 );
static_assert( sizeof( T5Entry ) == 24 );
static_assert( sizeof( T6Entry ) == 16 );
static_assert( sizeof( T7Entry ) == 8  );

template<TableId table, uint32 _numBuckets>
struct DiskPlotInfo
{
    static constexpr uint32 _k = _K;

    using Entry    = FpEntry<table>;
    using TMeta    = typename TableMetaType<table>::MetaOut;

    using InputInfo = DiskPlotInfo<table-1, _numBuckets>;
    
    static constexpr int64  MaxBucketEntries       = (int64)bbconst_ceil( ( static_cast<int64>( (1ull << _k) / _numBuckets ) * BB_DP_XTRA_ENTRIES_PER_BUCKET ) );

    // Size of data outputted/generated by this table
    static constexpr uint32 MetaMultiplier         = static_cast<uint32>( TableMetaOut<table>::Multiplier );

    static constexpr uint32 BucketBits             = bblog2( _numBuckets );
    static constexpr uint32 BucketShift            = table < TableId::Table7 ? ( _k + kExtraBits) - BucketBits : _k - BucketBits;

    static constexpr uint32 YBitSize               = table < TableId::Table7 ? _k - bblog2( _numBuckets ) + kExtraBits : _k - bblog2( _numBuckets );
    static constexpr uint32 MapBitSize             = table == TableId::Table1 ? 0 : _k + 1; // This should be key bit size

    static constexpr uint32 PairBitSizeL           = _k + 1 - BucketBits;
    static constexpr uint32 PairBitSizeR           = 9;
    static constexpr uint32 PairBitSize            = PairBitSizeL + PairBitSizeR;

    static constexpr uint32 EntrySizePackedBits    = YBitSize + MapBitSize + ( MetaMultiplier * _k );
    static constexpr uint32 EntrySizeExpandedBits  = CDiv( EntrySizePackedBits, 64 ) * 64;

};
