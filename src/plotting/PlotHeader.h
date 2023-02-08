#pragma once
#include "ChiaConsts.h"

enum class PlotVersion : uint32
{
    v1_0 = 0,
    v2_0 = CHIA_PLOT_VERSION_2_0_0
};

enum class PlotFlags : uint32
{
    None       = 0,
    Compressed = 1 << 0,

}; ImplementFlagOps( PlotFlags );


struct PlotHeader
{
    byte   id  [BB_PLOT_ID_LEN]        = { 0 };
    byte   memo[BB_PLOT_MEMO_MAX_SIZE] = { 0 };
    uint   memoLength                  = 0;
    uint32 k                           = 0;
    uint64 tablePtrs[10]               = { 0 };
};

typedef PlotHeader PlotFileHeaderV1;

struct PlotFileHeaderV2 : PlotFileHeaderV1
{
    PlotFlags flags             = PlotFlags::None;
    byte      compressionLevel  = 0;
    uint64    tableSizes[10]    = { 0 };
};