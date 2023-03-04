#pragma once
#include "fse/fse.h"

struct CompressionInfo
{
    uint32 entrySizeBits;
    uint32 subtSizeBits;
    size_t tableParkSize;
    double ansRValue;
};

// #TODO: Rename to GetCompressionCTable/DTable
// #TODO: Add to a namespace
FSE_CTable* CreateCompressionCTable( const uint32 compressionLevel, size_t* outTableSize = nullptr );
FSE_DTable* CreateCompressionDTable( const uint32 compressionLevel, size_t* outTableSize = nullptr );
CompressionInfo GetCompressionInfoForLevel( const uint32 compressionLevel );
uint32 GetCompressedLPBitCount( const uint32 compressionLevel );

template<uint32 level>
struct CompressionLevelInfo 
{
    // static_assert( false, "Invalid compression level." );
};

template<>
struct CompressionLevelInfo<1>
{
    static constexpr uint32 ENTRY_SIZE      = 16;
    static constexpr uint32 STUB_BIT_SIZE   = 28;
    static constexpr size_t TABLE_PARK_SIZE = 8316;
    static constexpr double ANS_R_VALUE     = 5.66;
};

template<>
struct CompressionLevelInfo<2>
{
    static constexpr uint32 ENTRY_SIZE      = 15;
    static constexpr uint32 STUB_BIT_SIZE   = 24;
    static constexpr size_t TABLE_PARK_SIZE = 7350;
    static constexpr double ANS_R_VALUE     = 7.55;
};

template<>
struct CompressionLevelInfo<3>
{
    static constexpr uint32 ENTRY_SIZE      = 14;
    static constexpr uint32 STUB_BIT_SIZE   = 20;
    static constexpr size_t TABLE_PARK_SIZE = 6350; //6337;
    static constexpr double ANS_R_VALUE     = 8.89;
};

template<>
struct CompressionLevelInfo<4>
{
    static constexpr uint32 ENTRY_SIZE      = 13;
    static constexpr uint32 STUB_BIT_SIZE   = 16;
    static constexpr size_t TABLE_PARK_SIZE = 5325; //5312;
    static constexpr double ANS_R_VALUE     = 9.30;
};

template<>
struct CompressionLevelInfo<5>
{
    static constexpr uint32 ENTRY_SIZE      = 12;
    static constexpr uint32 STUB_BIT_SIZE   = 12;
    static constexpr size_t TABLE_PARK_SIZE = 4300; //4290;
    static constexpr double ANS_R_VALUE     = 9.30;
};

template<>
struct CompressionLevelInfo<6>
{
    static constexpr uint32 ENTRY_SIZE      = 11;
    static constexpr uint32 STUB_BIT_SIZE   = 8;
    static constexpr size_t TABLE_PARK_SIZE = 3273; //3263;
    static constexpr double ANS_R_VALUE     = 9.10;
};

template<>
struct CompressionLevelInfo<7>
{
    static constexpr uint32 ENTRY_SIZE      = 10;
    static constexpr uint32 STUB_BIT_SIZE   = 4;
    static constexpr size_t TABLE_PARK_SIZE = 2250; //2240;
    static constexpr double ANS_R_VALUE     = 8.60;
};

// #TODO: These are dummy values for now... Update with real values
template<>
struct CompressionLevelInfo<8>
{
    static constexpr uint32 ENTRY_SIZE      = 9;
    static constexpr uint32 STUB_BIT_SIZE   = 4;
    static constexpr size_t TABLE_PARK_SIZE = 6350; //2240;
    static constexpr double ANS_R_VALUE     = 8.60;
};

template<>
struct CompressionLevelInfo<9>
{
    static constexpr uint32 ENTRY_SIZE      = 8;
    static constexpr uint32 STUB_BIT_SIZE   = 30;
    static constexpr size_t TABLE_PARK_SIZE = 8808;
    static constexpr double ANS_R_VALUE     = 4.54;
};





