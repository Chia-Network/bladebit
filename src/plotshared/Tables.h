#pragma once

enum class TableId
{
    Table1 = 0,
    Table2 = 1,
    Table3 = 2,
    Table4 = 3,
    Table5 = 4,
    Table6 = 5,
    Table7 = 6

    ,_Count
};


///
/// Helpers for working with metadata
///
struct Meta3 { uint64 m0, m1; };  // Used for when the metadata multiplier == 3
struct Meta4 : Meta3 {};          // Used for when the metadata multiplier == 4
struct NoMeta {};                 // Used for when the metadata multiplier == 0

template<typename TMeta>
struct SizeForMeta;

template<> struct SizeForMeta<uint32> { static constexpr size_t Value = 1; };
template<> struct SizeForMeta<uint64> { static constexpr size_t Value = 2; };
template<> struct SizeForMeta<Meta3>  { static constexpr size_t Value = 3; };
template<> struct SizeForMeta<Meta4>  { static constexpr size_t Value = 4; };
template<> struct SizeForMeta<NoMeta> { static constexpr size_t Value = 0; };

template<TableId Table>
struct TableMetaType;

template<> struct TableMetaType<TableId::Table2> { using MetaIn = uint32; using MetaOut = uint64; };
template<> struct TableMetaType<TableId::Table3> { using MetaIn = uint64; using MetaOut = Meta4;  };
template<> struct TableMetaType<TableId::Table4> { using MetaIn = Meta4;  using MetaOut = Meta4;  };
template<> struct TableMetaType<TableId::Table5> { using MetaIn = Meta4;  using MetaOut = Meta3;  };
template<> struct TableMetaType<TableId::Table6> { using MetaIn = Meta3;  using MetaOut = uint64; };
template<> struct TableMetaType<TableId::Table7> { using MetaIn = uint64; using MetaOut = NoMeta; };


/// Helper for obtaining the correct fx (y) output type per table
template<TableId table> struct YOut                  { using Type = uint64; };
template<>              struct YOut<TableId::Table7> { using Type = uint32; };