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
}; ImplementArithmeticOps( TableId );

///
/// Helpers for working with metadata
///
struct  NoMeta {};                // Used for when the metadata multiplier == 0
struct  Meta4 { uint64 m0, m1; }; // Used for when the metadata multiplier == 4
struct  Meta3 : Meta4{};          // Used for when the metadata multiplier == 3

template<typename TMeta>
struct SizeForMeta;

template<> struct SizeForMeta<uint32> { static constexpr size_t Value = 1; };
template<> struct SizeForMeta<uint64> { static constexpr size_t Value = 2; };
template<> struct SizeForMeta<Meta3>  { static constexpr size_t Value = 3; };
template<> struct SizeForMeta<Meta4>  { static constexpr size_t Value = 4; };
template<> struct SizeForMeta<NoMeta> { static constexpr size_t Value = 0; };

template<TableId Table>
struct TableMetaType;

template<> struct TableMetaType<TableId::Table1> { using MetaIn = NoMeta; using MetaOut = uint32; };
template<> struct TableMetaType<TableId::Table2> { using MetaIn = uint32; using MetaOut = uint64; };
template<> struct TableMetaType<TableId::Table3> { using MetaIn = uint64; using MetaOut = Meta4;  };
template<> struct TableMetaType<TableId::Table4> { using MetaIn = Meta4;  using MetaOut = Meta4;  };
template<> struct TableMetaType<TableId::Table5> { using MetaIn = Meta4;  using MetaOut = Meta3;  };
template<> struct TableMetaType<TableId::Table6> { using MetaIn = Meta3;  using MetaOut = uint64; };
template<> struct TableMetaType<TableId::Table7> { using MetaIn = uint64; using MetaOut = NoMeta; };



template<TableId table> struct YOut                  { using Type = uint64; };
template<>              struct YOut<TableId::Table7> { using Type = uint32; };


///
/// For Disk Plotter
/// 

// Metadata input
template<TableId Table>
struct TableMetaIn;

template<> struct TableMetaIn<TableId::Table1>
{
    using MetaA = NoMeta; using MetaB = NoMeta;
    static constexpr size_t SizeA = 0; static constexpr size_t SizeB = 0;
    static constexpr size_t Multiplier = 0;
};

template<> struct TableMetaIn<TableId::Table2>
{
    using MetaA = uint32; using MetaB = NoMeta; 
    static constexpr size_t SizeA = sizeof( uint32 ); static constexpr size_t SizeB = 0;
    static constexpr size_t Multiplier = ( SizeA + SizeB ) / 4;
};
template<> struct TableMetaIn<TableId::Table3>
{
    using MetaA = uint64; using MetaB = NoMeta;
    static constexpr size_t SizeA = sizeof( uint64 ); static constexpr size_t SizeB = 0;
    static constexpr size_t Multiplier = ( SizeA + SizeB ) / 4;
};
template<> struct TableMetaIn<TableId::Table4>
{
    using MetaA = uint64; using MetaB = uint64;
    static constexpr size_t SizeA = sizeof( uint64 ); static constexpr size_t SizeB = sizeof( uint64 );
    static constexpr size_t Multiplier = ( SizeA + SizeB ) / 4;
};
template<> struct TableMetaIn<TableId::Table5>
{
    using MetaA = uint64; using MetaB = uint64;
    static constexpr size_t SizeA = sizeof( uint64 ); static constexpr size_t SizeB = sizeof( uint64 );
    static constexpr size_t Multiplier = ( SizeA + SizeB ) / 4;
};
template<> struct TableMetaIn<TableId::Table6>
{
    using MetaA = uint64; using MetaB = uint32;
    static constexpr size_t SizeA = sizeof( uint64 ); static constexpr size_t SizeB = sizeof( uint32 );
    static constexpr size_t Multiplier = ( SizeA + SizeB ) / 4;
};
template<> struct TableMetaIn<TableId::Table7>
{
    using MetaA = uint64; using MetaB = NoMeta;
    static constexpr size_t SizeA = sizeof( uint64 ); static constexpr size_t SizeB = 0;
    static constexpr size_t Multiplier = ( SizeA + SizeB ) / 4;
};


// Metadata output
template<TableId Table>
struct TableMetaOut;

template<> struct TableMetaOut<TableId::Table1>
{
    static constexpr TableId NextTable = TableId::Table2;

    using MetaA = TableMetaIn<NextTable>::MetaA; using MetaB = TableMetaIn<NextTable>::MetaB;
    static constexpr size_t SizeA = TableMetaIn<NextTable>::SizeA;
    static constexpr size_t SizeB = TableMetaIn<NextTable>::SizeB;
    static constexpr size_t Multiplier = ( SizeA + SizeB ) / 4;
};

template<> struct TableMetaOut<TableId::Table2>
{
    static constexpr TableId NextTable = TableId::Table3;

    using MetaA = TableMetaIn<NextTable>::MetaA; using MetaB = TableMetaIn<NextTable>::MetaB;
    static constexpr size_t SizeA = TableMetaIn<NextTable>::SizeA; 
    static constexpr size_t SizeB = TableMetaIn<NextTable>::SizeB;
    static constexpr size_t Multiplier = ( SizeA + SizeB ) / 4;
};

template<> struct TableMetaOut<TableId::Table3>
{
    static constexpr TableId NextTable = TableId::Table4;

    using MetaA = TableMetaIn<NextTable>::MetaA; using MetaB = TableMetaIn<NextTable>::MetaB;
    static constexpr size_t SizeA = TableMetaIn<NextTable>::SizeA;
    static constexpr size_t SizeB = TableMetaIn<NextTable>::SizeB;
    static constexpr size_t Multiplier = ( SizeA + SizeB ) / 4;
};

template<> struct TableMetaOut<TableId::Table4>
{
    static constexpr TableId NextTable = TableId::Table5;

    using MetaA = TableMetaIn<NextTable>::MetaA; using MetaB = TableMetaIn<NextTable>::MetaB;
    static constexpr size_t SizeA = TableMetaIn<NextTable>::SizeA;
    static constexpr size_t SizeB = TableMetaIn<NextTable>::SizeB;
    static constexpr size_t Multiplier = ( SizeA + SizeB ) / 4;
};

template<> struct TableMetaOut<TableId::Table5>
{
    static constexpr TableId NextTable = TableId::Table6;

    using MetaA = TableMetaIn<NextTable>::MetaA; using MetaB = TableMetaIn<NextTable>::MetaB;
    static constexpr size_t SizeA = TableMetaIn<NextTable>::SizeA;
    static constexpr size_t SizeB = TableMetaIn<NextTable>::SizeB;
    static constexpr size_t Multiplier = ( SizeA + SizeB ) / 4;
};

template<> struct TableMetaOut<TableId::Table6>
{
    static constexpr TableId NextTable = TableId::Table7;

    using MetaA = TableMetaIn<NextTable>::MetaA; using MetaB = TableMetaIn<NextTable>::MetaB;
    static constexpr size_t SizeA = TableMetaIn<NextTable>::SizeA;
    static constexpr size_t SizeB = TableMetaIn<NextTable>::SizeB;
    static constexpr size_t Multiplier = ( SizeA + SizeB ) / 4;
};


template<> struct TableMetaOut<TableId::Table7>
{
    using MetaA = NoMeta; using MetaB = NoMeta;
    static constexpr size_t SizeA      = 0; static constexpr size_t SizeB = 0;
    static constexpr size_t Multiplier = 0;
};

// static_assert( sizeof( NoMeta ) == 0, "Invalid NoMeta" );


/// For bounded k32
typedef uint32 K32Meta1;
typedef uint64 K32Meta2;
struct K32Meta3 { uint64 m0, m1; };
struct K32Meta4 { uint64 m0, m1; };
struct K32NoMeta {};

template<TableId rTable>
struct K32MetaType{};

template<> struct K32MetaType<TableId::Table1>{ using In = K32NoMeta; using Out = K32Meta1;  };
template<> struct K32MetaType<TableId::Table2>{ using In = K32Meta1;  using Out = K32Meta2;  };
template<> struct K32MetaType<TableId::Table3>{ using In = K32Meta2;  using Out = K32Meta4;  };
template<> struct K32MetaType<TableId::Table4>{ using In = K32Meta4;  using Out = K32Meta4;  };
template<> struct K32MetaType<TableId::Table5>{ using In = K32Meta4;  using Out = K32Meta3;  };
template<> struct K32MetaType<TableId::Table6>{ using In = K32Meta3;  using Out = K32Meta2;  };
template<> struct K32MetaType<TableId::Table7>{ using In = K32Meta2;  using Out = K32NoMeta; };

/// Helper for obtaining the correct fx (y) output type per table
template<TableId rTable> struct K32TYOut { using Type = uint64; };
template<>               struct K32TYOut<TableId::Table7> { using Type = uint32; };
