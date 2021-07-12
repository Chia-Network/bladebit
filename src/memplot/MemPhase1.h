#pragma once
#include "PlotContext.h"

struct kBCJob;

template<typename T>
struct ReadWriteBuffer
{
    const T* read ;
    T*       write;

    inline ReadWriteBuffer( const T* read, T* write ) 
        : read ( read )
        , write( write )
    {}

    inline const T* Swap()
    {
        T* newWrite = (T*)read;
        read  = write;
        write = newWrite;
        // std::swap( read, write );
        return read;
    }
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



class MemPhase1
{
public:
    MemPhase1( MemPlotContext& context );

    void Run();

private:
    uint64 GenerateF1();

    void ForwardPropagate( uint64 entryCount );
    uint64 FpScan( const uint64 entryCount, const uint64* yBuffer, 
                   uint32* groupBoundaries, kBCJob jobs[MAX_THREADS] );

    uint64 FpPair( const uint64* yBuffer, kBCJob jobs[MAX_THREADS],
                   const uint64 groupCount, Pair* tmpPairBuffer, Pair* outPairBuffer );

    template<TableId tableId>
    uint64 FpComputeTable( uint64 entryCount, 
                           ReadWriteBuffer<uint64>& yBuffer, 
                           ReadWriteBuffer<uint64>& metaBuffer );

    template<TableId tableId>
    uint64 FpComputeSingleTable( uint64 entryCount, Pair* pairBuffer,
                               ReadWriteBuffer<uint64>& yBuffer, 
                               ReadWriteBuffer<uint64>& metaBuffer );

    template<TableId tableId, typename TMetaIn, typename TMetaOut>
    void FpComputeFx( const uint64 entryCount, const Pair* lrPairs,
                      const TMetaIn* inMetaBuffer, const uint64* inYBuffer,
                      TMetaOut* outMetaBuffer, uint64* outYBuffer );
    
    
    void WaitForPreviousPlotWriter();

private:
    MemPlotContext& _context;
};