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