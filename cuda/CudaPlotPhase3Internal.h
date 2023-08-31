#pragma once

#include "CudaPlotContext.h"
#include "plotting/CTables.h"
#include "ChiaConsts.h"

#if _DEBUG
    #include "util/BitField.h"
    #include "plotmem/LPGen.h"
    #include "plotdisk/jobs/IOJob.h"
    #include "algorithm/RadixSort.h"
    #include "plotmem/ParkWriter.h"
    #include "b3/blake3.h"

    void DbgValidateStep2Output( CudaK32PlotContext& cx );

    void DbgHashData( const void* data, size_t size, const char* name, uint32 index );

    void DbgFinishAndPrintHash( blake3_hasher& hasher, const char* name, uint32 index );
    template<typename T>
    inline void DbgHashDataT( const T* data, uint64 count, const char* name, uint32 index )
    {
        DbgHashData( data, (size_t)count * sizeof( T ), name, index );
    }
#endif

using LMap = CudaK32Phase3::LMap;
using RMap = CudaK32Phase3::RMap;

static_assert( alignof( LMap ) == sizeof( uint32 ) );

// #TODO: Remove this. It is unneeeded.
#define P3_PRUNED_BUCKET_MULTIPLIER     0.98    // Enough to hold the largest pruned bucket size

#define P3_PRUNED_SLICE_MAX             BBCU_MAX_SLICE_ENTRY_COUNT     //(CuCDiv( (size_t)((BBCU_TABLE_ENTRY_COUNT/BBCU_BUCKET_COUNT/BBCU_BUCKET_COUNT)*P3_PRUNED_BUCKET_MULTIPLIER), 4096 ) * 4096 + 4096)
#define P3_PRUNED_BUCKET_MAX            BBCU_BUCKET_ALLOC_ENTRY_COUNT  //(P3_PRUNED_SLICE_MAX*BBCU_BUCKET_COUNT)
#define P3_PRUNED_TABLE_MAX_ENTRIES     BBCU_TABLE_ALLOC_ENTRY_COUNT   //(P3_PRUNED_BUCKET_MAX*BBCU_BUCKET_COUNT)
#define P3_PRUNED_MAX_PARKS_PER_BUCKET  ((P3_PRUNED_BUCKET_MAX/kEntriesPerPark)+2)


static constexpr size_t P3_MAX_CTABLE_SIZE         = 38u * 1024u;  // Should be more than enough
static constexpr size_t P3_MAX_P7_PARKS_PER_BUCKET = CDiv( BBCU_BUCKET_ALLOC_ENTRY_COUNT, kEntriesPerPark ) + 2;
static constexpr size_t P3_PARK_7_SIZE             = CalculatePark7Size( BBCU_K );
static_assert( sizeof( uint64 ) * BBCU_BUCKET_ALLOC_ENTRY_COUNT >= P3_MAX_P7_PARKS_PER_BUCKET * P3_PARK_7_SIZE );

static constexpr size_t MAX_PARK_SIZE            = CalculateParkSize( TableId::Table1 );
static constexpr size_t DEV_MAX_PARK_SIZE        = CuCDiv( MAX_PARK_SIZE, sizeof( uint64 ) ) * sizeof( uint64 );   // Align parks to 64 bits, for easier writing of stubs

