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

    void DbgValidateStep2Output( CudaK32PlotContext& cx );
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

static constexpr size_t P3_MAX_CTABLE_SIZE = 38u * 1024u;  // Should be more than enough

//static constexpr size_t P3_LP_BUCKET_COUNT       = BBCU_BUCKET_COUNT;// << 1;
//static constexpr size_t P3_LP_SLICE_ENTRY_COUNT  = BBCU_MAX_SLICE_ENTRY_COUNT;
//static constexpr uint32 P3_LP_BUCKET_BITS        = BBC_BUCKET_BITS;

// static constexpr uint32 P3_LP_BUCKET_BITS        = (uint32)(CuBBLog2( P3_LP_BUCKET_COUNT ));
//static constexpr size_t P3_LP_SLICE_ENTRY_COUNT  = ( CuCDiv( (size_t)( ( BBCU_TABLE_ENTRY_COUNT / P3_LP_BUCKET_COUNT / P3_LP_BUCKET_COUNT ) * P3_LP_BUCKET_MULTIPLER ),
                                                     //BBCU_XTRA_ENTRIES_PER_SLICE ) * BBCU_XTRA_ENTRIES_PER_SLICE + BBCU_XTRA_ENTRIES_PER_SLICE );
// static constexpr size_t P3_LP_BUCKET_ENTRY_COUNT = P3_LP_SLICE_ENTRY_COUNT * P3_LP_BUCKET_COUNT;

//static constexpr size_t P3_LP_BUCKET_STRIDE      = BBCU_BUCKET_ALLOC_ENTRY_COUNT;

// static constexpr size_t P3_LP_BUCKET_ALLOC_COUNT = ( CuCDiv( (size_t)( ( BBCU_TABLE_ENTRY_COUNT / P3_LP_BUCKET_COUNT / P3_LP_BUCKET_COUNT ) * P3_LP_BUCKET_MULTIPLER ),
//                                                     BBCU_XTRA_ENTRIES_PER_SLICE ) * BBCU_XTRA_ENTRIES_PER_SLICE + BBCU_XTRA_ENTRIES_PER_SLICE );
// //static constexpr size_t P3_LP_TABLE_ALLOC_COUNT  = P3_LP_BUCKET_STRIDE * BBCU_BUCKET_COUNT;

static constexpr size_t MAX_PARK_SIZE            = CalculateParkSize( TableId::Table1 );
static constexpr size_t DEV_MAX_PARK_SIZE        = CuCDiv( MAX_PARK_SIZE, sizeof( uint64 ) ) * sizeof( uint64 );   // Align parks to 64 bits, for easier writing of stubs

