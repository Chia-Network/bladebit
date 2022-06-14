#pragma once
#include "ChiaConsts.h"

#define BB_DP_MAX_JOBS 256u

#define BB_DP_BUCKET_COUNT              ( 1u << kExtraBits ) // 64 with kExtraBits == 6 // #TODO: Remove this and make buckets configurable

#define BB_DP_MIN_BUCKET_COUNT 64      // Below 128 we can't fit y+map in a qword, so it's only available in bounded mode.
#define BB_DP_MAX_BUCKET_COUNT 1024

#define BB_DP_ENTRIES_PER_BUCKET        ( ( 1ull << _K ) / BB_DP_BUCKET_COUNT )
#define BB_DP_XTRA_ENTRIES_PER_BUCKET   1.1
#define BB_DP_ENTRY_SLICE_MULTIPLIER    1.07


#define BB_DP_MAX_BC_GROUP_PER_BUCKET 300000        // There's around 284,190 groups per bucket (of bucket size 64)
#define BB_DP_MAX_BC_GROUP_PER_K_32   (BB_DP_MAX_BC_GROUP_PER_BUCKET * 64ull)
#define BB_DP_XTRA_MATCHES_PER_THREAD 1024

// How many extra entries to load from the next bucket to ensure we have enough to hold the 2 groups's
// worth of entries. This is so that we can besure that we can complete matches from groups from the previous
// bucket that continue on to the next bucket. There's around 280-320 entries per group on k32. This should be enough
#define BB_DP_CROSS_BUCKET_MAX_ENTRIES 768

// Pretty big right now, but when buckets == 1024 it is needed.
// Might change it to dynamic.
#define BB_DISK_QUEUE_MAX_CMDS (4096*8) //1024

// Use at 256 buckets for line points so that
// we can save 1 iteration when sorting it.
#define BB_DPP3_LP_BUCKET_COUNT 256


/// 
/// DEBUG
///
// #define BB_DP_DBG_READ_EXISTING_F1 1
// #define BB_DP_DBG_VALIDATE_F1   1
// #define BB_DP_DBG_VALIDATE_FX   1
// #define BB_DP_DBG_VALIDATE_META 1
// #define BB_DP_DBG_PROTECT_FP_BUFFERS 1

// #define BB_DP_DBG_DONT_DELETE_TMP_FILES_PHASE 3

#define BB_DP_DBG_READ_BUCKET_COUNT_FNAME "bucket_count.tmp"
#define BB_DP_TABLE_COUNTS_FNAME          "table_counts.tmp"
#define BB_DP_DBG_PTR_BUCKET_COUNT_FNAME  "ptr_bucket_count.tmp"

#define BB_DP_DBG_TEST_DIR      "/home/harold/plot/dbg/"
#define BB_DP_DBG_REF_DIR       "/mnt/p5510a/reference/"

#define BB_DP_DBG_SKIP_PHASE_1  1
// #define BB_DP_DBG_SKIP_PHASE_2  1

// Skip all of Phase 1 except the C tables writing.
// #NOTE: BB_DP_DBG_SKIP_PHASE_1 Must be defined
// #define BB_DP_DBG_SKIP_TO_C_TABLES 1

// #define BB_DP_P1_SKIP_TO_TABLE 1
// #define BB_DP_P1_START_TABLE TableId::Table4

// Tmp file deletion (useful to keep around when developing)
#if _DEBUG
    // #define BB_DP_P1_KEEP_FILES 1
    // #define BB_DP_P3_KEEP_FILES 1
#endif


#if _DEBUG
    // Skip Phase 3 to the specific table (must have the files on disk ready)
    // #define BB_DBG_SKIP_P3_S1 1
    // #define BB_DP_DBG_P3_START_TABLE Table7

    // #define BB_DP_DBG_P3_KEEP_FILES 1
#endif

#define BB_DP_FP_MATCH_X_BUCKET 1
