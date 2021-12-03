#pragma once

#include "ChiaConsts.h"

#define BB_DP_MAX_JOBS 256u

#define BB_DP_MIN_RAM_SIZE              5ull GB //( 3 GB + 9 MB )
#define BB_DP_BUCKET_COUNT              ( 1u << kExtraBits ) // 64 with kExtraBits == 6
#define BB_DP_ENTRIES_PER_BUCKET        ( ( 1ull << _K ) / BB_DP_BUCKET_COUNT )
#define BB_DP_XTRA_ENTRIES_PER_BUCKET   18000000u
#define BB_DP_MAX_ENTRIES_PER_BUCKET    ( BB_DP_ENTRIES_PER_BUCKET + BB_DP_XTRA_ENTRIES_PER_BUCKET )   // 67108864 per bucket if k=32, but
                                                                                                       // allow for more matches per bucket (after table 1)

#define BB_DP_BUCKET_TMP_SIZE     (65337 * 64ull)   // 4 MiB worth of ChaCha blocks + 1 block
#define BB_DP_ALL_BUCKET_TMP_SIZE ( BB_DP_BUCKET_TMP_SIZE * BB_DP_BUCKET_COUNT )


#define BB_DP_MAX_BC_GROUP_PER_BUCKET 300000        // There's around 284,190 groups per bucket
#define BB_DP_XTRA_MATCHES_PER_THREAD 1024



/// 
/// DEBUG
///
// #define BB_DP_DBG_READ_EXISTING_F1 1
// #define BB_DP_DBG_VALIDATE_Y    1

#define BB_DP_DBG_READ_BUCKET_COUNT_FNAME "bucket_count.tmp"

#define BB_DP_DBG_TEST_DIR      "/mnt/p5510a/disk_dbg/"
#define BB_DP_DBG_REF_DIR       "/mnt/p5510a/reference/"

