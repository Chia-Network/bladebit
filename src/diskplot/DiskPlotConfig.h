#pragma once

#include "ChiaConsts.h"

#define BB_DP_MIN_RAM_SIZE ( 2 GB + 512 MB )
#define BB_DP_BUCKET_COUNT ( 1u << kExtraBits ) // 64 with kExtraBits == 6

#define BB_DP_XTRA_MATCHES_PER_THREAD 1024

#define BB_DP_MAX_JOBS 256u

#define BB_DP_BUCKET_TMP_SIZE     (65337 * 64ull)   // 4 MiB worth of ChaCha blocks + 1 block
#define BB_DP_ALL_BUCKET_TMP_SIZE ( BB_DP_BUCKET_TMP_SIZE * BB_DP_BUCKET_COUNT )


#define BB_DP_MAX_BC_GROUP_PER_BUCKET 300000    // There's around 284,190 groups per bucket



/// 
/// DEBUG
///
#define BB_DP_DBG_READ_EXISTING_F1 1

#define BB_DP_DBG_READ_BUCKET_COUNT_FNAME "bucket_count.tmp"

