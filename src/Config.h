
// Maximum number of supported threads. 
// Jobs will be stack allocated to this size.
// #TODO: This should perhaps be named max jobs.
#define MAX_THREADS 256
#define BB_MAX_JOBS MAX_THREADS

// Perform Y sorts at the block level.
// Unrolling loops by chacha block size.
#define Y_SORT_BLOCK_MODE 1

///
/// Debug Stuff
///

// #define DBG_VERIFY_SORT_F1 1
// #define DBG_VERIFY_SORT_FX 1

// #define DBG_VALIDATE_KB_GROUPS 1
// #define DBG_TEST_PAIRS 1

// #define DBG_WRITE_Y_VALUES 1

// #define BOUNDS_PROTECTION 1

#define DBG_TEST_PROOF_RANGE { 167000899, 3 }
// #define DBG_DUMP_PROOFS 1

// #define DBG_WRITE_PHASE_1_TABLES 1
#if !DBG_WRITE_PHASE_1_TABLES
    // #define DBG_READ_PHASE_1_TABLES 1
#endif

// #define DBG_WRITE_MARKED_TABLES 1
#if !DBG_WRITE_MARKED_TABLES 
    // #define DBG_READ_MARKED_TABLES 1
#endif

// #define DBG_WRITE_LINE_POINTS 1
// #define DBG_WRITE_SORTED_F7_TABLE 1

// Enable to test plotting times without writing to disk
// #define BB_BENCHMARK_MODE 1

// Enable for verbose debug logging.
// Good for tracking internal state changes and finding bugs.
// #define DBG_LOG_ENABLE 1


