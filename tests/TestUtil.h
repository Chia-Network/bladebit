#pragma once

#include "catch2/catch_test_macros.hpp"
#include "util/Util.h"
#include "util/Log.h"
#include "threading/MTJob.h"
#include "threading/ThreadPool.h"
#include "util/jobs/MemJobs.h"

// Helper to break on the assertion failure when debugging a test.
// You can pass an argument to the catch to break on asserts, however,
// REQUIRE itself is extremely heavy to call directly.
#define ENSURE( x ) \
    if( !(x) ) { ASSERT( x ); REQUIRE( x ); }

