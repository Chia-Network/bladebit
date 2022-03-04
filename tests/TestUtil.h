#pragma once

#include "catch2/catch_test_macros.hpp"
#include "util/Util.h"
#include "util/Log.h"
#include "threading/MTJob.h"
#include "threading/ThreadPool.h"
#include "util/jobs/MemJobs.h"


#define ENSURE( x ) \
    if( !(x) ) { ASSERT( x ); REQUIRE( x ); }

