#pragma once

// Helpte to break on the assertion failure when debugging a test
#define ENSURE( ... ) \
    ASSERT( __VA_ARGS__ ); REQUIRE( __VA_ARGS__ )
