#pragma once

// Only for our own files, don't include in third party C files
#ifdef __cplusplus

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <cstdarg>
#include <chrono>
#include <atomic>
#include <cmath>
#include <memory>
#include <vector>
#include "Platform.h"

// Defined in Util.cpp
bool AssertLog( int line, const char* file, const char* func );

#if _DEBUG
    #include <assert.h>
    #define ASSERT( condition ) \
        { if( !(condition) ) { AssertLog( __LINE__, __FILE__, __FUNCTION__ ); BBDebugBreak(); } }
//     assert( x )
#else
    #define ASSERT( x ) 
#endif

// Only include from C++ files
#include "Globals.h"
#include "Types.h"
#include "Config.h"
#include "util/Span.h"

#endif // __cplusplus
