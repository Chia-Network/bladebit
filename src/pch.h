#pragma once

// Only for our own files, don't include in third party C files
#ifdef __cplusplus

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <cstdarg>
#include <chrono>

#if _DEBUG
    #include <assert.h>
    #define ASSERT( x ) assert( x )
#else
    #define ASSERT( x ) 
#endif

// Only include from C++ files
#include "Globals.h"
#include "Types.h"

#endif // __cplusplus