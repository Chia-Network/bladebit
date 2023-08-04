#pragma once
#include <stddef.h>
#include "GreenReaper.h"

#ifdef _WIN32
    #define NOMINMAX 1
    #define WIN32_LEAN_AND_MEAN 1
    #include <libloaderapi.h>
#elif defined( __linux__ ) || defined( __APPLE__ )
    #include <dlfcn.h>
#else
    #error "Unsupported Platform"
#endif

#ifdef __cplusplus
extern "C" {
#endif

inline void* grLoadModule( const char* path )
{
    void* module = NULL;

    #ifdef _WIN32
        module = (void*)LoadLibraryA( (LPCSTR)path );
    #else
        module = dlopen( path, RTLD_LAZY | RTLD_LOCAL );
    #endif

    return module;
}

inline GRResult grPopulateApiFromModule( void* module, GRApi* api, const size_t apiStructSize, const int apiVersion )
{
    if( module == NULL || api == NULL )
        return GRResult_Failed;

    #define GR_STR(x) # x
    
    typedef GRResult (*grPopulateApiProc)( GRApi* api, size_t apiStructSize, int apiVersion );

    grPopulateApiProc populateApi = NULL;

    #ifdef _WIN32
        populateApi = (grPopulateApiProc)GetProcAddress( (HMODULE)module, (LPCSTR)GR_STR( grPopulateApi ) );
    #else
        populateApi = (grPopulateApiProc)dlsym( module, GR_STR( grPopulateApi ) );
    #endif

    if( populateApi == NULL )
        return GRResult_Failed;

    return populateApi( api, apiStructSize, apiVersion );
    #undef GR_STR
}

#ifdef __cplusplus
} // extern "C"
#endif
