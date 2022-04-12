#pragma once

#ifdef _WIN32
    
    #define NOMINMAX 1
    #define WIN32_LEAN_AND_MEAN 1
    #include <Windows.h>
    #include <malloc.h>
    typedef HANDLE ThreadId;
    typedef HANDLE SemaphoreId;

    #define BBDebugBreak() __debugbreak()

// *nix
#elif __linux__ || __APPLE__
    
    #ifndef _GNU_SOURCE
        #define _GNU_SOURCE
    #endif

    #ifndef _LARGEFILE64_SOURCE 
        #define _LARGEFILE64_SOURCE 
    #endif

    #include <pthread.h>
    #include <semaphore.h>
    #include <alloca.h>
    #include <signal.h>
    #include <unistd.h>
    #include <sys/mman.h>
    #include <time.h>

    #if __linux__
        #include <sys/sysinfo.h>
    #elif __APPLE__
    #endif

    #define BBDebugBreak() raise( SIGTRAP )
    

    typedef pthread_t ThreadId;
    typedef sem_t     SemaphoreId;

#else
    #error Unsupported platform
#endif


#if !_DEBUG
    #undef BBDebugBreak
    #define BBDebugBreak() 
#endif

