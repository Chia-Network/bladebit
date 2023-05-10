#pragma once

#ifdef _WIN32
    
    #ifndef NOMINMAX
        #define NOMINMAX 1
    #endif
    #define WIN32_LEAN_AND_MEAN 1
    // #NOTE: Not including globally anymore as we're getting some naming conflicts
    //#include <Windows.h>

    #include <malloc.h>

    // Must match Windows.h definitions.
    typedef void* HANDLE;
    #define INVALID_WIN32_HANDLE ((HANDLE)(intptr_t)-1) // Same as INVALID_HANDLE_VALUE
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

        typedef pthread_t ThreadId;
        typedef sem_t     SemaphoreId;
    #elif __APPLE__
        #include <dispatch/dispatch.h>

        typedef pthread_t ThreadId;
        typedef dispatch_semaphore_t SemaphoreId;
    #endif

    #define BBDebugBreak() raise( SIGTRAP )


#else
    #error Unsupported platform
#endif


#if !_DEBUG
    #undef BBDebugBreak
    #define BBDebugBreak() 
#endif

