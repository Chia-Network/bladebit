#pragma once
#include "Platform.h"
#include <atomic>

typedef void (*ThreadRunner)( void* data );

class Thread
{
public:

    Thread( size_t stackSize );
    Thread();
    ~Thread();

//     static uint64 SetAffinity( uint64 affinity );
    // void SetName( const char* name );

    void Run( ThreadRunner runner, void* param );

    // Cause the current thread to sleep
    static void Sleep( long milliseconds );

    bool WaitForExit( long milliseconds = -1 );

    bool HasExited() const;

private:
    
    #if PLATFORM_IS_UNIX
        static void* ThreadStarterUnix( Thread* thread );
    #elif PLATFORM_IS_WINDOWS
        static DWORD ThreadStarterWin( LPVOID );
    #else
        #error Unimplemented
    #endif

private:
    ThreadId     _threadId = 0;
    ThreadRunner _runner   = nullptr;
    void*        _runParam = nullptr;

#if PLATFORM_IS_UNIX
    // Used for launching the thread and 
    // suspending it until it actually runs.
    pthread_mutex_t _launchMutex;
    pthread_cond_t  _launchCond ;
#endif

    enum class ThreadState : int
    {
        ReadyToRun = 0,
        Running    = 1,
        Exited     = 2
    };

    std::atomic<ThreadState> _state;
};