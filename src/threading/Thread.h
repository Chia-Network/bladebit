#pragma once
#include "Platform.h"


typedef void (*ThreadRunner)( void* data );

enum class ThreadPriority
{
    Normal = 0,
    High
};

class Thread
{
public:

    Thread( size_t stackSize );
    Thread();
    ~Thread();

//     static uint64 SetAffinity( uint64 affinity );
    // void SetName( const char* name );
    template<typename TParam>
    void Run( void (*runner)( TParam* param ), TParam* param );

    void Run( ThreadRunner runner, void* param );

    // Cause the current thread to sleep
    static void Sleep( long milliseconds );

    // Not thread safe, should only ever be called by a single thread.
    bool WaitForExit( long milliseconds = -1 );

    bool HasExited() const;

    bool SetPriority( const ThreadPriority priority );
    
    // Unsafe
    inline ThreadId GetNativeHandle() const { return _threadId; }

private:
    
    #if PLATFORM_IS_UNIX
        static void* ThreadStarterUnix( Thread* thread );
    #elif PLATFORM_IS_WINDOWS
        static unsigned long ThreadStarterWin( intptr_t );
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
    pthread_mutex_t _launchMutex = {};
    pthread_cond_t  _launchCond  = {};

    pthread_mutex_t _exitMutex = {};
    pthread_cond_t  _exitCond  = {};
#endif

    enum class ThreadState : int
    {
        ReadyToRun = 0,
        Running,
        Exited,
    };

    std::atomic<ThreadState> _state;
};


//-----------------------------------------------------------
template<typename TParam>
inline void Thread::Run( void ( *runner )( TParam* param ), TParam* param )
{
    Run( reinterpret_cast<ThreadRunner>( runner ), (void*)param );
}

