#include "../../threading/Thread.h"
#include "../../Util.h"
#include "../../Globals.h"
#include "SysHost.h"
#include "util/Log.h"

typedef void* (*PthreadFunc)( void* param );

//-----------------------------------------------------------
Thread::Thread( size_t stackSize )
{
     // Configure stack size
    if( stackSize < 1024 * 4 )
        Fatal( "Thread stack size is too small." );
    
    // Align to 8 bytes
    stackSize = RoundUpToNextBoundary( stackSize, 8 );

    _state.store( ThreadState::ReadyToRun, std::memory_order_release );

#if PLATFORM_IS_UNIX

    pthread_attr_t  attr;
    
    int r = pthread_attr_init( &attr );
    if( r ) Fatal( "pthread_attr_init() failed." );
    
    r = pthread_attr_setstacksize( &attr, stackSize );
    if( r ) Fatal( "pthread_attr_setstacksize() failed." );

    // Initialize suspended mode signal
    r = pthread_cond_init(  &_launchCond,  NULL );
    if( r ) Fatal( "pthread_cond_init() failed." );

    r = pthread_mutex_init( &_launchMutex, NULL );
    if( r ) Fatal( "pthread_mutex_init() failed." );
    
    r = pthread_create( &_threadId, &attr, (PthreadFunc)&Thread::ThreadStarterUnix, this );
    if( r ) Fatal( "pthread_create() failed." );
    
    r = pthread_attr_destroy( &attr );
    if( r ) Fatal( "pthread_attr_destroy() failed." );

#elif PLATFORM_IS_WINDOWS
    

#else
    #error Not implemented
#endif
}

//-----------------------------------------------------------
Thread::Thread() : Thread( 8 MB )
{
}

//-----------------------------------------------------------
Thread::~Thread()
{
    const bool didExit = _state.load( std::memory_order_relaxed ) == ThreadState::Exited;

    if( !didExit )
    {
        // Thread should have exited already
        #if PLATFORM_IS_UNIX
            pthread_cancel( _threadId );

            pthread_mutex_destroy( &_launchMutex );
            pthread_cond_destroy ( &_launchCond );

            ZeroMem( &_launchMutex );
            ZeroMem( &_launchCond  );
        #else
            #error Unimplemented
        #endif
    }

    _threadId = 0;
}

//-----------------------------------------------------------
// uint64 Thread::SetAffinity( uint64 affinity )
// {
//     return SysHost::SetCurrentThreadAffinityMask( affinity );
// }

//-----------------------------------------------------------
bool Thread::HasExited() const
{
    return _state.load( std::memory_order_relaxed ) == ThreadState::Exited;
}

//-----------------------------------------------------------
void Thread::Run( ThreadRunner runner, void* param )
{
    ASSERT( runner );
    const bool canRun = _state.load( std::memory_order_relaxed ) == ThreadState::ReadyToRun;

    ASSERT( canRun );
    if( !canRun )
        return;

    ThreadState expected = ThreadState::ReadyToRun;
    if( !_state.compare_exchange_strong( expected, ThreadState::Running,
                                    std::memory_order_release,
                                    std::memory_order_relaxed ) )
    {
        // Another thread ran us first.
        return;
    }
    
    _runner   = runner;
    _runParam = param;

    #if PLATFORM_IS_UNIX
        // Signal thread to resume
        int r = pthread_mutex_lock( &_launchMutex );
        if( r ) Fatal( "pthread_mutex_lock() failed." );

        r = pthread_cond_signal( &_launchCond );
        if( r ) Fatal( "pthread_cond_signal() failed." );

        r = pthread_mutex_unlock( &_launchMutex );
        if( r ) Fatal( "pthread_mutex_unlock() failed." );
    #endif
}


//-----------------------------------------------------------
void Thread::Sleep( long milliseconds )
{
    #if PLATFORM_IS_UNIX

        long seconds = milliseconds / 1000;
        milliseconds -= seconds * 1000;

        struct timespec req;
        req.tv_sec  = seconds;
        req.tv_nsec = milliseconds * 1000000;

        #if _DEBUG
        int r =
        #endif
        nanosleep( &req, NULL );
        ASSERT( !r );
    #else
        #error Unimplemented
    #endif
}

//-----------------------------------------------------------
bool Thread::WaitForExit( long milliseconds )
{
    ThreadState state = _state.load( std::memory_order_relaxed );
    if( state == ThreadState::Exited )
        return true;

    #if PLATFORM_IS_UNIX
        
        int r;

        if( milliseconds > 0 )
        {   
            // #TODO: Support on apple with a condition variable and mutex pair
            #if __APPLE__
                return false;
            #else
                long seconds = milliseconds / 1000;
                milliseconds -= seconds * 1000;

                struct timespec abstime;

                if( clock_gettime( CLOCK_REALTIME, &abstime ) == -1 )
                {
                    ASSERT( 0 );
                    return false;
                }

                abstime.tv_sec  += seconds;
                abstime.tv_nsec += milliseconds * 1000000;

                r = pthread_timedjoin_np( _threadId, NULL, &abstime );
                ASSERT( !r || r == ETIMEDOUT );
            #endif
        }
        else
        {
            r = pthread_join( _threadId, NULL );
        }

        return r == 0;
    #else
        #error Unimplemented
    #endif
}

// Starts up a thread.
//-----------------------------------------------------------
void* Thread::ThreadStarterUnix( Thread* t )
{
    // On Linux, it suspends it until it is signaled to run.
    int r = pthread_mutex_lock( &t->_launchMutex );
    if( r ) Fatal( "pthread_mutex_lock() failed." );

    while( t->_state.load( std::memory_order_relaxed ) == ThreadState::ReadyToRun )
    {
        r = pthread_cond_wait( &t->_launchCond, &t->_launchMutex );
        if( r ) Fatal( "pthread_cond_wait() failed." );
        break;
    }

    r = pthread_mutex_unlock( &t->_launchMutex );
    if( r ) Fatal( "pthread_mutex_unlock() failed." );

    pthread_mutex_destroy( &t->_launchMutex );
    pthread_cond_destroy ( &t->_launchCond  );

    ZeroMem( &t->_launchMutex );
    ZeroMem( &t->_launchCond  );

    // Run the thread function
    t->_runner( t->_runParam );

    // Thread has exited
    t->_state.store( ThreadState::Exited, std::memory_order_release );
    
    // TODO: Signal if waiting to be joined
    pthread_exit( nullptr );
    
    return nullptr;
}

