#include "threading/Thread.h"
#include "util/Util.h"
#include "Globals.h"
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

    pthread_attr_t  attr;
    
    int r = pthread_attr_init( &attr );
    PanicIf( r, "pthread_attr_init() failed with error %d.", r );
    
    r = pthread_attr_setstacksize( &attr, stackSize );
    PanicIf( r, "pthread_attr_setstacksize() failed with error %d.", r );

    // Initialize suspended mode signal and exit signals
    r = pthread_cond_init(  &_launchCond,  NULL );
    PanicIf( r, "pthread_cond_init() failed with error %d.", r );

    r = pthread_mutex_init( &_launchMutex, NULL );
    PanicIf( r, "pthread_mutex_init() failed with error %d.", r );

    r = pthread_cond_init(  &_exitCond,  NULL );
    PanicIf( r, "pthread_cond_init() failed with error %d.", r );

    r = pthread_mutex_init( &_exitMutex, NULL );
    PanicIf( r, "pthread_mutex_init() failed with error %d.", r );
    
    r = pthread_create( &_threadId, &attr, (PthreadFunc)&Thread::ThreadStarterUnix, this );
    PanicIf( r, "pthread_create() failed with error %d.", r );
    
    r = pthread_attr_destroy( &attr );
    PanicIf( r, "pthread_attr_destroy() failed with error %d.", r );
}

//-----------------------------------------------------------
Thread::Thread() : Thread( 8 MiB )
{
}

//-----------------------------------------------------------
Thread::~Thread()
{
    const bool didExit = _state.load( std::memory_order_relaxed ) == ThreadState::Exited;

    if( !didExit )
    {
        // Thread should have exited already
        ASSERT( 0 );
        
        pthread_cancel( _threadId );

        pthread_mutex_destroy( &_launchMutex );
        pthread_cond_destroy ( &_launchCond );

        ZeroMem( &_launchMutex );
        ZeroMem( &_launchCond  );
    }

    pthread_mutex_destroy( &_exitMutex );
    pthread_cond_destroy ( &_exitCond );

    ZeroMem( &_exitMutex );
    ZeroMem( &_exitCond  );

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
        // Another thread preempted us.
        return;
    }
    
    _runner   = runner;
    _runParam = param;


    // Signal thread to resume
    int r = pthread_mutex_lock( &_launchMutex );
    PanicIf( r, "pthread_mutex_lock() failed with error %d.", r );

    r = pthread_cond_signal( &_launchCond );
    PanicIf( r, "pthread_cond_signal() failed with error %d.", r );

    r = pthread_mutex_unlock( &_launchMutex );
    PanicIf( r, "pthread_mutex_unlock() failed with error %d.", r );
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

        #if _DEBUG
        ASSERT( !r );
        #endif

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

    // Immediate return?
    if( milliseconds == 0 || state != ThreadState::Running )
        return false;

    int r = 0;
    int waitResult = 0;

    if( milliseconds > 0 )
    {
        int r = pthread_mutex_lock( &_exitMutex );
        PanicIf( r, "pthread_mutex_lock() failed with error %d.", r );

        state = _state.load( std::memory_order_relaxed );
        if( state != ThreadState::Exited )
        {
            struct timespec abstime = {};
            
            long seconds = milliseconds / 1000;
            milliseconds -= seconds * 1000;

            #if !__APPLE__
                r = clock_gettime( CLOCK_REALTIME, &abstime );
                PanicIf( r, "clock_gettime() failed with error %d", r );
            #endif

            abstime.tv_sec  += seconds;
            abstime.tv_nsec += milliseconds * 1000000l;

            // #NOTE: On macOS it seems that using absolute time (pthread_cond_timedwait) with anything 
            //        less than 1 second is invalid, therefore we use this variant which works with
            //        smaller wait times.
            #if __APPLE__
                waitResult = pthread_cond_timedwait_relative_np( &_exitCond, &_exitMutex, &abstime );
            #else
                waitResult = pthread_cond_timedwait( &_exitCond, &_exitMutex, &abstime );
            #endif
            if( waitResult != 0 && waitResult != ETIMEDOUT ) 
                Panic( "pthread_cond_timedwait() failed with error %d.", waitResult );
        }

        r = pthread_mutex_unlock( &_exitMutex );
        PanicIf( r, "pthread_mutex_unlock() failed with error %d.", r );

        state = _state.load( std::memory_order_relaxed );
        if( waitResult == ETIMEDOUT && state != ThreadState::Exited )
            return false;
    }

    void* ret = nullptr;
    r = pthread_join( _threadId, &ret ); 
    ASSERT( !r ); (void)r;
    ASSERT( _state.load( std::memory_order_relaxed ) == ThreadState::Exited );

    return true;
}

// Starts up a thread.
//-----------------------------------------------------------
void* Thread::ThreadStarterUnix( Thread* t )
{
    // On Linux, it suspends it until it is signaled to run.
    {
        int r = pthread_mutex_lock( &t->_launchMutex );
        PanicIf( r, "pthread_mutex_lock() failed with error %d.", r );

        if( t->_state.load( std::memory_order_relaxed ) == ThreadState::ReadyToRun )
        {
            r = pthread_cond_wait( &t->_launchCond, &t->_launchMutex );
            PanicIf( r, "pthread_cond_wait() failed with error %d.", r );
        }

        r = pthread_mutex_unlock( &t->_launchMutex );
        PanicIf( r, "pthread_mutex_unlock() failed with error %d.", r );

        ASSERT( t->_state.load( std::memory_order_acquire ) == ThreadState::Running );
    }

    pthread_mutex_destroy( &t->_launchMutex );
    pthread_cond_destroy ( &t->_launchCond  );

    ZeroMem( &t->_launchMutex );
    ZeroMem( &t->_launchCond  );

    // Run the thread function
    t->_runner( t->_runParam );


    // Signal if waiting to be joined
    {
        int r = pthread_mutex_lock( &t->_exitMutex );
        PanicIf( r, "pthread_mutex_lock() failed with error %d.", r );

        // Thread has exited
        t->_state.store( ThreadState::Exited, std::memory_order_release );

        r = pthread_cond_signal( &t->_exitCond );
        PanicIf( r, "pthread_cond_signal() failed with error %d.", r );

        r = pthread_mutex_unlock( &t->_exitMutex );
        PanicIf( r, "pthread_mutex_unlock() failed with error %d.", r );
    }

    // pthread_exit( nullptr );
    
    return nullptr;
}

//-----------------------------------------------------------
bool Thread::SetPriority( const ThreadPriority priority )
{
    // #TODO: Implement
    // struct sched_param sched;

    switch( priority )
    {
        // case ThreadPriority::Normal:
        
        //     break;

        // case ThreadPriority::High:
        
        //     break;
            
        
        default:
            ASSERT( 0 );
            return false;
    }
}

