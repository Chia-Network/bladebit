#include "AutoResetSignal.h"
#include "Util.h"
#include "util/Log.h"

//-----------------------------------------------------------
AutoResetSignal::AutoResetSignal()
{
#if PLATFORM_IS_WINDOWS
    _object = CreateEvent( NULL, FALSE, FALSE, NULL );
    FatalIf( !_object, "AutoResetSignal::AutoResetSignal() CreateEvent() failed." );
#else
    ZeroMem( &_object );

    int r;
    r = pthread_mutex_init( &_object.mutex, 0 );
    FatalIf( r, "AutoResetSignal::AutoResetSignal() pthread_mutex_init failed with error %d.", r );

    r = pthread_cond_init(  &_object.cond , 0 );
    FatalIf( r, "AutoResetSignal::AutoResetSignal() pthread_cond_init failed with error %d.", r );
#endif
}


//-----------------------------------------------------------
AutoResetSignal::~AutoResetSignal()
{
#if PLATFORM_IS_WINDOWS
    BOOL r = CloseHandle( _object );
    ASSERT( r );
    if( !r )
        Log::Error( "AutoResetSignal::~AutoResetSignal() CloseHandle() failed." );
#else
    // #TODO: Log error
    int r;
    r = pthread_mutex_destroy( &_object.mutex );
    if( r )
        Log::Error( "AutoResetSignal::~AutoResetSignal() pthread_mutex_destroy() failed." );

    r = pthread_cond_destroy ( &_object.cond  );
    if( r )
        Log::Error( "AutoResetSignal::~AutoResetSignal() pthread_cond_destroy() failed." );
#endif
}

//-----------------------------------------------------------
void AutoResetSignal::Signal()
{
#if PLATFORM_IS_WINDOWS
    BOOL r = SetEvent( _object );
    ASSERT( r );
    if( !r )
        Log::Error( "AutoResetSignal::Signal() SetEvent failed." );    
#else
    int r;

    r = pthread_mutex_lock( &_object.mutex );
    FatalIf( r, "AutoResetSignal::Signal pthread_mutex_lock() failed with error %d.", r );

    _object.signaled = true;

    r = pthread_cond_signal( &_object.cond  );
    FatalIf( r, "AutoResetSignal::Signal pthread_cond_signal() failed with error %d.", r );

    r = pthread_mutex_unlock( &_object.mutex );
    FatalIf( r, "AutoResetSignal::Signal pthread_mutex_unlock() failed with error %d.", r );
#endif
}

//-----------------------------------------------------------
AutoResetSignal::WaitResult AutoResetSignal::Wait( int32 timeoutMS )
{
    // #TODO: Check if the OS implementations fpin for a bit first,
    //        if not, we should spin for a bit before suspending.

#if PLATFORM_IS_WINDOWS
    if( timeoutMS == WaitInfinite )
        timeoutMS = INFINITE;
    
    DWORD r = WaitForSingleObject( _object, (DWORD)timeoutMS );

    switch( r )
    {
        case WAIT_OBJECT_0:
            return WaitResultOK;

        case WAIT_TIMEOUT:
            return WaitResultTimeOut;

        default:
            return WaitResultError;
    }
    
#else

    int r;
    int rc = 0;

    if( timeoutMS == WaitInfinite )
    {
        r = pthread_mutex_lock( &_object.mutex );
        FatalIf( r, "AutoResetSignal::Wait pthread_mutex_lock() failed with error %d.", r );

        while( !_object.signaled )
            rc = pthread_cond_wait( &_object.cond, &_object.mutex );

        r = pthread_mutex_unlock( &_object.mutex );
        FatalIf( r, "AutoResetSignal::Wait pthread_mutex_unlock() failed with error %d.", r );
    }
    else
    {
        struct timespec t;

        r = pthread_mutex_lock( &_object.mutex );
        FatalIf( r, "AutoResetSignal::Wait pthread_mutex_lock() failed with error %d.", r );

        clock_gettime( CLOCK_REALTIME, &t );
        timeoutMS += ( t.tv_sec * 1000 ) + ( t.tv_nsec / 1000000 );

        t.tv_sec  = (time_t)( timeoutMS / 1000 );
        t.tv_nsec = (long)( timeoutMS - (t.tv_sec * 1000) ) * 1000000;

        while( !_object.signaled && rc == 0 )
            rc = pthread_cond_timedwait( &_object.cond, &_object.mutex, &t );
        r = pthread_mutex_unlock( &_object.mutex );
        FatalIf( r, "AutoResetSignal::Wait pthread_mutex_unlock() failed with error %d.", r );
    }

    switch( rc )
    {
        case 0:
            return WaitResultOK;
        case ETIMEDOUT:
            return WaitResultTimeOut;
        default:
            return WaitResultError;
    }
#endif
}
