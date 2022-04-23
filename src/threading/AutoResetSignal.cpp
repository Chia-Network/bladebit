#include "AutoResetSignal.h"
#include "util/Util.h"
#include "util/Log.h"

#if PLATFORM_IS_WINDOWS
    #include <Windows.h>
#endif

//-----------------------------------------------------------
AutoResetSignal::AutoResetSignal()
{
#if PLATFORM_IS_WINDOWS
    _object = CreateEvent( NULL, FALSE, FALSE, NULL );
    PanicIf( !_object, "AutoResetSignal::AutoResetSignal() CreateEvent() failed with error: %d.", (int32)::GetLastError() );
#else
    ZeroMem( &_object );

    int r;
    r = pthread_mutex_init( &_object.mutex, 0 );
    PanicIf( r, "AutoResetSignal::AutoResetSignal() pthread_mutex_init failed with error %d.", r );

    r = pthread_cond_init(  &_object.cond , 0 );
    PanicIf( r, "AutoResetSignal::AutoResetSignal() pthread_cond_init failed with error %d.", r );
#endif
}


//-----------------------------------------------------------
AutoResetSignal::~AutoResetSignal()
{
#if PLATFORM_IS_WINDOWS
    const BOOL r = ::CloseHandle( _object );
    PanicIf( !r, "AutoResetSignal::~AutoResetSignal() CloseHandle() failed with error: %d.", (int32)::GetLastError() );
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
void AutoResetSignal::Reset()
{
#if PLATFORM_IS_WINDOWS
    const BOOL r = ::ResetEvent( _object );
    PanicIf( !r, "AutoResetSignal::Reset ResetEvent() failed with error: %d.", (int32)GetLastError() );
#else
    int r;

    r = pthread_mutex_lock( &_object.mutex );
    PanicIf( r, "AutoResetSignal::Signal pthread_mutex_lock() failed with error %d.", r );

    _object.signaled = false;

    r = pthread_mutex_unlock( &_object.mutex );
    PanicIf( r, "AutoResetSignal::Signal pthread_mutex_unlock() failed with error %d.", r );
#endif
}

//-----------------------------------------------------------
void AutoResetSignal::Signal()
{
#if PLATFORM_IS_WINDOWS
    const BOOL r = SetEvent( _object );
    PanicIf( !r, "AutoResetSignal::Signal() SetEvent() failed with error: %d.", (int32)::GetLastError() );
#else
    int r;

    r = pthread_mutex_lock( &_object.mutex );
    PanicIf( r, "AutoResetSignal::Signal pthread_mutex_lock() failed with error %d.", r );

    _object.signaled = true;

    r = pthread_cond_signal( &_object.cond  );
    PanicIf( r, "AutoResetSignal::Signal pthread_cond_signal() failed with error %d.", r );

    r = pthread_mutex_unlock( &_object.mutex );
    PanicIf( r, "AutoResetSignal::Signal pthread_mutex_unlock() failed with error %d.", r );
#endif
}

//-----------------------------------------------------------
AutoResetSignal::WaitResult AutoResetSignal::Wait( int32 timeoutMS )
{
#if PLATFORM_IS_WINDOWS
    if( timeoutMS == WaitInfinite )
        timeoutMS = INFINITE;
    
    const DWORD r = WaitForSingleObject( _object, (DWORD)timeoutMS );

    switch( r )
    {
        case WAIT_OBJECT_0:
            return WaitResultOK;

        case WAIT_TIMEOUT:
            return WaitResultTimeOut;

        default:
            // We never really handle this case, so just panic out
            Panic( "AutoResetSignal::Wait WaitForSingleObject() failed with error: %d.", (int32)::GetLastError() );
            return WaitResultError;
    }
    
#else

    int r;
    int rc = 0;

    if( timeoutMS == WaitInfinite )
    {
        r = pthread_mutex_lock( &_object.mutex );
        PanicIf( r, "AutoResetSignal::Wait pthread_mutex_lock() failed with error %d.", r );

        while( !_object.signaled )
            rc = pthread_cond_wait( &_object.cond, &_object.mutex );

        _object.signaled = false;

        r = pthread_mutex_unlock( &_object.mutex );
        PanicIf( r, "AutoResetSignal::Wait pthread_mutex_unlock() failed with error %d.", r );
    }
    else
    {
        struct timespec t;

        r = pthread_mutex_lock( &_object.mutex );
        PanicIf( r, "AutoResetSignal::Wait pthread_mutex_lock() failed with error %d.", r );

        clock_gettime( CLOCK_REALTIME, &t );
        timeoutMS += ( t.tv_sec * 1000 ) + ( t.tv_nsec / 1000000 );

        t.tv_sec  = (time_t)( timeoutMS / 1000 );
        t.tv_nsec = (long)( timeoutMS - (t.tv_sec * 1000) ) * 1000000;

        while( !_object.signaled && rc == 0 )
            rc = pthread_cond_timedwait( &_object.cond, &_object.mutex, &t );

        _object.signaled = false;

        r = pthread_mutex_unlock( &_object.mutex );
        PanicIf( r, "AutoResetSignal::Wait pthread_mutex_unlock() failed with error %d.", r );
    }

    switch( rc )
    {
        case 0:
            return WaitResultOK;
        case ETIMEDOUT:
            return WaitResultTimeOut;
        default:
            Panic( "AutoResetSignal::Wait Unexpected return code for pthread_cond_timedwait(): %d.", rc );
            return WaitResultError;
    }
#endif
}
