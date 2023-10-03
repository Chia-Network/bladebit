#include "Semaphore.h"
#include "util/Util.h"

#if PLATFORM_IS_WINDOWS
    #include <Windows.h>
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

//-----------------------------------------------------------
Semaphore::Semaphore( int initialCount )
    #if PLATFORM_IS_WINDOWS || PLATFORM_IS_APPLE
    : _count( initialCount )
    #endif
{
    ASSERT( initialCount >= 0 );

    #if PLATFORM_IS_LINUX
        int r = sem_init( &_id, 0, initialCount );
        if( r != 0 )
            Panic( "sem_init() failed with error: %d.", errno );
    #elif PLATFORM_IS_WINDOWS
        _id = CreateSemaphore( NULL,(LONG)initialCount, std::numeric_limits<LONG>::max(), NULL );
        if( _id == NULL )
        {
            Panic( "CreateSemaphore() failed with error: %d.", (int32)::GetLastError() );
        }
    #elif PLATFORM_IS_APPLE
        _id = dispatch_semaphore_create( (intptr_t)initialCount );
        PanicIf( !_id, "dispatch_semaphore_create failed." );
    #else
        #error Unimplemented
    #endif
}

//-----------------------------------------------------------
Semaphore::~Semaphore()
{
    #if PLATFORM_IS_LINUX
        int r = sem_destroy( &_id );
        if( r != 0 )
            Panic( "sem_destroy() failed with error: %d.", errno );
    #elif PLATFORM_IS_WINDOWS
        if( !CloseHandle( _id ) )
            Panic( "CloseHandle( semaphore ) failed with error: %d.", (int32)::GetLastError() );
    #elif PLATFORM_IS_APPLE
        dispatch_release( _id );
    #else
        #error Unimplemented
    #endif
}

//-----------------------------------------------------------
void Semaphore::Wait()
{
    #if PLATFORM_IS_LINUX
        const int r = sem_wait( &_id );
        if( r != 0 )
            Panic( "sem_wait() failed." );
    #elif PLATFORM_IS_WINDOWS
        const DWORD r = WaitForSingleObject( _id, INFINITE );
        if( r != WAIT_OBJECT_0 )
            Panic( "Semaphore::Wait - WaitForSingleObject() failed with error: %d.", (int32)::GetLastError() );
    #elif PLATFORM_IS_APPLE
        intptr_t r = dispatch_semaphore_wait( _id, DISPATCH_TIME_FOREVER );
        PanicIf( r != 0, "dispatch_semaphore_wait() failed." );
    #else
        #error Unimplemented
    #endif
}

//-----------------------------------------------------------
bool Semaphore::Wait( long milliseconds )
{
    ASSERT( milliseconds > 0 );

    if( milliseconds < 1 )
    {
        Wait();
        return true;
    }

    #if PLATFORM_IS_LINUX

        long seconds = milliseconds / 1000;
        milliseconds -= seconds * 1000;

        struct timespec abstime;

        if( clock_gettime( CLOCK_REALTIME, &abstime ) == -1 )
        {
            ASSERT( 0 );
            return false;
        }

        int r = sem_timedwait( &_id, &abstime );

        if( r != 0 && r != ETIMEDOUT )
            Panic( "sem_wait() failed." );

        return r == 0;

    #elif PLATFORM_IS_WINDOWS
        const DWORD r = WaitForSingleObject( _id, (DWORD)milliseconds );

        if( r != WAIT_OBJECT_0 && r != WAIT_TIMEOUT )
            Panic( "Semaphore::Wait - WaitForSingleObject() failed with error: %d", (int32)GetLastError() );

        return r == WAIT_OBJECT_0;
    #elif PLATFORM_IS_APPLE
        const int64_t         nanoseconds = (int64_t)milliseconds * 1000000ll;
        const dispatch_time_t timeout     = dispatch_time( DISPATCH_TIME_NOW, nanoseconds );

        const intptr_t r = dispatch_semaphore_wait( _id, timeout );

        return r == 0;  // Non-zero = timed-out
    #else
        #error Unimplemented
    #endif
}

//-----------------------------------------------------------
int Semaphore::GetCount()
{
    #if PLATFORM_IS_LINUX
        
        int value = 0;
        int r = sem_getvalue( &_id, &value );
        
        if( r != 0 )
            Panic( "sem_getvalue() failed." );

        return value;
    #elif PLATFORM_IS_WINDOWS || PLATFORM_IS_APPLE
        return _count.load( std::memory_order_release );
    #else
        #error Unimplemented
    #endif
}

//-----------------------------------------------------------
void Semaphore::Release()
{
    #if PLATFORM_IS_LINUX
        int r = sem_post( &_id );
        if( r != 0 )
            Panic( "sem_post() failed." );
    #elif PLATFORM_IS_WINDOWS
        LONG prevCount;
        BOOL r = ::ReleaseSemaphore( _id, 1, &prevCount );
        
        if( !r )
            Panic( "ReleaseSemaphore() failed with error: %d.", (int32)::GetLastError() );

        _count++;
    #elif PLATFORM_IS_APPLE
        // #TODO: Return value? Apple doesn't say what it does:
        //  #See: https://developer.apple.com/documentation/dispatch/1452919-dispatch_semaphore_signal
        dispatch_semaphore_signal( _id );

        _count++;
    #else
        #error Unimplemented
    #endif
}

#pragma GCC diagnostic pop

