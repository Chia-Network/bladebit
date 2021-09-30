#include "Semaphore.h"
#include "Util.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

//-----------------------------------------------------------
Semaphore::Semaphore( int initialCount )
    #if PLATFORM_IS_WINDOWS
    : _count( initialCount )
    #endif
{
    #if PLATFORM_IS_UNIX
        int r = sem_init( &_id, 0, initialCount );
        if( r != 0 )
            Fatal( "sem_init() failed." );
    #elif PLATFORM_IS_WINDOWS
        _id = CreateSemaphore( NULL,(LONG)initialCount, std::numeric_limits<LONG>::max(), NULL );
        if( _id == NULL )
            Fatal( "CreateSemaphore() failed." );
    #else
        #error Unimplemented
    #endif
}

//-----------------------------------------------------------
Semaphore::~Semaphore()
{
    #if PLATFORM_IS_UNIX
        int r = sem_destroy( &_id );
        if( r != 0 )
            Fatal( "sem_destroy() failed." );
    #elif PLATFORM_IS_WINDOWS
        if( !CloseHandle( _id ) )
            Fatal( "CloseHandle( semaphore ) failed." );
    #else
        #error Unimplemented
    #endif
}

//-----------------------------------------------------------
void Semaphore::Wait()
{
    #if PLATFORM_IS_UNIX
        const int r = sem_wait( &_id );
        if( r != 0 )
            Fatal( "sem_wait() failed." );    
    #elif PLATFORM_IS_WINDOWS
        const DWORD r = WaitForSingleObject( _id, INFINITE );
        if( r != WAIT_OBJECT_0 )
            Fatal( "PLATFORM_IS_WINDOWS( semaphore ) failed." );
    #else
        #error Unimplemented
    #endif
}

//-----------------------------------------------------------
bool Semaphore::Wait( long milliseconds )
{
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
            Fatal( "sem_wait() failed." );    

        return r == 0;
    #elif PLATFORM_IS_MACOS
        // #TODO: Implement on macOS with condition variable 
        return false;
    #elif PLATFORM_IS_WINDOWS
        const DWORD r = WaitForSingleObject( _id, (DWORD)milliseconds );
        return r == WAIT_OBJECT_0 || r == WAIT_TIMEOUT;
    #else
        #error Unimplemented
    #endif
}

//-----------------------------------------------------------
int Semaphore::GetCount()
{
    #if PLATFORM_IS_UNIX
        
        int value = 0;
        int r = sem_getvalue( &_id, &value );
        
        if( r != 0 )
            Fatal( "sem_getvalue() failed." );    

        return value;
    #elif PLATFORM_IS_WINDOWS
        return _count.load( std::memory_order::memory_order_release );
    #else
        #error Unimplemented
    #endif
}

//-----------------------------------------------------------
void Semaphore::Release()
{
    #if PLATFORM_IS_UNIX
        int r = sem_post( &_id );
        if( r != 0 )
            Fatal( "sem_post() failed." );
    #elif PLATFORM_IS_WINDOWS
        LONG prevCount;
        BOOL r = ::ReleaseSemaphore( _id, 1, &prevCount );
        
        if( !r )
            Fatal( "ReleaseSemaphore() failed." );

        _count++;
    #else
        #error Unimplemented
    #endif
}

#pragma GCC diagnostic pop

