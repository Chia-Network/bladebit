#include "../../threading/Thread.h"
#include "../../Util.h"
#include "../../Globals.h"
#include "SysHost.h"
#include "util/Log.h"

//-----------------------------------------------------------
Thread::Thread( size_t stackSize )
{
    // Configure stack size
    if( stackSize < 1024 * 4 )
        Fatal( "Thread stack size is too small." );
    
    _state.store( ThreadState::ReadyToRun, std::memory_order_release );

    _threadId = CreateThread(
                    NULL,
                    (SIZE_T)stackSize,
                    Thread::ThreadStarterWin,
                    (LPVOID)this,
                    CREATE_SUSPENDED,
                    nullptr );

    if( _threadId == NULL )
        Fatal( "Failed to create thread." );
}

//-----------------------------------------------------------
Thread::Thread() : Thread( 8 MB ) {}

//-----------------------------------------------------------
Thread::~Thread()
{
    const bool didExit = _state.load( std::memory_order_relaxed ) == ThreadState::Exited;

    if( !didExit )
    {
        // Thread should have exited already, terminate it abruptly
        Log::Error( "Warning: Thread did not exit properly." );
        TerminateThread( _threadId, 0 );
    }

    if( !CloseHandle( _threadId ) )
        Log::Error( "An error occurred when closing a thread." );

    _threadId = NULL;
}

//-----------------------------------------------------------
bool Thread::HasExited() const
{
    return _state.load( std::memory_order_relaxed ) == ThreadState::Exited;
}

//-----------------------------------------------------------
void Thread::Run( ThreadRunner runner, void* param )
{
    // Ensure we can actually start
    ASSERT( runner );
    const bool canRun = _state.load( std::memory_order_relaxed ) == ThreadState::ReadyToRun;

    ASSERT( canRun );
    if( !canRun )
        return;

    // Change the thread state
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

    // Start the thread
    const DWORD r = ResumeThread( _threadId );
    if( r != 1 )
        Fatal( "Failed to resume thread %d.", _threadId );
}


//-----------------------------------------------------------
void Thread::Sleep( long milliseconds )
{
    ::Sleep( (DWORD)milliseconds );
}

//-----------------------------------------------------------
bool Thread::WaitForExit( long milliseconds )
{
    ThreadState state = _state.load( std::memory_order_relaxed );
    if( state == ThreadState::Exited )
        return true;
    
    const DWORD r = WaitForSingleObject( _threadId, (DWORD)milliseconds );
    if( r != WAIT_TIMEOUT && r != WAIT_OBJECT_0 )
    {
        Log::Error( "Thread %d wait for exit failed: %d", _threadId, GetLastError() );
        return false;
    }

    return true;
}

// Thread entry point
//-----------------------------------------------------------
DWORD Thread::ThreadStarterWin( LPVOID param )
{
    ASSERT( param );

    Thread* t = (Thread*)param;
    ASSERT( t->_state == ThreadState::Running );

    // Run the thread function
    t->_runner( t->_runParam );

    // Thread has exited
    t->_state.store( ThreadState::Exited, std::memory_order_release );
    
    // TODO: Signal if waiting to be joined
    return 0;
}

