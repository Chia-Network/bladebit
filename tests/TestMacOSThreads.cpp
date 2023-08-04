#include "TestUtil.h"
#include "threading/ThreadPool.h"

//-----------------------------------------------------------
TEST_CASE( "macos-threads", "[sandbox]" )
{
    std::atomic<bool> signal = false;

    Thread* thread = new Thread();

    ThreadRunner tfunc = (ThreadRunner)[]( void* p ) -> void {
        Thread::Sleep( 1500 );
        reinterpret_cast<std::atomic<bool>*>( p )->store( true, std::memory_order_release );
    };
    thread->Run( tfunc, &signal );

    ENSURE( thread->WaitForExit( 100 ) == false );
    ENSURE( thread->WaitForExit() );
    ENSURE( thread->HasExited() );

    delete thread;
}