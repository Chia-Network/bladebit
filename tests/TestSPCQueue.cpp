#include "TestUtil.h"
#include "SysHost.h"
#include "util/SPCQueue.h"
#include "threading/AutoResetSignal.h"
#include "threading/MTJob.h"

static void ProducerThread( void* param = nullptr );
static void ConsumerThread( void* param = nullptr );

namespace {

    static AutoResetSignal       _signal;
    static GrowableSPCQueue<int> _queue;
    static std::atomic<bool>     _exitThread = false;
}

//-----------------------------------------------------------
TEST_CASE( "spc-queue", "[unit-core]" )
{
    Thread producer;

    producer.Run( ProducerThread, nullptr );

    ConsumerThread();

    producer.WaitForExit();
}

//-----------------------------------------------------------
void ProducerThread( void* param )
{
    const int MAX_COUNT  = std::numeric_limits<int>::max();
    const int BATCH_SIZE = 16;

    int j;
    for( int i = 0; i < MAX_COUNT; i += j )
    {
        const int max = std::min( BATCH_SIZE, MAX_COUNT-i );
        for( j = 0; j < max; j++ )
        {
            const int v = i+j;
            if( !_queue.Enqueue( v ) )
                break;
        }

        _queue.Commit();
        _signal.Signal();
    }

    _exitThread = true;
    _signal.Signal();
}

//-----------------------------------------------------------
void ConsumerThread( void* param )
{
    const int INTERVAL = 100000000;
    const int CAPACITY = 64;
    int buffer[CAPACITY];

    int next = 0;

    for( ;; )
    {
        _signal.Wait();

        // Take all entries until there's no more
        for( ;; )
        {
            const int count = _queue.Dequeue( buffer, CAPACITY );
            ENSURE( count <= CAPACITY );

            if( count == 0 )
            {
                if( _exitThread )
                    return;

                break;
            }

            for( int i = 0; i < count; i++ )
            {
                ENSURE( buffer[i] == next + i );
            }

            next += count;
            if( next % INTERVAL == 0 )
                Log::Line( "Count: %d", next );
        }
    }
}

