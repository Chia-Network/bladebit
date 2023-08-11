#pragma once
#include "MPMCQueue.h"
#include "threading/Thread.h"
#include "threading/AutoResetSignal.h"
#include "util/Span.h"
#include "util/Util.h"

/// Multi-producer command queue base class
template<typename TCommand, uint32 _MaxDequeue = 32>
class MPCommandQueue
{
    using TSelf = MPCommandQueue<TCommand, _MaxDequeue>;

    enum State : uint32
    {
        Default = 0,
        Running,
        Exiting,
    };

public:
    MPCommandQueue() {}

    virtual inline ~MPCommandQueue()
    {
        _state.store( Exiting, std::memory_order_release );
        _consumerSignal.Signal();
        _consumerThread.WaitForExit();
    }

    void StartConsumer()
    {
        PanicIf( _state.load( std::memory_order_relaxed ) != Default, "Unexpected state" );

        State expectedState = Default;
        if( !_state.compare_exchange_weak( expectedState, Running,
                                           std::memory_order_release,
                                           std::memory_order_relaxed ) )
        {
            Panic( "Unexpected state %u.", expectedState );
        }

        _consumerThread.Run( ConsumerThreadMain , this );
    }

    /// Thread-safe
    void Submit( const TCommand& cmd )
    {
        Submit( &cmd, 1 );
    }

    void Submit( const TCommand* commands, const i32 count )
    {
        ASSERT( commands );
        ASSERT( count > 0 );

        _queue.Enqueue( commands, (size_t)count );
        _consumerSignal.Signal();
    }

protected:
    /// Implementors must implement this
    virtual void ProcessCommands( const Span<TCommand> items ) = 0;

    /// Command thread
    static void ConsumerThreadMain( TSelf* self )
    {
        self->ConsumerThread();
    }

    void ConsumerThread()
    {
        TCommand items[_MaxDequeue] = {};

        for( ;; )
        {
            _consumerSignal.Wait();

            if( _state.load( std::memory_order_relaxed ) == Exiting )
                break;

            const size_t itemCount = _queue.Dequeue( items, _MaxDequeue );

            if( itemCount > 0 )
                this->ProcessCommands( Span<TCommand>( items, itemCount ) );
        }
    }

private:
    MPMCQueue<TCommand> _queue;
    Thread              _consumerThread;
    AutoResetSignal     _consumerSignal;
    std::atomic<State>  _state = Default;
};

