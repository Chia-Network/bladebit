#pragma once
#include "threading/AutoResetSignal.h"
#include "util/Util.h"

class Fence
{
public:
    Fence();
    ~Fence();

    // inline uint32 Value() const { return _value.load( std::memory_order_acquire ); }
    inline uint32 Value() const { return _value; }

    // Should be only called by a single producer (ie. only 1 thread).
    void Signal();

    void Signal( uint32 value );

    // Should only be used when you know that the producer thread will not call Signal anymore.
    void Reset( uint32 value = 0 );

    // Wait until the fence is signalled with any value
    void Wait();

    // Wait until the fence is signalled with any value
    void Wait( Duration& accumulator );

    // Wait until the fence reaches or passes the specified value
    void Wait( uint32 value );

    // Wait until the fence reaches or passes the specified value
    void Wait( uint32 value, Duration& accumulator );

private:
    // std::atomic<uint32> _value;
    // #NOTE: Don't think we need atomic, since memory ordering ought to be enforced by the mutex.
    volatile uint32     _value = 0;
    AutoResetSignal     _signal;

};

class FencePool
{
public:
    //-----------------------------------------------------------
    inline
    FencePool( const uint32 capacity )
        : _fences( new Fence[capacity] )
        , _availableFences( new Fence*[capacity] )
        , _capacity( capacity )
    {
        ASSERT( capacity > 0 );

        RestoreAllFences();
    }

    //-----------------------------------------------------------
    inline ~FencePool()
    {
        delete[] _availableFences;
        delete[] _fences;
    }

    //-----------------------------------------------------------
    inline Fence* GetFence()
    {
        for( uint32 i = 0; i < _capacity; i++ )
        {
            if( _availableFences[i] != nullptr )
            {
                Fence* fence = _availableFences[i];
                _availableFences[i] = nullptr;
                fence->Reset( 0 );
                return fence;
            }
        }

        return nullptr;
    }

    //-----------------------------------------------------------
    inline Fence& RequireFence()
    {
        Fence* fence = GetFence();
        PanicIf( !fence, "No fences available in pool." );

        return *fence;
    }

    //-----------------------------------------------------------
    inline void ReleaseFence( Fence& fence )
    {
        #if _DEBUG
        {
            bool foundFence = false;
            for( uint32 i = 0; i < _capacity; i++ )
            {
                if( _fences+i == &fence )
                {
                    foundFence = true;
                    break;
                }
            }
            ASSERT( foundFence );

            for( uint32 i = 0; i < _capacity; i++ )
            {
                ASSERT( _availableFences[i] != &fence );
            }
        }
        #endif

        for( uint32 i = 0; i < _capacity; i++ )
        {
            if( _availableFences[i] == nullptr )
            {
                _availableFences[i] = &fence;
                return;
            }
        }

        ASSERT( 0 );
    }

    //-----------------------------------------------------------
    inline void RestoreAllFences()
    {
        for( uint32 i = 0; i < _capacity; i++ )
            _availableFences[i] = _fences+i;
    }

private:
    Fence*  _fences;
    Fence** _availableFences;
    uint32  _capacity;
};

