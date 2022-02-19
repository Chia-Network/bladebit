#pragma once
#include "threading/AutoResetSignal.h"
#include "util/Util.h"

class SPCQueueIterator
{
    template<typename T, int Capacity>
    friend class SPCQueue;

    int _iteration;     // One of potentially 2.
    int _endIndex[2];

    bool HasValues() const;
    void Next();
};

// Statically Fixed-Size Single Producer-Consumer Queue
template<typename T, int Capacity>
class SPCQueue
{
public:
    SPCQueue();
    ~SPCQueue();

    
    // Adds a pending value to the queue and returns it for the user to write,
    // but it does not commit it yet (makes it visible for reading).
    // This allows you to enqueue multiple commands before committing/publishing them.
    // Call Commit() after a successful call to Write().
    // If this call does not return true, do NOTE call Commit().
    bool Write( T*& outValue );

    // Publish pending commands to be visible for reading.
    void Commit();

    // Does Write() and Commit() in a single call.
    // Returns false if there's currently no space available in the queue.
    bool Enqueue( const T& value );

    int Dequeue( T* values, int capacity );

    int Count() const;

private:
    int              _writePosition  = 0;   // Current write position in our buffer.
    int              _pendingCount   = 0;   // Entries pending to be committed.
    std::atomic<int> _committedCount = 0;   // Current number of committed entries
    int              _readPosition   = 0;
    T                _buffer[Capacity];
};


// Dynamically-Sized Single Producer-Consumer Queue
template<typename T, size_t _growSize = 64>
class GrowableSPCQueue
{
public:

    explicit GrowableSPCQueue(size_t capacity );
    GrowableSPCQueue();
    ~GrowableSPCQueue();

    // Grab an entry from the heap buffer to encode
    // the item into the queue, but not commit it for consumption yet.
    // Use Commit() to publish the uncommitted (written) commands presently in the queue.
    bool Write( T*& outValue );

    // Publish pending commands to be visible for reading.
    void Commit();

    // Does Write() and Commit() in a single call.
    bool Enqueue( const T& value );

    int Dequeue( T* values, int capacity );

    // Returns the amount of entries currently comitted for dequeuing.
    // The value returned here may or may not be accurate
    // as a thread may have enqueued or dequeued entries immediately
    // before/after the comitted counts are retrieved.
    // This should only be called from the consumer or the producer thread.
    // Calling it from any other thread is underfined behavior.
    // #TODO: Not sure if we can actually implement this in a safe manner,
    //        even from the consumer or producer thread.
    // inline int64 Count() const
    // {
    //     int64 producerCount = _producerState->committedCount.load( std::memory_order_relaxed );

    //     const auto* oldState = _newState.load( std::memory_order_relaxed );
    //     if( oldState )
    //         producerCount += oldState->committedCount.load( std::memory_order_relaxed );

    //     return producerCount;
    // }

private:

    struct ConsumerState
    {
        T*               buffer;         // Element buffer
        int              capacity;       // Buffer capacity
        std::atomic<int> committedCount; // How many elements are ready to be read
    };


private:

    // Producer
    int              _writePosition     = 0;     // Current write position in our buffer.
    int              _pendingCount      = 0;     // Elements not yet committed (published) to the consumer thread
    int              _oldPendingCount   = 0;     // Pending count before we resized the buffer and switched to a new state
    ConsumerState*   _producerState;
    ConsumerState    _states[2];
    int              _nextState         = 1;
    bool             _pendingState      = false; // Avoid atomic _newState check during Commit()

    // Shared
    std::atomic<ConsumerState*> _newState = nullptr;

    // Consumer
    ConsumerState*   _consumerState;
    int              _readPosition      = 0;
};


#include "SPCQueue.inl"



