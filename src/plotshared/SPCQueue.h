#pragma once
#include "threading/AutoResetSignal.h"

class SPCQueueIterator
{
    template<typename T, int Capacity>
    friend class SPCQueue;

    int _iteration;     // One of potentially 2.
    int _endIndex[2];

    bool HasValues() const;
    void Next();
};

// Single Producer-Consumer Queue
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
    T                _buffer[Capacity];
};

#include "SPCQueue.inl"



