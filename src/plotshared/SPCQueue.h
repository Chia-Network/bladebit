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
    ProducerConsumerQueue();
    ~ProducerConsumerQueue();

    void Enqueue( const T& value );
    int Dequeue( T* values, int capacity );


    //Iterator Begin() const;
    //void End( const Iterator& iter );

    //T& Get( const SPCQueueIterator& iter );

    // Block and wait to be signaled that the
    // producer thread has added something to the buffer
    //void WaitForProduction();

    int Count() const;

private:
    int              _writePosition;
    std::atomic<int> _count;
    T                _buffer[Capacity];
    //AutoResetSignal  _signal;
};




