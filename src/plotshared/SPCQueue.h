#pragma once
#include "threading/AutoResetSignal.h"

// Single Producer-Consumer Queue
template<typename T, int Capacity>
class SPCQueue
{
    public:
    struct Iterator
    {
        int _iteration;     // One of potentially 2
        int _endIndex[2];

        bool Finished() const;
        void Next();
    };

    ProducerConsumerQueue();
    ~ProducerConsumerQueue();


    void Produce();
    T*   Consume();

    Iterator Begin() const;

    T& Get( const Iterator& iter );

    // Block and wait to be signaled that the
    // producer thread has added something to the buffer
    void WaitForProduction();

private:
    int              _writePosition;
    std::atomic<int> _count;
    T                _buffer[Capacity];
    AutoResetSignal  _signal;
};




