#pragma once
#include <mutex>
#include <queue>

/// Lock-based multi-producer, multi-consumer queue
/// Simple and good enough for most uses
template<typename T>
class MPMCQueue
{
public:
    inline MPMCQueue() {}

    void Enqueue( const T& item )
    {
        _mutex.lock();
        _queue.push( item );
        _mutex.unlock();
    }

    void Enqueue( const T* items, const size_t count )
    {
        if( count < 1 )
            return;

        _mutex.lock();

        for( size_t i = 0; i < count; i++ )
            _queue.push( items[i] );

        _mutex.unlock();
    }

    size_t Dequeue( T* outItem, const size_t maxDequeue )
    {
        _mutex.lock();

        const size_t max = std::min( maxDequeue, _queue.size() );

        for( size_t i = 0; i < max; i++ )
        {
            outItem[i] = _queue.front();
            _queue.pop();
        }
        _mutex.unlock();

        return max;
    }

    bool Dequeue( T* outItem )
    {
        _mutex.lock();
        const bool hasItem = !_queue.empty();

        if( hasItem )
        {
            *outItem = _queue.front();
            _queue.pop();
        }
        _mutex.unlock();

        return hasItem;
    }

private:
    std::mutex    _mutex;
    std::queue<T> _queue;
};

