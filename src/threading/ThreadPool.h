#pragma once

#include "./Semaphore.h"
#include "Thread.h"
#include <atomic>

typedef void (*JobFunc)( void* data );

///
/// Used for running parallel jobs.
///
class ThreadPool
{
public:
    enum class Mode
    {
        Fixed  = 0, // Only can run as many jobs as we have threads.
        Greedy = 1  // Can run more jobs that we have threads.
                    // Threads will continue to process jobs as long 
                    // as there are jobs available.
    };

    ThreadPool( uint threadCount, Mode mode = Mode::Fixed, bool disableAffinity = false );
    ~ThreadPool();

    void RunJob( JobFunc func, void* data, uint count, size_t dataSize );

    template<typename T>
    inline void RunJob( void (*TJobFunc)( T* ), T* data, uint count );

    inline uint ThreadCount() { return _threadCount; }
private:

    void DispatchFixed( JobFunc func, byte* data, uint count, size_t dataSize );
    void DispatchGreedy( JobFunc func, byte* data, uint count, size_t dataSize );

    static void FixedThreadRunner( void* tParam );
    static void GreedyThreadRunner( void* tParam );

    struct ThreadData
    {
        ThreadPool* pool;
        int         index;
        uint        cpuId;     // CPU Id affinity
        Semaphore   jobSignal; // Used for fixed mode
    };

private:
    uint              _threadCount;         // Reserved number of thread running jobs
    Mode              _mode;
    bool              _disableAffinity;
    Thread*           _threads;
    ThreadData*       _threadData;
    Semaphore         _jobSignal;           // Used to signal threads that there's a new job
    Semaphore         _poolSignal;          // Used to signal the pool that a thread has finished its job
    std::atomic<bool> _exitSignal = false;  // Used to signal threads to exit


    // Current job group
    std::atomic<uint> _jobIndex    = 0;            // Next jobi index
    uint              _jobCount    = 0;            // Current number of jobs
    JobFunc           _jobFunc     = nullptr;
    byte*             _jobData     = nullptr;
    size_t            _jobDataSize = 0;
};


//-----------------------------------------------------------
template<typename T>
inline void ThreadPool::RunJob( void (*TJobFunc)( T* ), T* data, uint count )
{
    RunJob( (JobFunc)TJobFunc, data, count, sizeof( T ) );
}