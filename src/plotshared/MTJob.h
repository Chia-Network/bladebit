#pragma once

#include "Config.h"
#include "threading/ThreadPool.h"

template<typename TJob>
struct MTJob
{
    // Override in the job itself
    virtual void Run() = 0;

    // Synchronize all threads before continuing to the next step
    inline void SyncThreads();

    // Lock Threads & to perform something before continueing ot the next parallel step.
    // Returns true if the job id is 0, that is, the control thread.
    inline bool LockThreads();

    // Only to be called from the control thread.
    // Releases threads that are waiting for the control
    // thread to do some synchronized work.
    inline void ReleaseThreads();

    // Use this if LockThreads returns false.
    // This is to be used by the non-control threads
    // to wait for the control thread to release the lock on us.
    inline void WaitForRelease();

    inline uint JobId()    const { return _jobId; }
    inline uint JobCount() const { return _jobCount; }

protected:
    std::atomic<uint>* _finishedCount;
    std::atomic<uint>* _releaseLock;
    uint               _jobId;
    uint               _jobCount;
};


template<typename TJob, uint MaxJobs = BB_MAX_JOBS>
struct MTJobRunner
{
    MTJobRunner( ThreadPool& pool );

    double Run();

    inline TJob& operator[]( uint index ) const { return this->_jobs[index]; }
    inline TJob& operator[]( int index )  const { return this->_jobs[index]; }

private:
    static void RunJobWrapper( TJob* job );

private:
    TJob        _jobs[MaxJobs];
    ThreadPool& _pool;
};



template<typename TJob, uint MaxJobs>
inline MTJobRunner<TJob, MaxJobs>::MTJobRunner( ThreadPool& pool )
    : _pool( pool )
{}

template<typename TJob, uint MaxJobs>
inline double MTJobRunner<TJob, MaxJobs>::Run()
{
    // Set thread ids and atomic locks
    const uint threadCount = _pool.ThreadCount();
    ASSERT( _pool.ThreadCount() <= MaxJobs );

    std::atomic<uint> finishedCount = 0;
    std::atomic<uint> releaseLock   = 0;
    
    for( uint i = 0; i < threadCount; i++ )
    {
        MTJob<TJob>& job = *static_cast<MTJob<TJob>*>( &_jobs[i] );

        job._finishedCount = &finishedCount;
        job._releaseLock   = &releaseLock;
        job._jobId         = i;
        job._jobCount      = threadCount;
    }

    // Run the job
    const auto timer = TimerBegin();
    _pool->RunJob( RunJobWrapper, _jobs, threadCount );
    const double elapsed = TimerEnd( timer );

    return elapsed;
}

template<typename TJob, uint MaxJobs>
inline void MTJobRunner<TJob, MaxJobs>::RunJobWrapper( TJob* job )
{
    //job->Run();
    static_cast<MTJob<TJob>*>( job )->Run();
}


template<typename TJob>
inline void MTJob<TJob>::SyncThreads()
{
    auto& finishedCount        = *this->finishedCount;
    auto& releaseLock          = *this->releaseLock;
    const uint threadThreshold = this->threadCount - 1;

    // Are we the control thread?
    if( id == 0 ) 
    {
        // Wait for all threads to finish
        while( finishedCount.load( std::memory_order_relaxed ) != threadThreshold );

        // Release lock & signal other threads
        releaseLock  .store( 0, std::memory_order_release );
        finishedCount.store( 0, std::memory_order_release );
    }
    else
    {
        // Signal we're ready to sync
        uint count = finishedCount.load( std::memory_order_acquire );
        while( !finishedCount.compare_exchange_weak( count, count+1, std::memory_order_release, std::memory_order_relaxed ) );
        
        // Wait for the control thread (id == 0 ) to signal us
        while( finishedCount.load( std::memory_order_relaxed ) != 0 );

        // Ensure all threads have been released
        count = releaseLock.load( std::memory_order_acquire );
        while( !releaseLock.compare_exchange_weak( count, count+1, std::memory_order_release, std::memory_order_relaxed ) );
        while( releaseLock.load( std::memory_order_relaxed ) != threadThreshold );
    }
}

template<typename TJob>
inline bool MTJob<TJob>::LockThreads()
{
    if( this->_jobId == 0 )
    {
        auto& finishedCount        = *this->finishedCount;
        const uint threadThreshold = this->threadCount - 1;
    
        // Wait for all threads to finish
        while( finishedCount.load( std::memory_order_relaxed ) != threadThreshold );
    }

    return false;
}

template<typename TJob>
inline void MTJob<TJob>::ReleaseThreads()
{}

template<typename TJob>
inline void MTJob<TJob>::WaitForRelease()
{}
