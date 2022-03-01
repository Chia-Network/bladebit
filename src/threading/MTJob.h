#pragma once

#include "Config.h"
#include "threading/ThreadPool.h"
#include <cstring>
#if _DEBUG
    #include "util/Log.h"
#endif

template<typename TJob, uint MaxJobs>
struct MTJobRunner;

template<typename TJob>
struct MTJobSyncT
{
    std::atomic<uint>* _finishedCount;
    std::atomic<uint>* _releaseLock;
    uint               _jobId;
    uint               _jobCount;
    TJob*              _jobs;

    // Synchronize all threads before continuing to the next step
    inline void SyncThreads();

    // Lock Threads & to perform something before continuing to the next parallel step.
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

    // Reduce the _jobCount to the specified amount.
    // This is useful when a job needs to continue
    // from its Run() function, but it needs to complete
    // with less threads than it originally started with.
    inline bool ReduceThreadCount( uint newThreadCount );

    inline uint JobId()    const { return _jobId; }
    inline uint JobCount() const { return _jobCount; }

    inline bool IsControlThread() const { return _jobId == 0; }
    inline bool IsLastThread()    const { return _jobId == _jobCount-1; }

    inline const TJob& GetJob( uint index ) const
    {
        ASSERT( index < _jobCount );
        return _jobs[index];
    };

    inline const TJob& GetJob( int index ) const
    {   
        ASSERT( index >= 0 );
        ASSERT( (uint)index < _jobCount );
        return _jobs[index];
    };

    inline const TJob& LastJob() const { return _jobs[_jobCount-1]; }

    // For debugging
    #if _DEBUG
        inline void Trace( const char* msg, ... );
    #else
        inline void Trace( const char* msg, ... ) {}
    #endif
};

struct MTJobSync : public MTJobSyncT<MTJobSync> {};

template<typename TJob>
struct MTJob : public MTJobSyncT<TJob>
{
    inline virtual ~MTJob() {}

    template<typename,uint>
    friend struct MTJobRunner;

    // Override in the job itself
    virtual void Run() = 0;
};


template<typename TJob, uint MaxJobs = BB_MAX_JOBS>
struct MTJobRunner
{
    MTJobRunner( ThreadPool& pool );

    double Run();
    double Run( uint32 threadCount );

    inline TJob& operator[]( uint64 index ) { return this->_jobs[index]; }
    inline TJob& operator[]( uint index   ) { return this->_jobs[index]; }
    inline TJob& operator[]( int index    ) { return this->_jobs[index]; }

    inline TJob* Jobs() { return _jobs; }

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
    return this->Run( this->_pool.ThreadCount() );
}

template<typename TJob, uint MaxJobs>
inline double MTJobRunner<TJob, MaxJobs>::Run( uint32 threadCount )
{
    // Set thread ids and atomic locks
    ASSERT( threadCount <= MaxJobs );

    std::atomic<uint> finishedCount = 0;
    std::atomic<uint> releaseLock   = 0;
    
    for( uint i = 0; i < threadCount; i++ )
    {
        MTJob<TJob>& job = *static_cast<MTJob<TJob>*>( &_jobs[i] );

        job._finishedCount = &finishedCount;
        job._releaseLock   = &releaseLock;
        job._jobId         = i;
        job._jobCount      = threadCount;
        job._jobs          = _jobs;
    }

    // Run the job
    const auto timer = TimerBegin();
    _pool.RunJob( RunJobWrapper, _jobs, threadCount );
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
inline void MTJobSyncT<TJob>::SyncThreads()
{
    if( LockThreads() )
        ReleaseThreads();
    else
        WaitForRelease();
}

template<typename TJob>
inline bool MTJobSyncT<TJob>::LockThreads()
{
    if( this->_jobId == 0 )
    {
        ASSERT( _jobId == 0 );

        auto& finishedCount        = *this->_finishedCount;
        const uint threadThreshold = this->_jobCount - 1;

        // Trace( "Locking Threads..." );
        // Wait for all threads to finish
        while( finishedCount.load( std::memory_order_relaxed ) != threadThreshold );
        // Trace( "---- All threads Locked ----." );
        return true;
    }

    return false;
}

template<typename TJob>
inline void MTJobSyncT<TJob>::ReleaseThreads()
{
    ASSERT( _jobId == 0 );

    auto& finishedCount        = *this->_finishedCount;
    auto& releaseLock          = *this->_releaseLock;
    
    // Trace( "Releasing threads..." );
    // Release lock & signal other threads
    // Trace( "++++ Releasing all threads ++++" );
    releaseLock  .store( 0, std::memory_order_release );
    finishedCount.store( 0, std::memory_order_release );
}

template<typename TJob>
inline void MTJobSyncT<TJob>::WaitForRelease()
{
    ASSERT( _jobId != 0 );

    auto& finishedCount        = *this->_finishedCount;
    auto& releaseLock          = *this->_releaseLock;
    const uint threadThreshold = this->_jobCount - 1;

    // Signal the control thread that we're ready to sync
    // uint count = finishedCount.load( std::memory_order_acquire );
    finishedCount++;
    ASSERT( finishedCount <= threadThreshold );
    // while( !finishedCount.compare_exchange_weak( count, count+1, std::memory_order_release, std::memory_order_relaxed ) );
    // Trace( "- locked: %d", count );
    
    // Wait for the control thread (id == 0 ) to signal us
    while( finishedCount.load( std::memory_order_relaxed ) != 0 );

    // Ensure all threads have been released (prevent re-locking before another thread has been released)
    // count = releaseLock.load( std::memory_order_acquire );
    // while( !releaseLock.compare_exchange_weak( count, count+1, std::memory_order_release, std::memory_order_relaxed ) );
    ASSERT( releaseLock <= threadThreshold );
    releaseLock++;
    while( releaseLock.load( std::memory_order_relaxed ) != threadThreshold );
    // Trace( " released: %d", count );
}

//-----------------------------------------------------------
template<typename TJob>
inline bool MTJobSyncT<TJob>::ReduceThreadCount( uint newThreadCount )
{
    ASSERT( newThreadCount < _jobCount );
    ASSERT( newThreadCount >= 0 );

    // Does this thread need to synchronize?
    // If not, don't participate
    if( _jobId >= newThreadCount )
        return false;
    
    // Update our job count
    this->_jobCount = newThreadCount;

    // Now synchronize
    this->SyncThreads();

    return true;
}


#if _DEBUG
//-----------------------------------------------------------
template<typename TJob>
inline void MTJobSyncT<TJob>::Trace( const char* msg, ... )
{
    const size_t BUF_SIZE = 512;
    char buf1[BUF_SIZE];
    char buf2[BUF_SIZE];
    
    va_list args;
    va_start( args, msg );
    int r = vsnprintf( buf1, BUF_SIZE, msg, args );
    va_end( args );
    ASSERT( r > 0 );

    r = snprintf( buf2, BUF_SIZE, "[%2u]: %s\n", _jobId, buf1 );
    ASSERT( r > 0 );

    const size_t size = std::min( BUF_SIZE, (size_t)r );
    Log::SafeWrite( buf2, size );
}

#endif

/// Helper Job that calculates a prefix sum
template<typename TJob, typename TCount = uint32>
struct PrefixSumJob : public MTJob<TJob>
{
    inline virtual ~PrefixSumJob() {}

    TCount* counts;

    inline void CalculatePrefixSum(
        uint32  bucketSize,
        TCount* counts,
        TCount* pfxSum,
        TCount* bucketCounts );
};

//-----------------------------------------------------------
template<typename TJob, typename TCount>
inline void PrefixSumJob<TJob,TCount>::CalculatePrefixSum(
        uint32  bucketSize,
        TCount* counts,
        TCount* pfxSum,
        TCount* bucketCounts )
{
    const uint32 jobId    = this->JobId();
    const uint32 jobCount = this->JobCount();

    this->counts = counts;
    this->SyncThreads();

    // Add up all of the jobs counts
    memset( pfxSum, 0, sizeof( TCount ) * bucketSize );

    for( uint32 i = 0; i < jobCount; i++ )
    {
        const TCount* tCounts = this->GetJob( i ).counts;

        for( uint32 j = 0; j < bucketSize; j++ )
            pfxSum[j] += tCounts[j];
    }

    // If we're the control thread, retain the total bucket count
    if( this->IsControlThread() )
    {
        memcpy( bucketCounts, pfxSum, sizeof( TCount ) * bucketSize );
    }

    // Calculate the prefix sum
    for( uint32 i = 1; i < bucketSize; i++ )
        pfxSum[i] += pfxSum[i-1];

    // Subtract the count from all threads after ours 
    // to get the correct prefix sum for this thread
    for( uint32 t = jobId+1; t < jobCount; t++ )
    {
        const TCount* tCounts = this->GetJob( t ).counts;

        for( uint32 i = 0; i < bucketSize; i++ )
            pfxSum[i] -= tCounts[i];
    }
}

