#pragma once

#include "Config.h"
#include "threading/ThreadPool.h"
#include "util/Util.h"
#include <cstring>
#if _DEBUG
    #include "util/Log.h"
#endif

#include <functional>

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

    // Locks the threads if this is the control thread and returns true.
    // otherwise, waits for release from the control thread and returns false.
    // inline bool LockOrWait();

    // Utility functions to simplify control-thread locking
    // and releasing code. This helps keep code the same for all threads
    // by locking only if it is the controls thread, otherwise
    // waiting for release.
    // BeginLockBlock() is the same as LockOrWait(). Only different for consistency in call name.
    // EndLockBlock() will do nothing if this is not the control thread.
    inline bool BeginLockBlock();
    inline void EndLockBlock();

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

template<typename F, typename... Args>
static void RunAnonymous( F&& f, Args&&... args );

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
    inline TJob& operator[]( int64  index ) { return this->_jobs[index]; }
    inline TJob& operator[]( uint index   ) { return this->_jobs[index]; }
    inline TJob& operator[]( int index    ) { return this->_jobs[index]; }

    inline TJob* Jobs() { return _jobs; }

private:
    static void RunJobWrapper( TJob* job );

private:
    TJob        _jobs[MaxJobs];
    ThreadPool& _pool;
};

struct AnonMTJob : public MTJob<AnonMTJob>
{
    std::function<void(AnonMTJob*)>* func;

    inline void Run() override { (*func)( this ); }

    // Run anononymous job, from a lambda, for example
    template<typename F,
        std::enable_if_t<
        std::is_invocable_r_v<void, F, AnonMTJob*>>* = nullptr>
    inline static void Run( ThreadPool& pool, const uint32 threadCount, F&& func )
    {
        std::function<void(AnonMTJob*)> f = func;
        
        MTJobRunner<AnonMTJob> jobs( pool );
        for( uint32 i = 0; i< threadCount; i++ )
        {
            auto& job = jobs[i];
            job.func = &f;
        }

        jobs.Run( threadCount );
    }

    template<typename F>
    inline static void Run( ThreadPool& pool, F&& func )
    {
        Run( pool, pool.ThreadCount(), func );
    }
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
// template<typename TJob>
// inline bool MTJobSyncT<TJob>::LockOrWait()
// {
//     if( this->IsControlThread() )
//     {
//         this->LockThreads();
//         return true;
//     }
//     else
//     {
//         this->WaitForRelease();
//     }

//     return false;
// }

//-----------------------------------------------------------
template<typename TJob>
inline bool MTJobSyncT<TJob>::BeginLockBlock()
{
    if( this->IsControlThread() )
    {
        this->LockThreads();
        return true;
    }

    return false;
}

//-----------------------------------------------------------
template<typename TJob>
inline void MTJobSyncT<TJob>::EndLockBlock()
{
    if( this->IsControlThread() )
        this->ReleaseThreads();
    else
        this->WaitForRelease();
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

    const TCount* counts;

    inline void CalculatePrefixSum(
        uint32        bucketSize,
        const TCount* counts,
        TCount*       pfxSum,
        TCount*       bucketCounts )
    {
        CalculatePrefixSumImpl<0>( bucketSize, counts, pfxSum, bucketCounts );
    }

    template<typename EntryType1>
    inline void CalculateBlockAlignedPrefixSum(
              uint32  bucketSize,
              uint32  blockSize,
        const TCount* counts,
              TCount* pfxSum,
              TCount* bucketCounts,
              TCount* offsets,
              TCount* alignedTotalCounts )
    {
        const uint32 entrySize       = (uint32)sizeof( EntryType1 );
        const uint32 entriesPerBlock = blockSize / entrySize;
        ASSERT( entriesPerBlock * entrySize == blockSize );

        CalculatePrefixSumImpl<1>( bucketSize, counts, pfxSum, bucketCounts, &entriesPerBlock, offsets, alignedTotalCounts );
    }

private:
    template<uint32 AlignEntryCount=0>
    inline void CalculatePrefixSumImpl(
              uint32  bucketSize,
        const TCount* counts,
              TCount* pfxSum,
              TCount* bucketCounts,
        const uint32* entriesPerBlocks   = nullptr,
              TCount* offsets            = nullptr,
              TCount* alignedTotalCounts = nullptr,
              TCount* pfxSum2            = nullptr
    );
};

//-----------------------------------------------------------
template<typename TJob, typename TCount>
template<uint32 AlignEntryCount>
inline void PrefixSumJob<TJob,TCount>::CalculatePrefixSumImpl(
        uint32  bucketSize,
  const TCount* counts,
        TCount* pfxSum,
        TCount* bucketCounts,
  const uint32* entriesPerBlocks,
        TCount* offsets,
        TCount* alignedTotalCounts,
        TCount* pfxSum2 )
{
    const uint32 jobId    = this->JobId();
    const uint32 jobCount = this->JobCount();

    this->counts = counts;
    this->SyncThreads();

    // Add up all of the jobs counts
    // Add-up all thread's bucket counts
    memset( pfxSum, 0, sizeof( TCount ) * bucketSize );
    
    for( uint32 i = 0; i < jobCount; i++ )
    {
        const TCount* tCounts = this->GetJob( i ).counts;

        for( uint32 j = 0; j < bucketSize; j++ )
            pfxSum[j] += tCounts[j];
    }

    // If we're the control thread, retain the total bucket count
    if( this->IsControlThread() && bucketCounts != nullptr )
    {
        memcpy( bucketCounts, pfxSum, sizeof( TCount ) * bucketSize );
    }

    uint32 alignedEntryIndex = 0;

    if constexpr ( AlignEntryCount > 0 )
    {
        // We now need to add padding to the total counts to ensure the starting
        // location of each slice is block aligned.
        const uint32 entriesPerBlock    = entriesPerBlocks[alignedEntryIndex++];
        const uint32 modEntriesPerBlock = entriesPerBlock - 1;

        for( uint32 i = bucketSize-1; i > 0; i-- )
        {
            // Round-up the previous bucket's entry count to be block aligned,
            // then we can add that as padding to this bucket's prefix count
            // ensuring the first entry of the slice falls onto the start of a block.
            const uint32 prevEntryCount        = pfxSum[i-1] + offsets[i-1];
            const uint32 prevAlignedEntryCount = CDivT( prevEntryCount, entriesPerBlock ) * entriesPerBlock;
            const uint32 paddingFromPrevBucket = prevAlignedEntryCount - prevEntryCount;

            // Calculate our next offset before updating our total count,
            // which is the # of entries that our last block occupies, if its not full.
            const uint32 offset            = offsets[i];
            const uint32 entryCount        = pfxSum[i] + offset;
            const uint32 alignedEntryCount = CDivT( entryCount, entriesPerBlock ) * entriesPerBlock;
            
            offsets[i] = ( entryCount - (alignedEntryCount - entriesPerBlock) ) & modEntriesPerBlock;  // Update our offset for the next round
            pfxSum[i] += paddingFromPrevBucket + offset;
            
            if( this->IsControlThread() )
                alignedTotalCounts[i] = alignedEntryCount;                                             // Set number of entries that have to be written to disk (always starts and ends at a block boundary)
        }

        // Add the offset to the first bucket slice as well
        pfxSum[0] += offsets[0];

        const uint32 b0AlignedCount = CDivT( pfxSum[0], entriesPerBlock ) * entriesPerBlock;

        offsets[0] = ( pfxSum[0] - (b0AlignedCount - entriesPerBlock) ) & modEntriesPerBlock;

        if( this->IsControlThread() )
            alignedTotalCounts[0] = b0AlignedCount;
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


//-----------------------------------------------------------
template<typename TCount = uint32>
struct AnonPrefixSumJob : public PrefixSumJob<AnonPrefixSumJob<TCount>, TCount>
{
    std::function<void(AnonPrefixSumJob*)>* func;

    inline void Run() override { (*func)( this ); }

    // Run anononymous job, from a lambda, for example
    template<typename F,
        std::enable_if_t<
        std::is_invocable_r_v<void, F, AnonPrefixSumJob<TCount>*>>* = nullptr>
    inline static void Run( ThreadPool& pool, const uint32 threadCount, F&& func )
    {
        std::function<void(AnonPrefixSumJob<TCount>*)> f = func;
        
        MTJobRunner<AnonPrefixSumJob<TCount>> jobs( pool );
        for( uint32 i = 0; i< threadCount; i++ )
        {
            auto& job = jobs[i];
            job.func = &f;
        }

        jobs.Run( threadCount );
    }

    template<typename F>
    inline static void Run( ThreadPool& pool, F&& func )
    {
        Run( pool, pool.ThreadCount(), func );
    }
};

//-----------------------------------------------------------
template<typename T>
inline void GetThreadOffsets( const uint32 id, const uint32 threadCount, const T totalCount, T& count, T& offset, T& end )
{
    const T countPerThread = totalCount / (T)threadCount;
    const T remainder      = totalCount - countPerThread * (T)threadCount;

    count  = countPerThread;
    offset = (T)id * countPerThread;

    if( id == threadCount-1 )
        count += remainder;
    
    end = offset + count;
}

//-----------------------------------------------------------
template<typename TJob, typename T>
inline void GetThreadOffsets( MTJob<TJob>* job, const T totalCount, T& count, T& offset, T& end )
{
    GetThreadOffsets( job->JobId(), job->JobCount(), totalCount, count, offset, end );
}

