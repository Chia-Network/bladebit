#include "ThreadPool.h"
#include "Util.h"
#include "util/Log.h"
#include "SysHost.h"


//-----------------------------------------------------------
ThreadPool::ThreadPool( uint threadCount, Mode mode, bool disableAffinity )
    : _threadCount( threadCount )
    , _mode           ( mode )
    , _disableAffinity( disableAffinity )
    , _jobSignal      ( 0 )
    , _poolSignal     ( 0 )
{
    if( threadCount < 1 )
        Fatal( "threadCount must be greater than 0." );
    
    _threads    = new Thread    [threadCount];
    _threadData = new ThreadData[threadCount];

    auto threadRunner = mode == Mode::Fixed ? FixedThreadRunner : GreedyThreadRunner;

    for( uint i = 0; i < threadCount; i++ )
    {
        _threadData[i].index = (int)i;
        _threadData[i].cpuId = i;
        _threadData[i].pool  = this;
        
        Thread& t = _threads[i];

        t.Run( threadRunner, &_threadData[i] );
    }
}

//-----------------------------------------------------------
ThreadPool::~ThreadPool()
{
    // Signal
    _exitSignal.store( true, std::memory_order_release );

    if( _mode == Mode::Fixed )
    {
        for( uint i = 0; i < _threadCount; i++ )
            _threadData[i].jobSignal.Release();
    }
    else
    {
        for( uint i = 0; i < _threadCount; i++ )
            _jobSignal.Release();
    }

    // #TODO: Wait for all threads to finish
    
    // #TODO: Signal thread for exit.
    // #TODO: Wait for all threads to exit

    delete[] _threads;
    delete[] _threadData;
    
    _threads    = nullptr;
    _threadData = nullptr;
}

//-----------------------------------------------------------
void ThreadPool::RunJob( JobFunc func, void* data, uint count, size_t dataSize )
{
    ASSERT( func     );
    ASSERT( data     );
    ASSERT( dataSize );

    // #TODO: Should lock here to prevent re-entrancy and wait
    //        until current jobs are finished, but that is not the intended usage.
    if( _mode == Mode::Fixed )
        DispatchFixed( func, (byte*)data, count, dataSize );
    else
        DispatchGreedy( func, (byte*)data, count, dataSize );
}

//-----------------------------------------------------------
void ThreadPool::DispatchFixed( JobFunc func, byte* data, uint count, size_t dataSize )
{
    _jobFunc     = func;
    _jobData     = (byte*)data;
    _jobDataSize = dataSize;

    ASSERT( count <= _threadCount );

    if( count > _threadCount )
        count = _threadCount;

    for( uint i = 0; i < count; i++ )
        _threadData[i].jobSignal.Release();

    // Wait until all running jobs finish
    uint releaseCount = 0;
    while( releaseCount < count )
    {
        _poolSignal.Wait();
        releaseCount++;
    }
}

//-----------------------------------------------------------
void ThreadPool::DispatchGreedy( JobFunc func, byte* data, uint count, size_t dataSize )
{
    // No jobs should currently be running
    ASSERT( _jobSignal.GetCount() == 0 );
    ASSERT( count );

    _jobCount    = count;
    _jobFunc     = func;
    _jobData     = (byte*)data;
    _jobDataSize = dataSize;
    _jobIndex.store( 0, std::memory_order_release );

    ASSERT( _poolSignal.GetCount() == 0 );

    // Signal release the job semaphore <coun> amount of times.
    // The job threads will grab jobs from the pool as long as there is one.
    for( uint i = 0; i < count; i++ )
        _jobSignal.Release();

    // Wait until all running jobs finish
    uint releaseCount = 0;
    while( releaseCount < count )
    {
        _poolSignal.Wait();
        releaseCount++;
    }

    ASSERT( _jobIndex == count );
    ASSERT( _poolSignal.GetCount() == 0 );

    // All jobs finished
    _jobFunc  = nullptr;
    _jobData  = nullptr;
    _jobCount = 0;
}

//-----------------------------------------------------------
void ThreadPool::FixedThreadRunner( void* tParam )
{
    ASSERT( tParam );
    ThreadData& d    = *(ThreadData*)tParam;
    ThreadPool& pool = *d.pool;

    if( !pool._disableAffinity )
        SysHost::SetCurrentThreadAffinityCpuId( d.cpuId );

    const uint index = (uint)d.index;

    std::atomic<bool>& exitSignal = pool._exitSignal;
    Semaphore&         poolSignal = pool._poolSignal;
    Semaphore&         jobSignal  = d.jobSignal;

    for( ;; )
    {
        if( exitSignal.load( std::memory_order::memory_order_acquire ) )
            return;

        // Wait until we are signalled to go
        jobSignal.Wait();

        // We may have been signalled to exit
        if( exitSignal.load( std::memory_order_acquire ) )
            return;
        
        // Run job
        pool._jobFunc( pool._jobData + pool._jobDataSize * index );

        // Finished job
        poolSignal.Release();
    }
}

//-----------------------------------------------------------
void ThreadPool::GreedyThreadRunner( void* tParam )
{
    ASSERT( tParam );

    ThreadData& d    = *(ThreadData*)tParam;
    ThreadPool& pool = *d.pool;

    if( !pool._disableAffinity )
        SysHost::SetCurrentThreadAffinityCpuId( d.cpuId );

    for( ;; )
    {
        if( pool._exitSignal.load( std::memory_order::memory_order_acquire ) )
            return;

        // Wait until we are signalled to go
        pool._jobSignal.Wait();

        // We may have been signalled to exit
        if( pool._exitSignal.load( std::memory_order_acquire ) )
            return;

        // Grab jobs until there's no more jobs
        uint jobIndex = pool._jobIndex.load( std::memory_order_acquire );

        while( jobIndex < pool._jobCount )
        {
            bool acquired = pool._jobIndex.compare_exchange_weak( jobIndex, jobIndex+1, 
                std::memory_order_release,
                std::memory_order_relaxed );

            if( acquired )
            {
                ASSERT( pool._jobFunc );

                // We acquired the job, run it
                pool._jobFunc( pool._jobData + pool._jobDataSize * jobIndex );
            }
        }

        // Finished jobs, or there were no jobs to run,
        pool._poolSignal.Release();
    }
}