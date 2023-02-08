#include "MTJob.h"
#include "ThreadPool.h"

///
/// An MTJob which shares a single context object across all jobs
///

template<typename TJobContext>
struct MonoJob;

template<typename TJobContext>
using MonoJobRunFunc = void (*)( MonoJob<TJobContext>* self );

template<typename TJobContext>
struct MonoJob : MTJob<MonoJob<TJobContext>>
{
    TJobContext*                context;
    MonoJobRunFunc<TJobContext> run;
    
    inline void Run() override { run( this ); }
};

template<typename TJobContext>
inline void MonoJobRun( ThreadPool& pool, const uint32 threadCount, TJobContext* jobContext, MonoJobRunFunc<TJobContext> runFunc )
{
    MTJobRunner<MonoJob<TJobContext>> jobs( pool );
    
    for( uint32 i = 0; i< threadCount; i++ )
    {
        auto& job = jobs[i];
        job.run     = runFunc;
        job.context = jobContext;
    }

    jobs.Run( threadCount );
}

template<typename TJobContext>
inline void MonoJobRun( ThreadPool& pool, TJobContext* jobContext, MonoJobRunFunc<TJobContext> runFunc )
{
    MonoJobRun( pool, pool.ThreadCount(), jobContext, runFunc );
}
