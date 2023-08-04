#include "MTJob.h"
#include "ThreadPool.h"

///
/// An MTJob which shares a single immutable context object across all jobs
/// and also contains a single mutable object shared across all threads for outputs.
///

template<typename TJobContextIn, typename TJobContextOut>
struct GenJob;

template<typename TJobContextIn, typename TJobContextOut>
using GenJobRunFunc = void (*)( GenJob<TJobContextIn, TJobContextOut>* self );

template<typename TJobContextIn, typename TJobContextOut>
struct GenJob : MTJob<GenJob<TJobContextIn, TJobContextOut>>
{
    const TJobContextIn*  input;
          TJobContextOut* output;
    
    GenJobRunFunc<TJobContextIn, TJobContextOut> run;
    
    inline void Run() override { run( this ); }
};

template<typename TJobContextIn, typename TJobContextOut>
inline void GenJobRun( ThreadPool& pool, const uint32 threadCount, 
                       const TJobContextIn* input, TJobContextOut* output,
                       GenJobRunFunc<TJobContextIn, TJobContextOut> runFunc )
{
    MTJobRunner<GenJob<TJobContextIn, TJobContextOut>> jobs( pool );
    
    for( uint32 i = 0; i< threadCount; i++ )
    {
        auto& job = jobs[i];
        job.run     = runFunc;
        job.input   = input;
        job.output  = output;
    }

    jobs.Run( threadCount );
}

template<typename TJobContextIn, typename TJobContextOut>
inline void GenJobRun( ThreadPool& pool, 
                       const TJobContextIn* input, TJobContextOut* output,
                       GenJobRunFunc<TJobContextIn, TJobContextOut> runFunc )
{
    GenJobRun( pool, pool.ThreadCount(), input, output, runFunc );
}
