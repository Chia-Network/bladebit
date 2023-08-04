#pragma once

#include "threading/MonoJob.h"

struct SortKeyJob
{
    template<typename T>
    inline static void GenerateKey( ThreadPool& pool, const uint32 threadCount, Span<T> keyBuffer )
    {
        ASSERT( pool.ThreadCount() >= threadCount );

        MonoJobRun<Span<T>>( pool, threadCount, &keyBuffer, []( MonoJob<Span<T>>* self ){

            Span<T> key = *self->context;

            T count, offset, end;
            GetThreadOffsets( self, (T)key.Length(), count, offset, end );

            for( T i = offset; i < end; i++ )
                key[i] = i;
        });
    }

    template<typename T>
    inline static void GenerateKey( ThreadPool& pool, Span<T> keyBuffer )
    {
        Run( pool, pool.ThreadCount(), keyBuffer );
    }

private:
    template<typename T, typename TKey>
    struct SortOnKeyJob
    {
        Span<T>    entriesIn;
        Span<T>    entriesOut;
        Span<TKey> key;
    };

public:
    template<typename T, typename TKey>
    inline static void SortOnKey( ThreadPool& pool, const uint32 threadCount, const Span<TKey> key, const Span<T> entriesIn, Span<T> entriesOut )
    {
        ASSERT( pool.ThreadCount() >= threadCount );

        using Job = SortOnKeyJob<T, TKey>;

        Job context;
        context.entriesIn  = entriesIn;
        context.entriesOut = entriesOut;
        context.key        = key;

        ASSERT( entriesIn.Length() == entriesOut.Length() && entriesIn.Length() == key.Length() );

        MonoJobRun<Job>( pool, threadCount, &context, []( MonoJob<Job>* self ){
            
            auto context    = self->context;
            auto entriesIn  = context->entriesIn;
            auto entriesOut = context->entriesOut;
            auto key        = context->key;

            TKey count, offset, end;
            GetThreadOffsets( self, (TKey)entriesIn.Length(), count, offset, end );

            for( TKey i = offset; i < end; i++ )
                entriesOut[i] = entriesIn[key[i]];
        });
    }

    template<typename T, typename TKey>
    inline static void SortOnKey( ThreadPool& pool, const Span<TKey> key, const Span<T> entriesIn, Span<T> entriesOut )
    {
        SortOnKey( pool, pool.ThreadCount(), key, entriesIn, entriesOut );
    }
};