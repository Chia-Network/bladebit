#pragma once
#include "threading/ThreadPool.h"
#include <cstring>

class RadixSort256
{
    template<typename T1, typename T2>
    struct SortJob
    {
        uint id;                    // Id 0 is in charge of performing the non-parallel steps.
        uint threadCount;           // How many threads are participating in the sort
        
        // When All threads have finished, we can
        // thread 0 (the control thread) can calculate jobs and signal them to continue
        std::atomic<uint>* finishedCount;
        std::atomic<uint>* releaseLock;

        uint64* counts;             // Counts array for each thread
        uint64* pfxSums;            // Prefix sums for each thread. We use a different buffers to avoid copying to tmp buffers.

        uint64 startIndex;          // Scan start index
        uint64 length;              // entry count in our scan region

        T1* input;
        T1* tmp;

        // For sort key gen jobs
        T2* keyInput;
        T2* keyTmp;
    };

    enum SortMode
    {
        Void                    = 0,
        ModeSingle              = 1 << 0,
        SortAndGenKey           = 1 << 1,   // Sort input and generate a key in keyInput at the same time
    };

public:
    template<uint32 ThreadCount, typename T1>
    static void Sort( ThreadPool& pool, T1* input, T1* tmp, uint64 length );

    template<uint32 ThreadCount, typename T1, typename TK>
    static void SortWithKey( ThreadPool& pool, T1* input, T1* tmp, TK* keyInput, TK* keyTmp, uint64 length );

private:

    template<uint32 ThreadCount, SortMode Mode, typename T1, typename TK>
    static void DoSort( ThreadPool& pool, T1* input, T1* tmp, TK* keyInput, TK* keyTmp, uint64 length );

    template<typename T1, typename T2, bool IsKeyed>
    static void RadixSortThread( SortJob<T1,T2>* job );

};


//-----------------------------------------------------------
template<uint32 ThreadCount, typename T1>
inline void RadixSort256::Sort( ThreadPool& pool, T1* input, T1* tmp, uint64 length )
{
    DoSort<ThreadCount, ModeSingle, T1, void>( pool, input, tmp, nullptr, nullptr, length );
}

//-----------------------------------------------------------
template<uint32 ThreadCount, typename T1, typename TK>
inline void RadixSort256::SortWithKey( ThreadPool& pool, T1* input, T1* tmp, TK* keyInput, TK* keyTmp, uint64 length )
{
    DoSort<ThreadCount, SortAndGenKey, T1, TK>( pool, input, tmp, keyInput, keyTmp, length );
}

//-----------------------------------------------------------
template<uint32 ThreadCount, RadixSort256::SortMode Mode, typename T1, typename TK>
void inline RadixSort256::DoSort( ThreadPool& pool, T1* input, T1* tmp, TK* keyInput, TK* keyTmp, uint64 length )
{
    const uint   threadCount      = ThreadCount > pool.ThreadCount() ? pool.ThreadCount() : ThreadCount;
    const uint64 entriesPerThread = length / threadCount;
    const uint64 trailingEntries  = length - ( entriesPerThread * threadCount ); 
    
    // #TODO: Make sure we have enough stack space for this.
    //        Create a large stack, or allocate this on the heap...
    //        In which case, make this an instance method and have preallocated buffers for the jobs too
    //        Since we don't know the thread count ahead of time, maybe we should just make sure we allocate enough space for the jobs
    //        and use an instance instead...
    uint64 counts    [ThreadCount*256]; 
    uint64 prefixSums[ThreadCount*256]; 

    std::atomic<uint> finishedCount = 0;
    std::atomic<uint> releaseLock   = 0;
    SortJob<T1, TK> jobs[ThreadCount];
    
    for( uint i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];

        job.id            = i;
        job.threadCount   = threadCount;
        job.finishedCount = &finishedCount;
        job.releaseLock   = &releaseLock;
        job.counts        = counts;
        job.pfxSums       = prefixSums;
        job.startIndex    = i * entriesPerThread;
        job.length        = entriesPerThread;
        job.input         = input;
        job.tmp           = tmp;

        job.keyInput = keyInput;
        job.keyTmp   = keyTmp;
    }

    jobs[threadCount-1].length += trailingEntries;
    
    if constexpr ( Mode == SortAndGenKey )
        pool.RunJob( RadixSortThread<T1, TK, true>, jobs, threadCount );
    else
        pool.RunJob( RadixSortThread<T1, TK, false>, jobs, threadCount );
}

#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

//-----------------------------------------------------------
template<typename T1, typename T2, bool IsKeyed>
void RadixSort256::RadixSortThread( SortJob<T1, T2>* job )
{
    constexpr uint Radix = 256;

    const uint32 iterations = sizeof( T1 );
    const uint32 shiftBase  = 8;
    
    uint32 shift = 0;

    const uint         id            = job->id;
    const uint         threadCount   = job->threadCount;
    std::atomic<uint>& finishedCount = *job->finishedCount;
    std::atomic<uint>& releaseLock   = *job->releaseLock;

    uint64*      counts    = job->counts  + id * Radix;
    uint64*      prefixSum = job->pfxSums + id * Radix;
    const uint64 length    = job->length;
    const uint64 offset    = job->startIndex;
    T1*          input     = job->input;
    T1*          tmp       = job->tmp;

    T2*          keyInput;
    T2*          keyTmp;

    if constexpr ( IsKeyed )
    {
        keyInput = job->keyInput;
        keyTmp   = job->keyTmp;
    }

    for( uint32 iter = 0; iter < iterations ; iter++, shift += shiftBase  )
    {
        // Zero-out the counts
        memset( counts, 0, sizeof( uint64 ) * Radix );

        // Grab our scan region from the input
        const T1* src = input + offset;
        
        const T2* keySrc;
        if constexpr ( IsKeyed )
            keySrc = keyInput + offset;
        

        // Store the occurrences of the current 'digit' 
        for( uint64 i = 0; i < length; i++ )
            counts[(src[i] >> shift) & 0xFF]++;
        
        // Synchronize with other threads to comput the correct prefix sum
        if( id == 0 )
        {
            // This is the control thread, it is in charge of computing the shared prefix sums.
            
            // Wait for all threads to finish
            while( finishedCount.load( std::memory_order_relaxed ) != threadCount-1 );

            uint64* allCounts  = job->counts;
            uint64* allPfxSums = job->pfxSums;

            // Use the last thread's prefix sum buffer as it will use
            // it does not require a second pass
            uint64* prefixSumBuffer = allPfxSums + (threadCount - 1) * Radix;

            // First sum all the counts. Start by copying ours to the last
            memcpy( prefixSumBuffer, counts, sizeof( uint64 ) * Radix );

            // Now add the rest of the thread's counts
            for( uint i = 1; i < threadCount; i++ )
            {
                uint64* tCounts = allCounts + i * Radix;

                for( uint32 j = 0; j < Radix; j++ )
                    prefixSumBuffer[j] += tCounts[j];
            }

            // Now we have the sum of all thread's counts,
            // we can calculate the prefix sum, which is
            // equivalent to the last thread's prefix sum
            for( uint32 j = 1; j < Radix; j++ )
                prefixSumBuffer[j] += prefixSumBuffer[j-1];
            
            const uint64* nextThreadCountBuffer = allCounts + (threadCount - 1) * Radix;

            // Now assign the adjusted prefix sum to each thread below the last thread
            // NOTE: We are traveling backwards from the last thread
            for( uint i = 1; i < threadCount; i++ )
            {
                uint64* tPrefixSum = prefixSumBuffer - Radix;

                // This thread's prefix sum is equal to the next thread's
                // prefix sum minus the next thread's count
                for( uint32 j = 0; j < Radix; j++ )
                    tPrefixSum[j] = prefixSumBuffer[j] - nextThreadCountBuffer[j];

                prefixSumBuffer       = tPrefixSum;
                nextThreadCountBuffer -= Radix;
            }

            // Finished, init release lock & signal other threads
            releaseLock  .store( 0, std::memory_order_release );
            finishedCount.store( 0, std::memory_order_release );
        }
        else
        {
            // Signal we've finished so we can calculate the shared prefix sum
            uint count = finishedCount.load( std::memory_order_acquire );

            while( !finishedCount.compare_exchange_weak( count, count+1, std::memory_order_release, std::memory_order_relaxed ) );
            
            // Wait for the control thread (id == 0 ) to signal us so
            // that we can continue working.
            while( finishedCount.load( std::memory_order_relaxed ) != 0 );

            // Ensure all threads have been released
            count = releaseLock.load( std::memory_order_acquire );
            while( !releaseLock.compare_exchange_weak( count, count+1, std::memory_order_release, std::memory_order_relaxed ) );
            while( releaseLock.load( std::memory_order_relaxed ) != threadCount-1 );
        }
        
        // Populate output array (access input in reverse now)
        // This writes to the whole output array, not just our section.
        // This can cause false sharing, but given that our inputs are
        // extremely large, and the accesses are random, we don't expect
        // a lot of this to be happening.
        for( uint64 i = length; i > 0; )
        {
            // Read the value & prefix sum index
            const T1 value = src[--i];

            const uint64 idx = (value >> shift) & 0xFF;

            // Store it at the right location by reading the count
            const uint64 dstIdx = --prefixSum[idx];
            tmp[dstIdx] = value;

            if constexpr ( IsKeyed )
                keyTmp[dstIdx] = keySrc[i];
        }

        // Swap arrays
        T1* t = input;
        input = tmp;
        tmp   = t;

        if constexpr ( IsKeyed )
        {
            T2* tk = keyInput;
            keyInput = keyTmp;
            keyTmp   = tk;
        }

        // If not the last iteration, signal we've finished so we can
        // safely read from the arrays after swapped. (all threads must finish writing)
        if( (iter+1) < iterations )
        {
            if( id == 0 )
            {
                // Wait for all threads
                while( finishedCount.load( std::memory_order_relaxed ) != (threadCount-1) );
                
                // Finished, init release lock & signal other threads
                releaseLock  .store( 0, std::memory_order_release );
                finishedCount.store( 0, std::memory_order_release );
            }
            else
            {
                // Signal control thread
                uint count = finishedCount.load( std::memory_order_acquire );

                while( !finishedCount.compare_exchange_weak( count, count+1, std::memory_order_release, std::memory_order_relaxed ) );

                // Wait for control thread to signal us
                while( finishedCount.load( std::memory_order_relaxed ) != 0 );

                // Ensure all threads have been released
                count = releaseLock.load( std::memory_order_acquire );
                while( !releaseLock.compare_exchange_weak( count, count+1, std::memory_order_release, std::memory_order_relaxed ) );
                while( releaseLock.load( std::memory_order_relaxed ) != threadCount-1 );
            }
        }
    }
}

#pragma GCC diagnostic pop


