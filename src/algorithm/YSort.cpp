#include "YSort.h"
#include "SysHost.h"
#include "threading/ThreadPool.h"
#include "Util.h"
#include "util/Log.h"
#include "Config.h"
#include "ChiaConsts.h"


template<typename JobT>
struct SortYBaseJob
{
    JobT* jobs;

    std::atomic<uint>* finishedCount;
    std::atomic<uint>* releaseLock;

    // #TODO: Convert these to a pointer of pointers so that we don't have to
    //        load to iterate on the job struct itself, which is really heavy,
    //        and will load more data than we need in the cache.
    uint32* counts;    // Counts array for each thread. This is set by the thread itself
    void*   pfxSum;

    uint    id;
    uint    threadCount;

protected:
    template<uint Radix, typename TPrefix>
    void CalculatePrefixSum( uint id, uint32* counts, TPrefix* pfxSum );

    void SyncThreads();
    void LockThreads();
    void ReleaseThreads();

};

struct SortYJob : SortYBaseJob<SortYJob>
{
    uint64  length;     // Total entries length

    uint64* input;
    uint64* tmp;

    uint32* sortKey;
    uint32* sortKeyTmp;
    
    template<bool HasSortKey>
    static void SortYThread( SortYJob* job );

private:
    template<bool HasSortKey, uint shift, typename YT>
    void SortBucket( const uint64 bucket, const uint offset, 
                     const uint bucketOffset, const uint32 length, 
                     uint32* counts, uint32* pfxSum,
                     uint32* input, YT* tmp,
                     uint32* sortKey, uint32* sortKeyTmp );
};

struct NumaSortJob
{
    uint id;
    uint threadCount;
    uint node;
    uint nodeCount;
    uint threadGroup;

    uint pageCount;
    uint pageStart;
    uint trailers;

    std::atomic<uint>* finishedCount;
    std::atomic<uint>* releaseLock;

    byte* pages;
    byte* tmpPages;

    byte* sortKeyPages;
    byte* tmpSortKeyPages;

    byte* countsPerPage;  // Count buffer, which is kept per page
    
    uint32* counts;       // Counts array for each thread.
    uint64* pfxSum;       // Prefix sums for each thread. We use a different buffers to avoid copying to tmp buffers.
};


//-----------------------------------------------------------
YSorter::YSorter( ThreadPool& pool )
    : _pool( pool )
{
    // const NumaInfo* numa = SysHost::GetNUMAInfo();
    // if( !numa )
    //     Fatal( "Cannot use NUMA y-sort on non-NUMA info." );
}

//-----------------------------------------------------------
YSorter::~YSorter()
{

}

//-----------------------------------------------------------
void YSorter::Sort( uint64 length, uint64* yBuffer, uint64* yTmp )
{
    DoSort( false, length, yBuffer, yTmp, nullptr, nullptr );
}

//-----------------------------------------------------------
void YSorter::Sort( 
        uint64 length, 
        uint64* yBuffer, uint64* yTmp,
        uint32* sortKey, uint32* sortKeyTmp )
{
    ASSERT( sortKey && sortKeyTmp );
    DoSort( true, length, yBuffer, yTmp, sortKey, sortKeyTmp );
}

//-----------------------------------------------------------
void YSorter::DoSort( bool useSortKey, uint64 length, 
                      uint64* yBuffer, uint64* yTmp,
                      uint32* sortKey, uint32* sortKeyTmp )
{
    ASSERT( length );
    ASSERT( yBuffer && yTmp );

    ThreadPool& pool        = _pool;
    const uint  threadCount = pool.ThreadCount();

    SortYJob jobs[MAX_THREADS];

    std::atomic<uint> finishedCount = 0;
    std::atomic<uint> releaseLock   = 0;

    for( uint i = 0; i < MAX_THREADS; i++ )
    {
        SortYJob& job = jobs[i];

        job.jobs          = jobs;
        job.finishedCount = &finishedCount;
        job.releaseLock   = &releaseLock;
        job.length        = length;
        job.counts        = nullptr;
        job.pfxSum        = nullptr;
        job.id            = i;
        job.threadCount   = threadCount;
        job.length        = length;
        job.input         = yBuffer;
        job.tmp           = yTmp;
        
        job.sortKey       = sortKey;
        job.sortKeyTmp    = sortKeyTmp;
    }

    if( useSortKey )
        pool.RunJob( SortYJob::SortYThread<true>, jobs, threadCount );
    else
        pool.RunJob( SortYJob::SortYThread<false>, jobs, threadCount );
}

//-----------------------------------------------------------
template<bool HasSortKey>
void SortYJob::SortYThread( SortYJob* job )
{
    constexpr uint Radix    = 256;
    constexpr uint Buckets  = (1u << kExtraBits);

    const uint id          = job->id;
    const uint threadCount = job->threadCount;

    uint64*    input       = job->input;
    uint64*    tmp         = job->tmp;

    uint32*    sortKey     = job->sortKey;
    uint32*    sortKeyTmp  = job->sortKeyTmp;

    uint32 counts[Radix];
    job->counts = counts;

    // Sort the last most significant byte first, yielding 256 buckets and
    // stripping out that byte, leaving us with a 32-bit element size for the radix sort.
    {
        uint64 pfxSum[Buckets];

        memset( counts, 0, sizeof( counts ) );
        memset( pfxSum, 0, sizeof( pfxSum ) );

              uint64 length = job->length / threadCount;
        const uint64 offset = length * id;

        if( id == threadCount - 1 )
            length += job->length - ( length * threadCount );

              uint64* src = input + offset;
        const uint64* end = src   + length;

        uint32* sortKeySrc;
        if constexpr ( HasSortKey )
            sortKeySrc = sortKey + offset;

        // Get counts
    #if !Y_SORT_BLOCK_MODE
        do { counts[*src >> 32]++; } 
        while( ++src < end );
    #else
        const uint64  numBlocks = length / 8;
        const uint64* blockEnd  = src + numBlocks * 8;
        do
        {
            counts[src[0] >> 32]++;
            counts[src[1] >> 32]++;
            counts[src[2] >> 32]++;
            counts[src[3] >> 32]++;

            counts[src[4] >> 32]++;
            counts[src[5] >> 32]++;
            counts[src[6] >> 32]++;
            counts[src[7] >> 32]++;
            
            src += 8;
        } while( src < blockEnd );
        
        while( src < end )
            counts[*src++ >> 32]++;
    #endif

        // Get prefix sum
        job->pfxSum = pfxSum;
        job->CalculatePrefixSum<Buckets>( id, counts, pfxSum );

        // Sort into buckets
        src = input + offset;
        uint32* tmp32 = (uint32*)tmp;
        // do
        for( uint64 i = length; i > 0; )
        {
            const uint64 value  = src[--i]; //*src;
            const byte   bucket = (byte)( value >> 32 );

            const uint64 idx = --pfxSum[bucket];
            tmp32[idx] = (uint32)value;

            if constexpr ( HasSortKey )
                sortKeyTmp[idx] = sortKeySrc[i];
        }
        // } while( ++src < end );

        std::swap( input, tmp );

        if constexpr ( HasSortKey )
            std::swap( sortKey, sortKeyTmp );
    }

    // Get lengths for each bucket
    {
        SortYJob* jobs = job->jobs;

        uint bucketLengths[Buckets];
        memset( bucketLengths, 0, sizeof( bucketLengths ) );

        for( uint i = 0; i < threadCount; i++ )
        {
            const uint32* tCounts = jobs[i].counts;

            for( uint j = 0; j < Buckets; j++ )
                bucketLengths[j] += tCounts[j];
        }

        // Ensure all threads have finished writing to tmp
        job->SyncThreads();


        // Now do a radix sort on the 3/4 bytes for each 32-bit entries stored in each bucket.
        uint pfxSum[Radix];
        job->pfxSum = pfxSum;

        uint bucketOffset = 0;
        for( uint bucket = 0; bucket < Buckets; bucket++ )
        {
            uint       length = bucketLengths[bucket] / threadCount;
            const uint offset = bucketOffset + length * id;

            // Add the remainder if we're the last thread
            if( id == threadCount-1 )
                length += bucketLengths[bucket] - (threadCount * length);

            job->SortBucket<HasSortKey, 0 >( bucket, offset, bucketOffset, length, counts, pfxSum, (uint32*)input, (uint32*)tmp  , sortKey,    sortKeyTmp );
            job->SortBucket<HasSortKey, 8 >( bucket, offset, bucketOffset, length, counts, pfxSum, (uint32*)tmp  , (uint32*)input, sortKeyTmp, sortKey    );
            job->SortBucket<HasSortKey, 16>( bucket, offset, bucketOffset, length, counts, pfxSum, (uint32*)input, (uint32*)tmp  , sortKey,    sortKeyTmp );

            bucketOffset += bucketLengths[bucket];
        }

        // Now do a final expansion sort for the MSB of the 32-bit entries.
        // NOTE: This has to be done as a last step, because if we do it within each
        //       bucket in the previous step, we would overwrite adjacent buckets during the expansion.

        bucketOffset = 0;

        for( uint bucket = 0; bucket < Buckets; bucket++ )
        {
            uint       length = bucketLengths[bucket] / threadCount;
            const uint offset = bucketOffset + length * id;

            // Add the remainder if we're the last thread
            if( id == threadCount-1 )
                length += bucketLengths[bucket] - (threadCount * length);

            job->SortBucket<HasSortKey, 24>( ((uint64)bucket) << 32, offset, bucketOffset, length, counts, pfxSum, (uint32*)tmp, input, sortKeyTmp, sortKey );

            bucketOffset += bucketLengths[bucket];
        }
    }
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"

//-----------------------------------------------------------
template<bool HasSortKey, uint shift, typename YT>
FORCE_INLINE void SortYJob::SortBucket( const uint64 bucket, const uint offset,
                                        const uint bucketOffset, const uint32 length, 
                                        uint32* counts, uint32* pfxSum,
                                        uint32* input, YT* tmp,
                                        uint32* sortKey, uint32* sortKeyTmp )
{
    const uint Radix = 256;

    const uint32* start = input + offset;
    const uint32* end   = start + length;

    uint32* src = (uint32*)start;
    uint32* keySrc;

    if constexpr ( HasSortKey )
        keySrc = sortKey + offset;
    

    // Get counts
    memset( counts, 0, sizeof( uint32 ) * Radix );

#if !Y_SORT_BLOCK_MODE
    do { counts[(*src >> shift) & 0xFF]++; }
    while( ++src < end );
#else
    // Assume block size = 64 bytes
    const uint64  numBlocks = length / 16;
    const uint32* blockEnd  = src + numBlocks * 16;
    do
    {
        counts[(src[0] >> shift) & 0xFF]++;
        counts[(src[1] >> shift) & 0xFF]++;
        counts[(src[2] >> shift) & 0xFF]++;
        counts[(src[3] >> shift) & 0xFF]++;
        counts[(src[4] >> shift) & 0xFF]++;
        counts[(src[5] >> shift) & 0xFF]++;
        counts[(src[6] >> shift) & 0xFF]++;
        counts[(src[7] >> shift) & 0xFF]++;

        counts[(src[8 ] >> shift) & 0xFF]++;
        counts[(src[9 ] >> shift) & 0xFF]++;
        counts[(src[10] >> shift) & 0xFF]++;
        counts[(src[11] >> shift) & 0xFF]++;
        counts[(src[12] >> shift) & 0xFF]++;
        counts[(src[13] >> shift) & 0xFF]++;
        counts[(src[14] >> shift) & 0xFF]++;
        counts[(src[15] >> shift) & 0xFF]++;
        
        src += 16;
    } while( src < blockEnd );
    
    while( src < end )
        counts[(*src++ >> shift) & 0xFF]++;
#endif

    // Get prefix sum
    CalculatePrefixSum<Radix>( id, counts, pfxSum );

    // Store in new location, iterating backwards
    src = (uint32*)start;
    YT* dst = tmp + bucketOffset;
    
    uint32* keyDst;
    if constexpr ( HasSortKey )
        keyDst = sortKeyTmp + bucketOffset;

    for( uint64 i = length; i > 0; )
    {
        YT value = src[--i];
        const byte cIdx = (byte)( value >> shift );

        const uint32 dstIdx = --pfxSum[cIdx];

        // Expand with bucket id
        if constexpr ( std::is_same<YT, uint64>::value )
            value |= bucket;

        dst[dstIdx] = value;

        if constexpr ( HasSortKey )
            keyDst[dstIdx] = keySrc[i];

    }

    SyncThreads();
}



//-----------------------------------------------------------
void SortYNumaThread( NumaSortJob* job )
{
    // Count entries for this radix in each page & store counts in that page's count buffer
    
}

//-----------------------------------------------------------
template<typename JobT>
template<uint Radix, typename TPrefix>
FORCE_INLINE void SortYBaseJob<JobT>::CalculatePrefixSum( uint id, uint32* counts, TPrefix* pfxSum )
{
    const uint  threadCount = this->threadCount;
    const auto* jobs        = this->jobs;

    SyncThreads();

    // Add all thread's counts
    memset( pfxSum, 0, sizeof( TPrefix ) * Radix );

    for( uint t = 0; t < threadCount; t++ )
    {
        const uint* tCounts = jobs[t].counts;

        for( uint i = 0; i < Radix; i++ )
            pfxSum[i] += tCounts[i];
    }

    // Calculate prefix sum for this thread
    for( uint i = 1; i < Radix; i++ )
        pfxSum[i] += pfxSum[i-1];

    // Substract the count from all threads after ours
    for( uint t = id+1; t < threadCount; t++ )
    {
        const uint* tCounts = jobs[t].counts;

        for( uint i = 0; i < Radix; i++ )
            pfxSum[i] -= tCounts[i];
    }
}

//-----------------------------------------------------------
template<typename JobT>
FORCE_INLINE void SortYBaseJob<JobT>::SyncThreads()
{
    auto& finishedCount        = *this->finishedCount;
    auto& releaseLock          = *this->releaseLock;
    const uint threadThreshold = this->threadCount - 1;

    // Reset the release lock
    if( id == 0 ) 
    {
        // Wait for all threads to finish
        while( finishedCount.load( std::memory_order_relaxed ) != threadThreshold );

        // Finished, initialize release lock & signal other threads
        releaseLock  .store( 0, std::memory_order_release );
        finishedCount.store( 0, std::memory_order_release );
    }
    else
    {
        // Signal we've finished so we can calculate the shared prefix sum
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

//-----------------------------------------------------------
template<typename JobT>
FORCE_INLINE void SortYBaseJob<JobT>::LockThreads()
{
    auto& finishedCount        = *this->finishedCount;
    const uint threadThreshold = this->threadCount - 1;
    
    // Wait for all threads to finish
    while( finishedCount.load( std::memory_order_relaxed ) != threadThreshold );
}

//-----------------------------------------------------------
template<typename JobT>
FORCE_INLINE void SortYBaseJob<JobT>::ReleaseThreads()
{
    auto& finishedCount = *this->finishedCount;
    auto& releaseLock   = *this->releaseLock;

    // Finished, init release lock & signal other threads
    releaseLock  .store( 0, std::memory_order_release );
    finishedCount.store( 0, std::memory_order_release );
}

#pragma GCC diagnostic pop

