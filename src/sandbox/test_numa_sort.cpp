#include "threading/ThreadPool.h"
#include "SysHost.h"
#include "util/Util.h"
#include "util/Log.h"
#include "algorithm/RadixSort.h"
#include "pos/chacha8.h"
#include "ChiaConsts.h"
#include "algorithm/YSort.h"
#include "plotmem/FxSort.h"
#include <atomic>
#include <thread>

#include "Config.h"

// #include <numa.h>
// #include <numaif.h>

template<typename T>
bool CheckSorted( const T* ptr, uint64 length );
bool ValidateSortKey( const uint32* sortKey, uint32* sortKeyTmp, uint64 length );

struct ThreadJob
{
    uint   id;
    uint   threadCount;
};

struct FaultPageJob : ThreadJob
{
    size_t size;
    byte*  pages;
};

void FaultPages( ThreadPool& pool, void* pages, size_t size );
void FaultPageThread( FaultPageJob* job );


struct ChaChaJob : ThreadJob
{
    uint64  length;
    byte*   key;
    byte*   blocks;
    uint64* yBuffer;

};
void GenChaCha( ThreadPool& pool, byte key[32], uint64 length, uint64* entries, uint64* blocks );
void GenChaChaThread( ChaChaJob* job );

struct NumaYSortJob
{
    uint id;
    uint threadCount;
    std::atomic<uint>* finishedCount;
    std::atomic<uint>* releaseLock;

    uint pageCount;
    uint pageStart;
    uint trailers;

    uint node;
    uint nodeCount;

    byte* pages;
    byte* tmpPages;

    uint64* counts;  // Counts array for each thread
    uint64* pfxSums; // Prefix sums for each thread. We use a different buffers to avoid copying to tmp buffers.

    byte* sortKeyPages;
    byte* tmpSortKeyPages;
};

template<bool HasSortKey = false>
void NumaRadixSortThread( NumaYSortJob* job );

void TestYSort();

//-----------------------------------------------------------
void RadixSortYNUMA( ThreadPool& pool, 
    const uint64 length, 
    uint64* yBuffer, uint64* yTmp )
{
    const uint blockSize = 64;
    ASSERT( length >= blockSize );

    const NumaInfo* numa     = SysHost::GetNUMAInfo();
    const uint      pageSize = (uint)SysHost::GetPageSize();
    
    ASSERT( numa );
    ASSERT( pageSize >= blockSize );

    const uint entriesPerPage = (uint)( pageSize / sizeof( uint64 ) );
    const uint pageCount      = (uint)( (length  * sizeof( uint64 )) / pageSize );// (uint)CDiv( length * sizeof( uint64 ),  (uint64)pageSize );
    const uint nodeCount      = numa->nodeCount;
    const uint pagesPerNode   = pageCount    / nodeCount;
    const uint pagesPerThread = (uint)(pagesPerNode / numa->cpuIds->length);    // #TODO: Only supports nodes with even number of CPUs
    const uint blockPerPage   = pageSize     / blockSize;
    const uint threadCount    = numa->cpuCount;

    // Trailers in an extra, trailing page
    const uint trailers = (uint32)( length - ( pagesPerNode * nodeCount * entriesPerPage ) );

    byte* yPages    = (byte*)yBuffer;
    byte* yTmpPages = (byte*)yTmp;

    // const uint MAX_THREADS = 128;

    ASSERT( SysHost::NumaGetNodeFromPage( yPages    ) == 0 );
    ASSERT( SysHost::NumaGetNodeFromPage( yTmpPages ) == 0 );

    std::atomic<uint> finishedCount = 0;
    std::atomic<uint> releaseLock   = 0;

    uint64 counts    [MAX_THREADS*256]; 
    uint64 prefixSums[MAX_THREADS*256]; 

    NumaYSortJob jobs[MAX_THREADS];

    for( uint i = 0; i < nodeCount; i++ )
    {   
        auto& cpus = numa->cpuIds[i];

        for( uint j = 0; j < cpus.length; j++ )
        {
            const uint cpuId = cpus.values[j];

            auto& job = jobs[cpuId];

            job.pageCount     = pagesPerThread;
            job.pageStart     = i + j * ( nodeCount * pagesPerThread );
            job.trailers      = 0;
            job.id            = cpuId;
            job.threadCount   = numa->cpuCount;
            job.pages         = yPages;
            job.tmpPages      = yTmpPages;

            job.counts        = counts;
            job.pfxSums       = prefixSums;

            job.node          = i;
            job.nodeCount     = nodeCount;

            job.finishedCount = &finishedCount;
            job.releaseLock   = &releaseLock;
        }
    }

    // if( 0 )
    {
        Log::Line( "Radix Sort..." );
        auto timer = TimerBegin();
        // RadixSort256::Sort<MAX_THREADS>( pool, yBuffer, yTmp, length );
        RadixSort256::SortY<MAX_THREADS>( pool, yBuffer, yTmp, length );
        // {
        //     uint32* sortKey    = (uint32*)SysHost::VirtualAlloc( sizeof( uint32 ) * length * 2 );
        //     uint32* sortKeyTmp = sortKey + length;

        //     for( uint64 i = 0; i < length; i++ )
        //         sortKey[i] = (uint32)i;

        //     RadixSort256::SortYWithKey<MAX_THREADS>( pool, yBuffer, yTmp, sortKey, sortKeyTmp, length );;
        // }
        double elapsed = TimerEnd( timer );
        Log::Line( "Finished Radix Sort in %.2lf seconds.", elapsed );
        
        Log::Line( "Checking it is sorted..." );
        if( !CheckSorted( yTmp, length ) )
            Log::Line( " Failed." );
        else
            Log::Line( " Success." );
    }
    Log::Line( "" );
    if( 0 )
    {
        Log::Line( "NUMA Radix Sort..." );
        auto timer = TimerBegin();
        pool.RunJob( NumaRadixSortThread, jobs, threadCount );
        double elapsed = TimerEnd( timer );
        Log::Line( "Finished NUMA Radix Sort in %.2lf seconds.", elapsed );

        Log::Line( "Checking it is sorted..." );
        if( !CheckSorted( yBuffer, length ) )
            Log::Line( " Failed." );
        else
            Log::Line( " Success." );
    }
}

//-----------------------------------------------------------
void TestNumaSort( int argc, const char* argv[] )
{
    TestYSort(); return;

    const uint   k   = 20;
    const uint64 len = (1ull << k);
    
    Log::Line( "Allocating Data." );

    const NumaInfo* numa   = SysHost::GetNUMAInfo();
    
    // Allocate an extra page for each node
    const size_t pageSize  = SysHost::GetPageSize();
    
    const size_t extraSize = numa->nodeCount * pageSize;
    const size_t size      = RoundUpToNextBoundary( len * sizeof( uint64 ), (int)pageSize ) + extraSize;

    uint64* yBuffer = (uint64*)SysHost::VirtualAlloc( size, false );
    uint64* yTmp    = (uint64*)SysHost::VirtualAlloc( size, false );

    if( !SysHost::NumaSetMemoryInterleavedMode( yBuffer, size ) )
        Fatal( "Failed to interleave yBuffer." );
    if( !SysHost::NumaSetMemoryInterleavedMode( yTmp, size ) )
        Fatal( "Failed to interleave yTmpBuffer." );

    // Fault pages
    const size_t pageCount = CDiv( size, (int)pageSize );

    byte* pages    = (byte*)yBuffer;
    byte* tmpPages = (byte*)yTmp;
    
    for( size_t i = 0; i < pageCount; i++ )
    {
        *pages    = 0;
        *tmpPages = 0;

        pages    += pageSize;
        tmpPages += pageSize;
    }

    pages    = (byte*)yBuffer;
    tmpPages = (byte*)yTmp;


    // Find the first page belonging to node 0
    for( uint i = 0; i < numa->nodeCount; i++ )
    {
        int node = SysHost::NumaGetNodeFromPage( pages );
        ASSERT( node >= 0 );
        
        if( node == 0 ) break;

        pages += pageSize;
    }

    for( uint i = 0; i < numa->nodeCount; i++ )
    {
        int node = SysHost::NumaGetNodeFromPage( tmpPages );
        ASSERT( node >= 0 );
        
        if( node == 0 ) break;

        tmpPages += pageSize;
    }

    yBuffer = (uint64*)pages;
    yTmp    = (uint64*)tmpPages;

    {
        Log::Line( "ChaChaGen" );

        byte key[32] 
        = { 22, 24, 11, 3, 1, 15, 11, 6, 23, 22,
            22, 24, 11, 3, 1, 15, 11, 6, 23, 22,
            22, 24, 11, 3, 1, 15, 11, 6, 23, 22, 5, 28 }
        ;
        // SysHost::Random( key, sizeof( key ) );
        

        chacha8_ctx chacha;
        ZeroMem( &chacha );

        chacha8_keysetup( &chacha, key, 256, NULL );
        chacha8_get_keystream( &chacha, 0, len / 8, (byte*)yBuffer );

        for( uint64 i = 0; i < len; i++ )
            yBuffer[i] = yBuffer[i] >> ( 64 - (32 + kExtraBits) );

        Log::Line( "OK." );
    }


    ASSERT( SysHost::NumaGetNodeFromPage( pages    ) == 0 );
    ASSERT( SysHost::NumaGetNodeFromPage( tmpPages ) == 0 );

    const uint threadCount = SysHost::GetNUMAInfo()->cpuCount;
    ThreadPool pool( threadCount, ThreadPool::Mode::Fixed );

    RadixSortYNUMA( pool, len, yBuffer, yTmp );
}

//-----------------------------------------------------------
void TestYSort()
{
    const uint   k   = 30;
    const uint64 len = (1ull << k);

    const uint   threadCount = SysHost::GetLogicalCPUCount();

    ThreadPool pool( threadCount, ThreadPool::Mode::Fixed );

    const size_t size = RoundUpToNextBoundary( len * sizeof( uint64 ), 64 );

    Log::Line( "Allocating buffers for k%d with %d threads.", k, threadCount );
    uint64* yBuffer = (uint64*)SysHost::VirtualAlloc( size, false );
    uint64* yTmp    = (uint64*)SysHost::VirtualAlloc( size, false );

    FaultPages( pool, yBuffer, size );
    FaultPages( pool, yTmp   , size );

    #define USE_SORT_KEY 1

    #if USE_SORT_KEY
        uint32* sortKey    = (uint32*)SysHost::VirtualAlloc( sizeof( uint32 ) * len, false );
        uint32* sortKeyTmp = (uint32*)SysHost::VirtualAlloc( sizeof( uint32 ) * len, false );
    #endif

    byte key[32] 
        = { 22, 24, 11, 3, 1, 15, 11, 6, 23, 22,
            22, 24, 11, 3, 1, 15, 11, 6, 23, 22,
            22, 24, 11, 3, 1, 15, 11, 6, 23, 22, 5, 28 }
        ;
        // SysHost::Random( key, sizeof( key ) );
       
    if( 0 )
    {
        Log::Line( "Classic Radix Sort" );

        Log::Line( "  Generating ChaCha" );
        GenChaCha( pool, key, len, yBuffer, yTmp );

        #if USE_SORT_KEY
            Log::Line( "  Generating Sort Key..." );
            GenSortKey<MAX_THREADS>( pool, len, sortKey );
        #endif

        Log::Line( "  Sorting..." );
        auto timer = TimerBegin();
        #if USE_SORT_KEY
            RadixSort256::SortWithKey<MAX_THREADS>( pool, yBuffer, yTmp, sortKey, sortKeyTmp, len );
        #else
            RadixSort256::Sort<MAX_THREADS>( pool, yBuffer, yTmp, len );
        #endif
        double elapsed = TimerEnd( timer );
        Log::Line( "Finished sort in %.2lf seconds.", elapsed );

        #if USE_SORT_KEY
            Log::Write( "Verifying Key... " ); Log::Flush();
            const bool okSortKey = ValidateSortKey( sortKey, sortKeyTmp, len );
            Log::Line( "%s", okSortKey ? "OK" : "Failed" );
        #endif

        Log::Write( "Verifying Sort... " ); Log::Flush();
        const bool ok = CheckSorted( yBuffer, len );
        Log::Line( "%s", ok ? "OK" : "Failed" );
    }

    Log::Line( "" );

    if( 0 )
    {
        Log::Line( "Radix Limited Y Sort" );

        Log::Line( "  Generating ChaCha" );
        GenChaCha( pool, key, len, yBuffer, yTmp );

        #if USE_SORT_KEY
            Log::Line( "  Generating Sort Key..." );
            GenSortKey<MAX_THREADS>( pool, len, sortKey );
        #endif

        Log::Line( "  Sorting..." );
        auto timer = TimerBegin();
        #if USE_SORT_KEY
            RadixSort256::SortYWithKey<MAX_THREADS>( pool, yBuffer, yTmp, sortKey, sortKeyTmp, len );
        #else
            RadixSort256::SortY<MAX_THREADS>( pool, yBuffer, yTmp, len );
        #endif
        double elapsed = TimerEnd( timer );
        Log::Line( "Finished sort in %.2lf seconds.", elapsed );

        #if USE_SORT_KEY
            Log::Write( "Verifying Key... " ); Log::Flush();
            const bool okSortKey = ValidateSortKey( sortKeyTmp, sortKey, len );
            Log::Line( "%s", okSortKey ? "OK" : "Failed" );
        #endif

        Log::Write( "Verifying Sort... " ); Log::Flush();
        const bool ok = CheckSorted( yTmp, len );
        Log::Line( "%s", ok ? "OK" : "Failed" );
    }

    Log::Line( "" );
    
    // if( 0 )
    {
        Log::Line( "Y Sort" );

        Log::Line( "  Generating ChaCha" );
        GenChaCha( pool, key, len, yBuffer, yTmp );
        
        for( uint i = 0; i < 10; i++ )
        {
            #if USE_SORT_KEY
                Log::Line( "  Generating Sort Key..." );
                GenSortKey<MAX_THREADS>( pool, len, sortKey );
            #endif

            Log::Line( "  Sorting..." );
            YSorter sorter( pool );
            auto timer = TimerBegin();

            #if USE_SORT_KEY
                sorter.Sort( len, yBuffer, yTmp, sortKey, sortKeyTmp );
            #else
                sorter.Sort( len, yBuffer, yTmp );
            #endif
            double elapsed = TimerEnd( timer );
            Log::Line( "Finished sort in %.2lf seconds.", elapsed );

            #if USE_SORT_KEY
                // Log::Write( "Verifying Key... " ); Log::Flush();
                // const bool okSortKey = ValidateSortKey( sortKeyTmp, sortKey, len );
                // Log::Line( "%s", okSortKey ? "OK" : "Failed" );
                // std::swap( sortKey, sortKeyTmp );
            #endif

            Log::Write( "Verifying Sort... " ); Log::Flush();
            const bool ok = CheckSorted( yTmp, len );
            Log::Line( "%s", ok ? "OK" : "Failed" );
            std::swap( yBuffer, yTmp );
        }
    }
}

//-----------------------------------------------------------
template<bool HasSortKey>
void NumaRadixSortThread( NumaYSortJob* job )
{
    constexpr uint Radix = 256;

    const uint32 iterations = 5;//sizeof( uint64 );
    const uint32 shiftBase  = 8;
    
    uint32 shift = 0;

    const uint         id            = job->id;
    const uint         threadCount   = job->threadCount;
    std::atomic<uint>& finishedCount = *job->finishedCount;
    std::atomic<uint>& releaseLock   = *job->releaseLock;

    uint64* counts    = job->counts  + id * Radix;
    uint64* prefixSum = job->pfxSums + id * Radix;
    
    const uint nodeCount       = job->nodeCount;
    const uint node            = job->node;
    
    const uint pageSize        = (uint)SysHost::GetPageSize();
    const uint pageCount       = job->pageCount;
    const uint pageStart       = job->pageStart;
    const uint blocksPerPage   = pageSize / 64;
    const uint entriesPerBlock = 64 / sizeof( uint64 );
    const uint entriesPerPage  = pageSize / sizeof( uint64 );
    const uint pageStride      = pageSize * nodeCount;
    const uint entryStride     = pageStride / sizeof( uint64 );
    const uint sortKeyStride   = pageStride / sizeof( uint32 );


    // const uint64 length    = job->length;
    // const uint64 offset    = job->startIndex;

    byte* pages    = job->pages    + pageStart * pageSize;
    byte* tmpPages = job->tmpPages + pageStart * pageSize;

    // uint64*      input     = job->input;
    // uint64*      tmp       = job->tmp;

    uint64* input    = (uint64*)pages;
    uint64* tmpInput = (uint64*)tmpPages;

    uint64* dst    = (uint64*)job->tmpPages;
    uint64* tmpDst = (uint64*)job->pages;

    byte*   sortKeyPages;
    byte*   sortKeyTmpPages;
    uint32* keyInput;
    uint32* keyTmpInput;
    uint32* keyDst;
    uint32* keyTmpDst;

    if constexpr ( HasSortKey )
    {
        sortKeyPages    = job->sortKeyPages    + pageStart * pageSize;
        sortKeyTmpPages = job->tmpSortKeyPages + pageStart * pageSize;

        keyInput    = (uint32*)sortKeyPages;
        keyTmpInput = (uint32*)sortKeyTmpPages;

        keyDst      = (uint32*)job->tmpSortKeyPages;
        keyTmpDst   = (uint32*)job->sortKeyPages;
    }

    for( uint32 iter = 0; iter < iterations ; iter++, shift += shiftBase  )
    {
        // Zero-out the counts
        memset( counts, 0, sizeof( uint64 ) * Radix );

        // Store the occurrences of the current 'digit' 
        {
            // Grab our scan region from the input
            const uint64* src = input;

            for( uint page = 0; page < pageCount; page++ )
            {
                const uint64* block = src;
                for( uint blockIdx = 0; blockIdx < blocksPerPage; blockIdx++ )
                {
                    uint i0 = (uint)( (block[0] >> shift) & 0xFF );
                    uint i1 = (uint)( (block[1] >> shift) & 0xFF );
                    uint i2 = (uint)( (block[2] >> shift) & 0xFF );
                    uint i3 = (uint)( (block[3] >> shift) & 0xFF );
                    uint i4 = (uint)( (block[4] >> shift) & 0xFF );
                    uint i5 = (uint)( (block[5] >> shift) & 0xFF );
                    uint i6 = (uint)( (block[6] >> shift) & 0xFF );
                    uint i7 = (uint)( (block[7] >> shift) & 0xFF );

                    if( id == 0 && page == pageCount-1 && blockIdx == blocksPerPage-1 )
                        Log::Line( "Last Page/Block." );

                    if( id == 0 && page == (pageCount-1) - 672 && blockIdx == 49 )
                        Log::Line( "Break" );

                    counts[i0]++;
                    counts[i1]++;
                    counts[i2]++;
                    counts[i3]++;
                    counts[i4]++;
                    counts[i5]++;
                    counts[i6]++;
                    counts[i7]++;

                    block += entriesPerBlock;
                }

                src += entryStride;
            }
        }
        
        // #TODO: Handle trailers
        
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
        
        {
            const uint64* src = input + ( pageCount * entryStride - entryStride );   // Start at the last page
            ASSERT( SysHost::NumaGetNodeFromPage( (void*)src ) == (int)job->node );

            uint sortKeyStrideMask = 0;
            const uint32* keyBlock;
            
            if constexpr ( HasSortKey )
            {
                keyBlock = keyInput + sortKeyStride * ( pageCount - 1 );
            }

            for( uint page = 0; page < pageCount; page++ )
            {
                const uint64* block = src + entriesPerPage - entriesPerBlock;

                for( uint blockIdx = 0; blockIdx < blocksPerPage; blockIdx++ )
                {
                    const uint64 v0 = block[0];
                    const uint64 v1 = block[1];
                    const uint64 v2 = block[2];
                    const uint64 v3 = block[3];
                    const uint64 v4 = block[4];
                    const uint64 v5 = block[5];
                    const uint64 v6 = block[6];
                    const uint64 v7 = block[7];

                    const uint idx0 = (uint)( (v0 >> shift) & 0xFF );
                    const uint idx1 = (uint)( (v1 >> shift) & 0xFF );
                    const uint idx2 = (uint)( (v2 >> shift) & 0xFF );
                    const uint idx3 = (uint)( (v3 >> shift) & 0xFF );
                    const uint idx4 = (uint)( (v4 >> shift) & 0xFF );
                    const uint idx5 = (uint)( (v5 >> shift) & 0xFF );
                    const uint idx6 = (uint)( (v6 >> shift) & 0xFF );
                    const uint idx7 = (uint)( (v7 >> shift) & 0xFF );

                    // Store it at the right location by reading the count
                    const uint64 dstIdx7 = --prefixSum[idx7];
                    const uint64 dstIdx6 = --prefixSum[idx6];
                    const uint64 dstIdx5 = --prefixSum[idx5];
                    const uint64 dstIdx4 = --prefixSum[idx4];
                    const uint64 dstIdx3 = --prefixSum[idx3];
                    const uint64 dstIdx2 = --prefixSum[idx2];
                    const uint64 dstIdx1 = --prefixSum[idx1];
                    const uint64 dstIdx0 = --prefixSum[idx0];

                    dst[dstIdx0] = v0;
                    dst[dstIdx1] = v1;
                    dst[dstIdx2] = v2;
                    dst[dstIdx3] = v3;
                    dst[dstIdx4] = v4;
                    dst[dstIdx5] = v5;
                    dst[dstIdx6] = v6;
                    dst[dstIdx7] = v7;

                    if constexpr ( HasSortKey )
                    {
                        const uint32 k0 = keyBlock[0];
                        const uint32 k1 = keyBlock[1];
                        const uint32 k2 = keyBlock[2];
                        const uint32 k3 = keyBlock[3];
                        const uint32 k4 = keyBlock[4];
                        const uint32 k5 = keyBlock[5];
                        const uint32 k6 = keyBlock[6];
                        const uint32 k7 = keyBlock[7];
                        
                        keyDst[dstIdx0] = k0;
                        keyDst[dstIdx1] = k1;
                        keyDst[dstIdx2] = k2;
                        keyDst[dstIdx3] = k3;
                        keyDst[dstIdx4] = k4;
                        keyDst[dstIdx5] = k5;
                        keyDst[dstIdx6] = k6;
                        keyDst[dstIdx7] = k7;

                        keyBlock += entriesPerBlock;
                    }

                    block -= entriesPerBlock;
                }

                src -= entryStride;

                if constexpr ( HasSortKey )
                {
                    // Only stride on the second round since
                    // we have double the key blocks per page.
                    keyBlock -= ( ( sortKeyStride + 1 ) & sortKeyStrideMask );

                    sortKeyStrideMask = ~sortKeyStrideMask;
                }
            }

            // #TODO: Handle trailers
        }

        // Swap arrays
        std::swap( input, tmpInput );
        std::swap( dst  , tmpDst   );

        if constexpr ( HasSortKey )
        {
            std::swap( keyInput, keyTmpInput );
            std::swap( keyDst  , keyTmpDst   );
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

template<typename T>
bool CheckSorted( const T* ptr, uint64 length )
{
    if( length < 2 )
        return true;

    for( uint64 i = 1; i < length; i++ )
    {
        const T a = ptr[i-1];
        const T b = ptr[i];

        if( a > b )
            return false;
    }

    return true;
}

//-----------------------------------------------------------
bool ValidateSortKey( const uint32* sortKey, uint32* sortKeyTmp, uint64 length )
{
    for( uint64 i = 0; i < length; i++ )
    {
        const uint32 key = sortKey[i];
        sortKeyTmp[key] = key;
    }

    for( uint64 i = 1; i < length; i++ )
    {
        if( sortKeyTmp[i] <= sortKeyTmp[i-1] )
            return false;
    }

    return true;
}

//-----------------------------------------------------------
void FaultPages( ThreadPool& pool, void* pages, size_t size )
{
    FaultPageJob jobs[MAX_THREADS];

    for( uint i = 0; i < pool.ThreadCount(); i++ )
    {
        FaultPageJob& job = jobs[i];

        job.id          = i;
        job.threadCount = pool.ThreadCount();
        job.pages       = (byte*)pages;
        job.size        = size;
    }

    pool.RunJob( FaultPageThread, jobs, pool.ThreadCount() );
}

//-----------------------------------------------------------
void FaultPageThread( FaultPageJob* job )
{
    const size_t pageSize = SysHost::GetPageSize();

    size_t size = job->size / job->threadCount;

    byte* pages = job->pages + size * job->id;

    if( job->id == job->threadCount - 1 )
        size += job->size - size * job->threadCount;

    const size_t pageCount = CDiv( size, (int)pageSize );

    for( uint64 i = 0; i < pageCount; i++ )
    {
        *pages = 0;
        pages += pageSize;
    }
}

//-----------------------------------------------------------
void GenChaCha( ThreadPool& pool, byte key[32], uint64 length, uint64* entries, uint64* blocks )
{
    ChaChaJob jobs[MAX_THREADS];

    const uint threadCount = pool.ThreadCount();

    for( uint i = 0; i < threadCount; i++ )
    {
        ChaChaJob& job = jobs[i];

        job.id          = i;
        job.threadCount = threadCount;
        job.key         = key;
        job.length      = length;
        job.blocks      = (byte*)blocks;
        job.yBuffer     = entries;
    }

    pool.RunJob( GenChaChaThread, jobs, pool.ThreadCount() );
}

//-----------------------------------------------------------
void GenChaChaThread( ChaChaJob* job )
{
    // if( job->id == 23 )
    //     Log::Line( "Break" );

    const size_t blockSize        = 64;
    const size_t entriesPerBlock  = blockSize / sizeof( uint32 );

    const uint64 blockCount       = job->length * sizeof( uint32 ) / blockSize;
          uint64 blocksPerThread  = blockCount  / job->threadCount;
    const size_t entriesPerThread = blocksPerThread * entriesPerBlock;

    uint32* blocks  = (uint32*)( job->blocks + job->id * (blocksPerThread * blockSize) );
    uint64* yBuffer = job->yBuffer + job->id * entriesPerThread;

    const uint64 blockIdx = blocksPerThread * job->id;
    const uint64 x        = entriesPerBlock * blocksPerThread * job->id;

    if( job->id == job->threadCount-1 )
        blocksPerThread += blockCount - blocksPerThread * job->threadCount;

    chacha8_ctx chacha;
    ZeroMem( &chacha );

    chacha8_keysetup( &chacha, job->key, 256, NULL );
    chacha8_get_keystream( &chacha, blockIdx, (uint)blockCount, (byte*)blocks );

    const uint64 length = blocksPerThread * entriesPerBlock;
    
    for( uint64 i = 0; i < length; i++ )
    {
        const uint64 y = blocks[i];// Swap32( blocks[i] );
        yBuffer[i] = ( y << kExtraBits ) | ( (x+i) >> (_K - kExtraBits) );
    }
}



