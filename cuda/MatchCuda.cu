#include "CudaPlotContext.h"
#include "ChiaConsts.h"

#define CU_MATCH_THREAD_COUNT (kExtraBitsPow)

#define BBCU_SCAN_GROUP_THREADS      128
#define BBCU_THREADS_PER_MATCH_GROUP 352
static constexpr uint32 BBCU_MAX_ENTRIES_PER_GROUP = 238;
static constexpr uint32 BBCU_MIN_ENTRIES_PER_GROUP = 230;
static constexpr uint32 BBCU_MIN_GROUP_COUNT       = ( CuCDiv( BBCU_BUCKET_ENTRY_COUNT, BBCU_MAX_ENTRIES_PER_GROUP ) );
static constexpr uint32 BBCU_MAX_GROUP_COUNT       = ( CuCDiv( BBCU_BUCKET_ENTRY_COUNT, BBCU_MIN_ENTRIES_PER_GROUP ) );

static_assert( CU_MAX_BC_GROUP_BOUNDARIES >= BBCU_MAX_GROUP_COUNT );

// #NOTE: The above have been tuned for 128 buckets, should check them for other bucket counts.
//static_assert( BBCU_BUCKET_COUNT == 128, "Unexpected bucket count" );

//-----------------------------------------------------------
__forceinline__ __device__ uint16 GenLTarget( const uint16 parity, uint16 rTargetIdx, const uint16 m )
{
    const uint16 indJ = rTargetIdx / kC;
    return ((indJ + m) % kB) * kC + (((2 * m + parity) * (2 * m + parity) + rTargetIdx) % kC);
}

//-----------------------------------------------------------
__global__ void CudaInitGroups( uint32* entries )
{
    const uint32 id       = threadIdx.x;
    const uint32 groupIdx = blockIdx.x;
    const uint32 gid      = blockIdx.x * blockDim.x + id;

    entries[gid] = 0xFFFFFFFF;
}

//-----------------------------------------------------------
__global__ void CudaSetFirstAndLastGroup( uint32* groups, const uint32 entryCount )
{
    const uint32 id       = threadIdx.x;
    const uint32 groupIdx = blockIdx.x;
    const uint32 gid      = blockIdx.x * blockDim.x + id;

    if( id == 0 )
        groups[id] = 0;
    else
        groups[id] = entryCount;
}

//-----------------------------------------------------------
__global__ void ScanGroupsCuda( const uint32* yEntries, uint32* groupBounadries, uint32* gGroupCount, const uint32 entryCount, const uint64 bucketMask )
{
    const uint32 threadCount  = gridDim.x * blockDim.x;
    const uint32 lastThreadId = threadCount - 1;
    const uint32 id           = threadIdx.x;
    const uint32 gid          = blockIdx.x * blockDim.x + id;

    if( gid >= entryCount-1 )
        return;
    
    __shared__ uint32 sharedGroupCount;
    if( id == 0 )
        sharedGroupCount = 0;

    __syncthreads();

    const uint64 currentGroup = ( bucketMask | yEntries[gid]   ) / kBC;
    const uint64 nextGroup    = ( bucketMask | yEntries[gid+1] ) / kBC;

    uint32 offset;
    if( currentGroup != nextGroup )
    {
        // #TODO: Use cooperative groups here instead, so we can just sync these threads
        offset = atomicAdd( &sharedGroupCount, 1 );
    }

    __syncthreads();

    // Global sync
    if( id == 0 )
        sharedGroupCount = atomicAdd( gGroupCount, sharedGroupCount );

    __syncthreads();

    if( currentGroup != nextGroup )
    {
        CUDA_ASSERT( sharedGroupCount + offset < CU_MAX_BC_GROUP_BOUNDARIES );
        groupBounadries[sharedGroupCount + offset] = gid+1;
    }
}

//-----------------------------------------------------------
__global__ void MatchCuda( const uint64 bucketMask, const uint32 entryCount, const uint32* gGroupCounts, const uint32* yEntries, const uint32* groupBoundaries, uint32* gMatchCount, Pair* outPairs )
{
    // 1 thread per y
    const uint32 id       = threadIdx.x;
    const uint32 groupIdx = blockIdx.x;
    const uint32 gid      = blockIdx.x * blockDim.x + id;

    if( groupIdx >= *gGroupCounts )
        return;

    // Let's iterate over the R group instead, having loaded all of L into
    // shared memory.

    const uint32 groupLIdx    = groupBoundaries[groupIdx];
    const uint32 groupRIdx    = groupBoundaries[groupIdx+1];
    const uint32 groupREnd    = groupBoundaries[groupIdx+2];
    const uint64 groupL       = ( bucketMask | yEntries[groupLIdx] ) / kBC;
    const uint64 groupR       = ( bucketMask | yEntries[groupRIdx] ) / kBC;
    const uint32 groupRLength = groupREnd - groupRIdx;
    const uint64 groupLYStart = groupL * kBC;
    const uint64 groupRYStart = groupR * kBC;
    const uint32 groupLLength = groupRIdx - groupLIdx;

#if _DEBUG
    if( groupLLength >= BBCU_THREADS_PER_MATCH_GROUP || groupRLength >= BBCU_THREADS_PER_MATCH_GROUP )
        printf( "[%u] Group %u is too large: %u\n", gid, groupRIdx, ( groupRIdx - groupLIdx ) );
#endif
    CUDA_ASSERT( groupLLength <= BBCU_THREADS_PER_MATCH_GROUP );
    CUDA_ASSERT( groupRLength <= BBCU_THREADS_PER_MATCH_GROUP );

    __shared__ uint32 rMap[kBC/2+1];
    __shared__ uint32 sharedMatchCount;

    if( groupR - groupL != 1 )
        return;
    
    if( id == 0 )
        sharedMatchCount = 0;

    const uint16 localLY = (uint16)(( bucketMask | yEntries[groupLIdx + min(id, groupLLength-1)] ) - groupLYStart );
    const uint16 localRY = (uint16)(( bucketMask | yEntries[groupRIdx + min(id, groupRLength-1)] ) - groupRYStart );

    {
        {
            uint32 mapIdx = id;
            while(mapIdx < kBC/2+1 )
            { 
                // Each entry is:
                // hi           lo
                // 7     9      7     9
                //(count,offset|count,offset) 
                rMap[mapIdx] = 0x01FF01FF;
                mapIdx += BBCU_THREADS_PER_MATCH_GROUP;
            }
        }

        __syncthreads();
        
        const uint16 shift  = ( ( localRY & 1 ) << 4 );   // Shift left by 16 bits if odd
        const uint32 idx    = localRY >> 1;               // Divide by 2

        // First set the offsets for the even ones (lower bits)
        if( id < groupRLength && ( localRY & 1 ) == 0 )
            atomicMin( &rMap[idx], id | 0x01FF0000 );
            
        __syncthreads();

        // Then set offset for the odd ones
        if( id < groupRLength && ( localRY & 1 ) )
            atomicMin( &rMap[idx], (id << 16) | (rMap[idx] & 0x0000FFFF) );

        __syncthreads();

        // Finally, add the counts
        if( id < groupRLength )
            atomicAdd( &rMap[idx], 0x200ul << shift );
        
    }

    if( id >= groupLLength )
        return;

    __syncthreads();


    // Begin matching
    constexpr uint32 MAX_MATCHES = 16;
    Pair matches[MAX_MATCHES];
    uint32 matchCount = 0;

    #pragma unroll
    for( uint32 i = 0; i < kExtraBitsPow; i++ )
    {
        const uint16 lTarget = GenLTarget( (byte)(groupL & 1), localLY, (uint16)i );
        const uint16 shift   = ( ( lTarget & 1 ) << 4 );   // Shift left by 16 bits if odd
        const uint16 rValue  = (uint16)(rMap[lTarget>>1] >> shift);
        const int16  rCount  = (int16)(rValue >> 9);

        for( int32 j = 0; j < rCount; j++ )
        {
            CUDA_ASSERT( matchCount < MAX_MATCHES );
            matches[matchCount++] = { groupLIdx + id, groupRIdx + (rValue & 0x1FF) + j };
        }
    }

    // Store final values
    const uint32 copyOffset = atomicAdd( &sharedMatchCount, matchCount );
    __syncthreads();
        
    // Store our shared match count and get our global offset
    if( id == 0 )
        sharedMatchCount = atomicAdd( gMatchCount, sharedMatchCount );
    __syncthreads();

    outPairs += copyOffset + sharedMatchCount;

    for( uint32 i = 0; i < matchCount; i++ )
    {
        CUDA_ASSERT( matches[i].left < entryCount );
        CUDA_ASSERT( matches[i].right < entryCount );

        outPairs[i] = matches[i];
    }
}

//-----------------------------------------------------------
void CudaK32Match( CudaK32PlotContext& cx, const uint32* devY, cudaStream_t stream, cudaEvent_t event )
{
    const TableId inTable    = cx.table - 1;
    const uint32  entryCount = cx.bucketCounts[(int)inTable][cx.bucket];
    const uint64  bucketMask = BBC_BUCKET_MASK( cx.bucket );

    constexpr uint32 kscanblocks = CuCDiv( BBCU_BUCKET_ALLOC_ENTRY_COUNT, BBCU_SCAN_GROUP_THREADS );

    uint32* tmpGroupCounts = (uint32*)cx.devMatches;

    {
        // Initialize the entries to the max value so that they are not included in the sort
        CudaInitGroups<<<kscanblocks, BBCU_SCAN_GROUP_THREADS, 0, stream>>>( tmpGroupCounts );
        
        // Initialize the end of the entries to the max value so that they are not included in the sort
        CudaSetFirstAndLastGroup<<<1,2,0,stream>>>( tmpGroupCounts, entryCount );
    }

    CudaErrCheck( cudaMemsetAsync( cx.devGroupCount, 0, sizeof( uint32 ), stream ) );
    CudaErrCheck( cudaMemsetAsync( cx.devMatchCount, 0, sizeof( uint32 ), stream ) );
    ScanGroupsCuda<<<kscanblocks, BBCU_SCAN_GROUP_THREADS, 0, stream>>>( devY, tmpGroupCounts+2, cx.devGroupCount, entryCount, bucketMask );
    
    byte* sortTmpAlloc = (byte*)( tmpGroupCounts + BBCU_MAX_GROUP_COUNT );
    size_t sortTmpSize = ( BBCU_BUCKET_ALLOC_ENTRY_COUNT - BBCU_MAX_GROUP_COUNT ) * sizeof( uint32 );

#if _DEBUG
    size_t sortSize = 0;
    cub::DeviceRadixSort::SortKeys<uint32, uint32>( nullptr, sortSize, nullptr, nullptr, BBCU_MAX_GROUP_COUNT, 0, 32 );
    ASSERT( sortSize <= sortTmpSize );
#endif

    cub::DeviceRadixSort::SortKeys<uint32, uint32>( sortTmpAlloc, sortTmpSize, tmpGroupCounts, cx.devGroupBoundaries, BBCU_MAX_GROUP_COUNT, 0, 32, stream );

    MatchCuda<<<BBCU_MAX_GROUP_COUNT, BBCU_THREADS_PER_MATCH_GROUP, 0, stream>>>( bucketMask, entryCount, cx.devGroupCount, devY, cx.devGroupBoundaries, cx.devMatchCount, cx.devMatches );
}

