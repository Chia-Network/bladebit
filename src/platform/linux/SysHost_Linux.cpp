#include "SysHost.h"
#include "Platform.h"
#include "Util.h"

#include <sys/random.h>
#include <execinfo.h>
#include <signal.h>
#include <atomic>
#include <numa.h>
#include <numaif.h>

// #if _DEBUG
    #include "util/Log.h"
// #endif

std::atomic<bool> _crashed = false;

//-----------------------------------------------------------
size_t SysHost::GetPageSize()
{
    return (size_t)getpagesize();
}

//-----------------------------------------------------------
size_t SysHost::GetTotalSystemMemory()
{
    const size_t pageSize = GetPageSize();
    return (size_t)get_phys_pages() * pageSize;
}

//-----------------------------------------------------------
size_t SysHost::GetAvailableSystemMemory()
{
    const size_t pageSize = GetPageSize();
    return (size_t)get_avphys_pages() * pageSize;
}

//-----------------------------------------------------------
void* SysHost::VirtualAlloc( size_t size, bool initialize )
{
    // Align size to page boundary
    const size_t pageSize = GetPageSize();

    size = RoundUpToNextBoundary( size, (int)pageSize );

    // #TODO: Don't use a whole page size. But provide a VirtualAllocAligned for the block-aligned allocations
    // Add one page to store our size (yup a whole page for it...)
    size += pageSize;

    void* ptr = mmap( NULL, size, 
        PROT_READ | PROT_WRITE, 
        MAP_ANONYMOUS | MAP_PRIVATE,
        -1, 0
    );

    if( ptr == MAP_FAILED )
    {
        #if _DEBUG
            const int err = errno;
            Log::Line( "Error: mmap() returned %d (0x%x).", err, err );
            ASSERT( 0 );
        #endif
    
        return nullptr;
    }

    if( initialize )
    {
        byte* page = (byte*)ptr;

        const size_t pageCount = size / pageSize;
        const byte*  endPage   = page + pageCount * pageSize;

        do
        {
            *page = 0;
            page += pageSize;
        } while( page < endPage );
    }

    *((size_t*)ptr) = size;

    return ((byte*)ptr)+pageSize;
}

//-----------------------------------------------------------
void SysHost::VirtualFree( void* ptr )
{
    ASSERT( ptr );
    if( !ptr )
        return;

    const size_t pageSize = GetPageSize();

    byte* realPtr    = ((byte*)ptr) - pageSize;
    const size_t size = *((size_t*)realPtr);

    munmap( realPtr, size );
}

//-----------------------------------------------------------
bool SysHost::VirtualProtect( void* ptr, size_t size, VProtect flags )
{
    ASSERT( ptr );

    int prot = 0;

    if( IsFlagSet( flags, VProtect::NoAccess ) )
    {
        prot = PROT_NONE;
    }
    else
    {
        if( IsFlagSet( flags, VProtect::Read ) )
            prot |= PROT_READ;
        if( IsFlagSet( flags, VProtect::Write ) )
            prot |= PROT_WRITE;
    }

    int r = mprotect( ptr, size, prot );
    ASSERT( !r );

    return r == 0;
}

//-----------------------------------------------------------
uint64 SysHost::SetCurrentProcessAffinityMask( uint64 mask )
{
    return SetCurrentThreadAffinityMask( mask );
}

//-----------------------------------------------------------
uint64 SysHost::SetCurrentThreadAffinityMask( uint64 mask )
{
    pthread_t thread = pthread_self();

    cpu_set_t cpuSet;
    CPU_ZERO( &cpuSet );

    if( mask == 0 )
        CPU_SET( 1, &cpuSet );
    else
    {
        for( uint i = 0; i < 64; i++ )
        {
            if( mask & (1ull << i ) )
                CPU_SET( i+1, &cpuSet );
        }
    }

    int r = pthread_setaffinity_np( thread, sizeof(cpu_set_t), &cpuSet );
    if( r != 0 )
    {
        ASSERT( 0 );
        return 0;
    }

    r = pthread_getaffinity_np( thread, sizeof(cpu_set_t), &cpuSet );
    if( r != 0 )
    {
        ASSERT( 0 );
        return 0;
    }

    return mask;
}

//-----------------------------------------------------------
bool SysHost::SetCurrentThreadAffinityCpuId( uint32 cpuId )
{
    pthread_t thread = pthread_self();
    
    cpu_set_t cpuSet;
    CPU_ZERO( &cpuSet );
    CPU_SET( cpuId, &cpuSet );

    int r = pthread_setaffinity_np( thread, sizeof(cpu_set_t), &cpuSet );
    return r == 0;
}

//-----------------------------------------------------------
void CrashHandler( int signal )
{
    // Only let the first thread handle the crash
    bool crashed = false;
    if( !_crashed.compare_exchange_strong( crashed, true, 
        std::memory_order_release, std::memory_order_relaxed ) )
    {
        return;
    }

    const size_t MAX_POINTERS = 256;
    void* stackTrace[256] = { 0 };

    fprintf( stderr, "*** Crashed! ***\n" );
    fflush( stderr );

    int traceSize = backtrace( stackTrace, (int)MAX_POINTERS );
    backtrace_symbols_fd( stackTrace, traceSize, fileno( stderr ) );
    fflush( stderr );
    
    exit( 1 );
}

//-----------------------------------------------------------
void SysHost::InstallCrashHandler()
{
    signal( SIGSEGV, CrashHandler ); 
}

//-----------------------------------------------------------
void SysHost::Random( byte* buffer, size_t size )
{
    // See: https://man7.org/linux/man-pages/man2/getrandom.2.html

    ssize_t sizeRead;
    byte* writer = buffer;
    const byte* end = writer + size;

    const size_t BLOCK_SIZE = 256;
    
    while( writer < end )
    {
        size_t readSize = (size_t)(end - writer);
        if( readSize > BLOCK_SIZE )
            readSize = BLOCK_SIZE;
            
        sizeRead = getrandom( writer, readSize, 0 );

        // Should never get EINTR, but docs say to check anyway.
        if( sizeRead < 0 && errno != EINTR )
            Fatal( "getrandom syscall failed." );

        writer += (size_t)sizeRead;
    }
}

// #NOTE: This is not thread-safe
//-----------------------------------------------------------
const NumaInfo* SysHost::GetNUMAInfo()
{
    if( numa_available() == -1 )
        return nullptr;

    static NumaInfo _info;
    static NumaInfo* info = nullptr;

    // Initialize if not initialized
    if( !info )
    {
        memset( &_info, 0, sizeof( NumaInfo ) );
        
        const uint nodeCount = (uint)numa_num_configured_nodes();
   
        uint totalCpuCount = 0;
        Span<uint>* cpuIds = (Span<uint>*)malloc( sizeof( uint* ) * nodeCount );


        for( uint i = 0; i < nodeCount; i++ )
        {
            bitmask* cpuMask = numa_allocate_cpumask();
            if( !cpuMask )
                Fatal( "Failed to allocate NUMA CPU mask." );

            int r = numa_node_to_cpus( i, cpuMask );

            if( r )
            {
                int err = errno;
                Fatal( "Failed to get cpus from NUMA node %u with error: %d (0x%x)", i, err, err );
            }

            // Count how many CPUs in this node
            uint cpuCount = 0;
            for( uint64 j = 0; j < cpuMask->size; j++ )
                if( numa_bitmask_isbitset( cpuMask, (uint)j ) )
                    cpuCount ++;

            // Allocate a buffer for this cpu
            cpuIds[i].values = (uint*)malloc( sizeof( uint ) * cpuCount );
            cpuIds[i].length = cpuCount;

            // Assign CPUs
            uint cpuI = 0;
            for( uint64 j = 0; j < cpuMask->size; j++ )
            {
                int s = numa_bitmask_isbitset( cpuMask, (uint)j );
                if( s )
                    cpuIds[i].values[cpuI++] = (uint)j;

                ASSERT( cpuI <= cpuCount );
            }

            totalCpuCount += cpuCount;

            // #TODO BUG: This is a memory leak,
            //        but we're getting crashes releasing it or
            //        using it multiple times with numa_node_to_cpus on
            //        a signle allocations.
            //        Fix it. (Not fatal as it is a small allocation, and this has re-entry protection)
            // numa_free_cpumask( cpuMask );
        }

        // Save instance
        _info.nodeCount = nodeCount;
        _info.cpuCount  = totalCpuCount;
        _info.cpuIds    = cpuIds;
        info = &_info;
    }

    return info;
}

//-----------------------------------------------------------
void SysHost::NumaAssignPages( void* ptr, size_t size, uint node )
{
    numa_tonode_memory( ptr, size, (int)node );
}

//-----------------------------------------------------------
bool SysHost::NumaSetThreadInterleavedMode()
{
    const NumaInfo* numa = GetNUMAInfo();
    if( !numa )
        return false;
    
    const size_t MASK_SIZE = 128;
    unsigned long mask[MASK_SIZE];
    memset( mask, 0xFF, sizeof( mask ) );
    
    const int maxPossibleNodes = numa_num_possible_nodes();
    ASSERT( (MASK_SIZE * 64) >= (size_t)maxPossibleNodes );

    long r = set_mempolicy( MPOL_INTERLEAVE, mask, maxPossibleNodes );

    #if _DEBUG
    if( r )
    {
        int err = errno;
        Log::Error( "Warning: set_mempolicy() failed with error %d (0x%x).", err, err );
    }
    #endif

    return r == 0;
}

//-----------------------------------------------------------
bool SysHost::NumaSetMemoryInterleavedMode( void* ptr, size_t size )
{
    const NumaInfo* numa = GetNUMAInfo();
    if( !numa )
        return false;

    const size_t MASK_SIZE = 128;
    unsigned long mask[MASK_SIZE];
    memset( mask, 0xFF, sizeof( mask ) );

    const int maxPossibleNodes = numa_num_possible_nodes();
    ASSERT( (MASK_SIZE * 64) >= (size_t)maxPossibleNodes );

    long r = mbind( ptr, size, MPOL_INTERLEAVE, mask, maxPossibleNodes, 0 ); 
    
    #if _DEBUG
    if( r )
    {
        int err = errno;
        Log::Error( "Warning: mbind() failed with error %d (0x%x).", err, err );
    }
    #endif

    return r == 0;
}

//-----------------------------------------------------------
int SysHost::NumaGetNodeFromPage( void* ptr )
{
    const NumaInfo* numa = GetNUMAInfo();
    if( !numa )
        return -1;

    int node = -1;
    int r = numa_move_pages( 0, 1, &ptr, nullptr, &node, 0 );

    if( r )
    {
        int err = errno;
        Log::Error( "Warning: numa_move_pages() failed with error %d (0x%x).", err, err );
    }
    else if( node < 0 )
    {
        int err = std::abs( node );
        Log::Error( "Warning: numa_move_pages() node retrieval failed with error %d (0x%x).", err, err );
    }

    return node;
}