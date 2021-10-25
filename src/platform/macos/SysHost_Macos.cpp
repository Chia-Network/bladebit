#include "SysHost.h"
#include "Platform.h"
#include "Util.h"
#include "util/Log.h"

//-----------------------------------------------------------
size_t SysHost::GetPageSize()
{
    // #TODO: Use host_page_size
    return (size_t)getpagesize();
}

//-----------------------------------------------------------
size_t SysHost::GetTotalSystemMemory()
{   
    uint count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vmstat;
    ZeroMem( &vmstat );

    if( host_statistics64( mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vmstat, &count ) != KERN_SUCCESS )
        return 0;
    
    const size_t pageSize = GetPageSize();
    
    return ( vmstat.free_count     +
             vmstat.active_count   +
             vmstat.inactive_count +
             vmstat.wire_count ) * pageSize;
}

//-----------------------------------------------------------
size_t SysHost::GetAvailableSystemMemory()
{
    uint count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vmstat;
    ZeroMem( &vmstat );

    if( host_statistics64( mach_host_self(), HOST_VM_INFO, (host_info64_t)&vmstat, &count ) != KERN_SUCCESS )
        return 0;
    
    const size_t pageSize = GetPageSize();
    
    return ( vmstat.free_count     +
             vmstat.inactive_count ) * pageSize;
}

//-----------------------------------------------------------
uint SysHost::GetLogicalCPUCount()
{
    host_basic_info info;
    ZeroMem( &info );

    mach_msg_type_number_t host_info_count = (mach_msg_type_number_t)sizeof( info );
    host_info( mach_host_self(), HOST_BASIC_INFO, (host_info_t)&info, &host_info_count );

    return (uint)info.avail_cpus;
}

//-----------------------------------------------------------
void* SysHost::VirtualAlloc( size_t size, bool initialize )
{
    // #TODO: Use vm_allocate
    // #TODO: Consider initialize
    
    const size_t pageSize = (size_t)getpagesize();

    // Align size to page boundary
    size = RoundUpToNextBoundary( size, (int)pageSize );

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
    else if( initialize )
    {
        // Initialize memory 
        // (since physical pages are not allocated until the actual pages are accessed)

        byte* page = (byte*)ptr;

        const size_t pageCount = size / pageSize;
        const byte*  endPage   = page + pageCount * pageSize;

        do
        {
            *page = 0;
            page += pageSize;
        } while( page < endPage );
    }

    return ptr;
}

//-----------------------------------------------------------
void SysHost::VirtualFree( void* ptr )
{
    // #TODO: Implement
}

/*
//-----------------------------------------------------------
uint64 SysHost::SetCurrentProcessAffinityMask( uint64 mask )
{
    return SetCurrentThreadAffinityMask( mask );
}

// #TODO: This should perhaps return bool.
//-----------------------------------------------------------
uint64 SysHost::SetCurrentThreadAffinityMask( uint64 mask )
{
    thread_port_t thread = mach_thread_self();

    // It seems macOS does not support 64 bit affinity masks.
    // This may be since they have never had a processor with more than 32 cores.
    thread_affinity_policy_data_t policy = { (integer_t)mask };

    kern_return_t r =  thread_policy_set( thread, THREAD_AFFINITY_POLICY, 
                                          (thread_policy_t)&policy,
                                           THREAD_AFFINITY_POLICY_COUNT );

    
    if( r == KERN_SUCCESS )
        return (uint64)(integer_t)mask;

    return 0;
}
*/

//-----------------------------------------------------------
bool SysHost::SetCurrentThreadAffinityCpuId( uint32 cpuId )
{
    ASSERT( cpuId < 32 );

    thread_port_t thread = mach_thread_self();

    // It seems macOS does not support 64 bit affinity masks
    thread_affinity_policy_data_t policy = { (integer_t)(1 << cpuId) };

    kern_return_t r =  thread_policy_set( thread, THREAD_AFFINITY_POLICY, 
                                          (thread_policy_t)&policy,
                                           THREAD_AFFINITY_POLICY_COUNT );

    if( r != KERN_SUCCESS )
    {
        Log::Error( "thread_policy_set failed on cpu id %u", cpuId );
        return false;
    }

    return true;
}
