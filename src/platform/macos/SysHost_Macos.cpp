#include "SysHost.h"
#include "Platform.h"
#include "util/Util.h"
#include "util/Log.h"
#include <errno.h>
#include <mach/mach.h>
#include <fcntl.h>
#include <pthread.h>

#if !defined( BB_IS_HARVESTER )
    #include <sodium.h>
#endif

//-----------------------------------------------------------
size_t SysHost::GetPageSize()
{
    vm_size_t pageSize = 0;
    kern_return_t r = host_page_size( mach_host_self(), & pageSize );
    PanicIf( r != 0, "host_page_size failed with error: %d.", (int32)r );

    return (size_t)pageSize;
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
    // #TODO: Remove initialize

    const size_t      pageSize = GetPageSize();
    const mach_port_t task     = mach_task_self();

    const vm_size_t allocSize = (vm_size_t)RoundUpToNextBoundary( size, (int)pageSize ) + pageSize;
    vm_address_t ptr = 0;
    kern_return_t r = vm_allocate( task,&ptr,allocSize, TRUE );

    if( r != 0 )
    {
        Log::Line( "Warning: vm_allocate() failed with error: %d .", (int32)r );
        return nullptr;
    }
    ASSERT( ptr );

    // #TODO: Use a hinting system for this.
    // Hint the memory to be accessed sequentially
    r = vm_behavior_set( task, ptr, allocSize, VM_BEHAVIOR_SEQUENTIAL );
    if( r != 0 )
    {
        #if !defined( BB_IS_HARVESTER )
            Log::Line( "Warning: vm_behavior_set() failed with error: %d .", (int32)r );
        #endif
    }

    // Try wiring it
    r = vm_wire( mach_host_self(), task, ptr, allocSize, VM_PROT_READ | VM_PROT_WRITE );
    if( r != 0 )
    {
        #if !defined( BB_IS_HARVESTER )
            Log::Line( "Warning: vm_wire() failed with error: %d .", (int32)r );
        #endif
    }

    // Store page size
    // #TODO: Have the user specify an alignment instead so we don't have to store at page size boundaries.
    (*(size_t*)ptr) = allocSize;

    return (void*)(uintptr_t)(ptr + pageSize);
//    const size_t pageSize = (size_t)getpagesize();
//
//    // Align size to page boundary
//    size = RoundUpToNextBoundary( size, (int)pageSize );
//
//    void* ptr = mmap( NULL, size,
//        PROT_READ | PROT_WRITE,
//        MAP_ANONYMOUS | MAP_PRIVATE,
//        -1, 0
//    );
//
//    if( ptr == MAP_FAILED )
//    {
//        #if _DEBUG
//            const int err = errno;
//            Log::Line( "Error: mmap() returned %d (0x%x).", err, err );
//            ASSERT( 0 );
//        #endif
//
//        return nullptr;
//    }
//    else if( initialize )
//    {
//        // Initialize memory
//        // (since physical pages are not allocated until the actual pages are accessed)
//
//        byte* page = (byte*)ptr;
//
//        const size_t pageCount = size / pageSize;
//        const byte*  endPage   = page + pageCount * pageSize;
//
//        do
//        {
//            *page = 0;
//            page += pageSize;
//        } while( page < endPage );
//    }
//
//    return ptr;
}

//-----------------------------------------------------------
void SysHost::VirtualFree( void* ptr )
{
    ASSERT( ptr );
    if( !ptr )
        return;

    // #TODO: Have user specify size instead, so we can store the size not at page alignment.
    const size_t pageSize = GetPageSize();

    byte* realPtr    = ((byte*)ptr) - pageSize;
    const size_t size = *((size_t*)realPtr);

//    munmap( realPtr, size );
    kern_return_t r = vm_deallocate( mach_task_self(), (vm_address_t)realPtr, (vm_size_t)size );
    if( r != 0 )
        Log::Line("Warning: vm_deallocate() failed with error %d.", (int32)r );
}

//-----------------------------------------------------------
bool SysHost::VirtualProtect( void* ptr, size_t size, VProtect flags )
{
    ASSERT( ptr );
    ASSERT( size );

    vm_prot_t prot = VM_PROT_NONE;

    // if( IsFlagSet( flags, VProtect::NoAccess ) )
    // {
    //     prot = PROT_NONE;
    // }
    // else
    // {
    if( IsFlagSet( flags, VProtect::Read ) )
        prot |= VM_PROT_READ;
    if( IsFlagSet( flags, VProtect::Write ) )
        prot |= VM_PROT_WRITE;
    // }

    const kern_return_t r = vm_protect( mach_task_self(), (vm_address_t)ptr, (vm_size_t)size, false, prot );
    ASSERT( !r );

    return r == 0;
}

//-----------------------------------------------------------
bool SysHost::SetCurrentThreadAffinityCpuId( uint32 cpuId )
{
    //  #NOTE: Thread affinity it not supported on macOS.
    //          These will always return "not implemented".
//    ASSERT( cpuId < 32 );
//
//    thread_port_t thread = mach_thread_self();
//
//    // It seems macOS does not support 64 bit affinity masks
//    thread_affinity_policy_data_t policy = { (integer_t)(1 << cpuId) };
//
//    kern_return_t r =  thread_policy_set( thread, THREAD_AFFINITY_POLICY,
//                                          (thread_policy_t)&policy,
//                                           THREAD_AFFINITY_POLICY_COUNT );
//
//    if( r != KERN_SUCCESS )
//    {
//        Log::Error( "thread_policy_set failed on cpu id %lld with error: %d.", cpuId, (int64)r );
//        return false;
//    }

    return true;
}

//-----------------------------------------------------------
void SysHost::InstallCrashHandler()
{
    // #TODO: Implement me
}

//-----------------------------------------------------------
void SysHost::DumpStackTrace()
{
    // #TODO: Implement me
}

//-----------------------------------------------------------
void SysHost::Random( byte* buffer, size_t size )
{
    #if defined(BB_IS_HARVESTER)
        Panic( "getrandom not supported on bladebit_harvester target.");
    #else
    randombytes_buf( buffer, size );
    #endif
}


///
/// NUMA (no support on macOS)
///
//-----------------------------------------------------------
const NumaInfo* SysHost::GetNUMAInfo()
{
    // Not supported
    return nullptr;
}

//-----------------------------------------------------------
void SysHost::NumaAssignPages( void* ptr, size_t size, uint node )
{
    // Not supported
}

//-----------------------------------------------------------
bool SysHost::NumaSetThreadInterleavedMode()
{
    // Not supported
    return false;
}

//-----------------------------------------------------------
bool SysHost::NumaSetMemoryInterleavedMode( void* ptr, size_t size )
{
    // Not supported
    return false;
}

//-----------------------------------------------------------
int SysHost::NumaGetNodeFromPage( void* ptr )
{
    // Not supported
    return 0;
}
