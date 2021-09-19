#include "SysHost.h"
#include "Platform.h"
#include "Util.h"

#include <processthreadsapi.h>

//-----------------------------------------------------------
size_t SysHost::GetPageSize()
{
    ASSERT( 0 );
    return 0;
}

//-----------------------------------------------------------
size_t SysHost::GetTotalSystemMemory()
{
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof( statex );

    BOOL r = GlobalMemoryStatusEx( &statex );
    if( !r )
        return 0;

    return statex.ullTotalPhys;
}

//-----------------------------------------------------------
size_t SysHost::GetAvailableSystemMemory()
{
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof( statex );

    BOOL r = GlobalMemoryStatusEx( &statex );
    if( !r )
        return 0;

    return statex.ullAvailPhys;
}

//-----------------------------------------------------------
void* SysHost::VirtualAlloc( size_t size, bool initialize )
{
    SYSTEM_INFO info;
    ::GetSystemInfo( &info );

    const size_t pageSize = (size_t)info.dwPageSize;
    size = CeildDiv( size, pageSize );

    void* ptr = ::VirtualAlloc( NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE );

    if( ptr && initialize )
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
    ASSERT( ptr );
    if( !ptr )
        return;

    // #TODO: Implement me
    ASSERT( 0 ); 
}

//-----------------------------------------------------------
bool SysHost::VirtualProtect( void* ptr, size_t size, VProtect flags )
{
    ASSERT( ptr );

    // #TODO: Implement me
    ASSERT( 0 ); return false;
}

//-----------------------------------------------------------
uint64 SysHost::SetCurrentProcessAffinityMask( uint64 mask )
{
    HANDLE hProcess = ::GetCurrentProcess();

    BOOL r = ::SetProcessAffinityMask( hProcess, mask );
    ASSERT( r );

    return r ? mask : 0;
}

//-----------------------------------------------------------
uint64 SysHost::SetCurrentThreadAffinityMask( uint64 mask )
{
    HANDLE hThread = ::GetCurrentThread();

    uint64 newMask = ::SetThreadAffinityMask( hThread, mask );
    ASSERT( newMask = mask );

    return newMask;
}

//-----------------------------------------------------------
bool SysHost::SetCurrentThreadAffinityCpuId( uint32 cpuId )
{
    // #TODO: Implement me
    ASSERT( 0 ); return false;
}

//-----------------------------------------------------------
void SysHost::InstallCrashHandler()
{
    // #TODO: Implement me
}

//-----------------------------------------------------------
void SysHost::Random( byte* buffer, size_t size )
{
    // #TODO: Implement me
    ASSERT( 0 ); 
}

// #NOTE: This is not thread-safe
//-----------------------------------------------------------
const NumaInfo* SysHost::GetNUMAInfo()
{
    // #TODO: Implement me
    return nullptr;
}

//-----------------------------------------------------------
void SysHost::NumaAssignPages( void* ptr, size_t size, uint node )
{
    // #TODO: Implement me
}

//-----------------------------------------------------------
bool SysHost::NumaSetThreadInterleavedMode()
{
    // #TODO: Implement me
    return false;
}

//-----------------------------------------------------------
bool SysHost::NumaSetMemoryInterleavedMode( void* ptr, size_t size )
{
    // #TODO: Implement me
    return false;
}

//-----------------------------------------------------------
int SysHost::NumaGetNodeFromPage( void* ptr )
{
    // #TODO: Implement me
    return -1;
}