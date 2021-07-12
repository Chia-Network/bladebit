#include "SysHost.h"
#include "Platform.h"
#include "Util.h"

#include <processthreadsapi.h>

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