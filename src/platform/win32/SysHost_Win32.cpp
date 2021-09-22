#include "SysHost.h"
#include "Platform.h"
#include "Util.h"
#include "util//Log.h"

#include <processthreadsapi.h>
#include <systemtopologyapi.h>

/*
* Based on source from libSodium: ref: https://github.com/jedisct1/libsodium/blob/master/src/libsodium/randombytes/sysrandom/randombytes_sysrandom.c
* ISC License
* Copyright( c ) 2013 - 2020
* Frank Denis <j at pureftpd dot org>
*/
#define RtlGenRandom SystemFunction036
extern "C" BOOLEAN NTAPI RtlGenRandom( PVOID RandomBuffer, ULONG RandomBufferLength );
#pragma comment( lib, "advapi32.lib" )

//-----------------------------------------------------------
size_t SysHost::GetPageSize()
{
    SYSTEM_INFO info;
    ::GetSystemInfo( &info );

    const size_t pageSize = (size_t)info.dwPageSize;
    return pageSize;
}

//-----------------------------------------------------------
size_t SysHost::GetTotalSystemMemory()
{
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof( statex );

    BOOL r = GlobalMemoryStatusEx( &statex );
    if( !r )
        return 0;

    // #TODO: Return total virtual... We will let the user use a page file.
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

    // #TODO: Return total virtual... We will let the user use a page file.
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
        // Fault memory pages

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

    const BOOL r = ::VirtualFree( (LPVOID)ptr, 0, MEM_RELEASE );
    if( !r )
    {
        const DWORD err = GetLastError();
        Log::Error( "VirtualFree() failed with error: %d", err );
    }
}

//-----------------------------------------------------------
bool SysHost::VirtualProtect( void* ptr, size_t size, VProtect flags )
{
    ASSERT( ptr );

    // #TODO: Implement me
    return true;
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

    const uint64 oldMask = ::SetThreadAffinityMask( hThread, mask );
    
    if( oldMask == 0 )
        return 0;

    return mask;
}

//-----------------------------------------------------------
bool SysHost::SetCurrentThreadAffinityCpuId( uint32 cpuId )
{
    // #TODO: Implement me to use masks > 64bits
    const uint64 mask = 1ull << ((uint64)cpuId);
    return mask == SetCurrentThreadAffinityMask( mask );
}

//-----------------------------------------------------------
void SysHost::InstallCrashHandler()
{
    // #TODO: Implement me
}

//-----------------------------------------------------------
void SysHost::Random( byte* buffer, size_t size )
{
    if( !RtlGenRandom( (PVOID)buffer, (ULONG)size ) )
    {
        Fatal( "System entropy gen failure." );
    }    
}

// #NOTE: This is not thread-safe
//-----------------------------------------------------------
const NumaInfo* SysHost::GetNUMAInfo()
{
    // #TODO: Check _WIN32_WINNT >= 0x0601

    static NumaInfo  _info;
    static NumaInfo* _pInfo = nullptr;

    if( !_pInfo )
    {
        uint nodeCount = 0;

        if( !GetNumaHighestNodeNumber( (PULONG)&nodeCount ) )
        {
            const DWORD err = GetLastError();
            Log::Error( "Warning: Failed to get NUMA info with error %d (0x%x).", err, err );
            return nullptr;
        }

        if( nodeCount < 2 )
            return nullptr;

        ZeroMem( &_pInfo );

        uint procGroupCount = 0;
        uint totalCpuCount  = 0;

        // Get number of processor groups in this system
        procGroupCount = (uint)GetActiveProcessorGroupCount();
        if( procGroupCount == 0 )
        {
            const DWORD err = GetLastError();
            Fatal( "GetActiveProcessorGroupCount() failed with error: %d (0x%x).", err, err );
        }

        // Get the total number of active CPUs
        totalCpuCount = (uint)GetActiveProcessorCount( ALL_PROCESSOR_GROUPS );
        if( totalCpuCount == 0 )
        {
            const DWORD err = GetLastError();
            Fatal( "GetActiveProcessorCount() failed with error: %d (0x%x).", err, err );
        }

//         _info.procGroup.length = procGroupCount;
//         _info.procGroup.values = (uint16*)malloc( procGroupCount * sizeof( uint16 ) );
//         memset( _info.procGroup.values, 0, procGroupCount * sizeof( uint16 ) );

        for( uint i = 0; i < procGroupCount; i++ )
        {
            // Get the number of processors in this group
            const uint procCount = (uint)GetActiveProcessorCount( (WORD)i );
            if( procCount == 0 )
            {
                const DWORD err = GetLastError();
                Fatal( "GetActiveProcessorCount( %u ) failed with error: %d (0x%x).", i, err, err );
            }

            GROUP_AFFINITY grp = { 0 };

            if( !GetNumaNodeProcessorMaskEx( (USHORT)i, &grp ) )
            {
                const DWORD err = GetLastError();
                Fatal( "GetNumaNodeProcessorMaskEx( %u ) failed with error: %d (0x%x).", i, err, err );
            }


            // Count nodes int 

        }

        _info.nodeCount = nodeCount;
        _info.cpuCount  = totalCpuCount;

        // #NOTE: Based on:
        // #See: https://docs.microsoft.com/en-us/windows/win32/memory/allocating-memory-from-a-numa-node
        for( uint8 i = 0; i < (uint8)totalCpuCount; i++ )
        {
            // int8 proc;

            // GetNumaProcessorNode()
            // uint64 cpuMask = 0;
            
            // if( !GetNumaNodeProcessorMask( i, (PULONGLONG)&cpuMask ) )
            // {
            //     DWORD err = GetLastError();
            //     Fatal( "Failed to get NUMA node cpu count eith error %d (0x%x).", err, err );
            // }

            
        }
    }
    
    
    return _pInfo;
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