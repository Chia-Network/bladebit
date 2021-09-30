#include "SysHost.h"
#include "Platform.h"
#include "Util.h"
#include "util//Log.h"

#include <processthreadsapi.h>
#include <systemtopologyapi.h>
#include <psapi.h>

/*
* Based on source from libSodium: ref: https://github.com/jedisct1/libsodium/blob/master/src/libsodium/randombytes/sysrandom/randombytes_sysrandom.c
* ISC License
* Copyright( c ) 2013 - 2020
* Frank Denis <j at pureftpd dot org>
*/
#define RtlGenRandom SystemFunction036
extern "C" BOOLEAN NTAPI RtlGenRandom( PVOID RandomBuffer, ULONG RandomBufferLength );
#pragma comment( lib, "advapi32.lib" )

// Helper structs to help us iterate these variable-length structs
template<typename T>
class PocessorInfoIter
{
    using ProcInfo = SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;

    byte*       _current;
    const byte* _end    ;

public:
    //-----------------------------------------------------------
    inline PocessorInfoIter( SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* info, DWORD size )
        : _current( (byte*)info )
        , _end    ( ((byte*)info) + size )
    {}

    //-----------------------------------------------------------
    inline bool HasNext() const
    {
        return _current < _end;
    }

    //-----------------------------------------------------------
    inline T& Next()
    {
        ASSERT( this->HasNext() );

        ProcInfo* info = (ProcInfo*)_current;

        T& r = *(T*)&info->Processor;

        _current += info->Size;
        return r;
    }
};

static GROUP_RELATIONSHIP* _procGroupInfo = nullptr;


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
uint SysHost::GetLogicalCPUCount()
{
    return (uint)GetActiveProcessorCount( ALL_PROCESSOR_GROUPS );
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
// uint64 SysHost::SetCurrentProcessAffinityMask( uint64 mask )
// {
//     HANDLE hProcess = ::GetCurrentProcess();

//     BOOL r = ::SetProcessAffinityMask( hProcess, mask );
//     ASSERT( r );

//     return r ? mask : 0;
// }

// //-----------------------------------------------------------
// uint64 SysHost::SetCurrentThreadAffinityMask( uint64 mask )
// {
//     HANDLE hThread = ::GetCurrentThread();
// 
//     const uint64 oldMask = ::SetThreadAffinityMask( hThread, mask );
//     
//     if( oldMask == 0 )
//         return 0;
// 
//     return mask;
// }

//-----------------------------------------------------------
bool SysHost::SetCurrentThreadAffinityCpuId( uint32 cpuId )
{
    ASSERT( cpuId < (uint)GetActiveProcessorCount( ALL_PROCESSOR_GROUPS ) );

    HANDLE hThread = ::GetCurrentThread();

    // #TODO: We might have to check for single-node system with more than 64 threads.
    // If in NUMA, we need to find the correct process group given a cpuId
    const NumaInfo* numa = GetNUMAInfo();
    if( numa )
    {
        ASSERT( _procGroupInfo );

        WORD processGroupId = 0;

        // Shave-out any process groups below the requested one
        for( WORD i = 0; i < _procGroupInfo->ActiveGroupCount; i++ )
        {
            const uint groupProcCount = _procGroupInfo->GroupInfo[i].ActiveProcessorCount;
            if( cpuId < groupProcCount )
            {
                processGroupId = i;
                break;
            }

            cpuId -= groupProcCount;
        }
        
        // Move this thread to the target process group
        GROUP_AFFINITY grpAffinity, prevGprAffinity;

        ZeroMem( &grpAffinity );
        grpAffinity.Mask  = 1ull << cpuId;
        grpAffinity.Group = processGroupId;
        if( !SetThreadGroupAffinity( hThread, &grpAffinity, &prevGprAffinity ) )
        {
            const DWORD err = GetLastError();
            Log::Error( "Error: Failed to set thread group affinity with error: %d (0x%x).", err, err );
            return false;
        }
    }

    const uint64 mask    = 1ull << cpuId;
    const uint64 oldMask = (uint64)::SetThreadAffinityMask( hThread, mask );

    if( oldMask == 0 )
    {
        const DWORD err = GetLastError();
        Log::Error( "Error: Failed to set thread affinity with error: %d (0x%x).", err, err );
    }

    return oldMask != 0;
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

// #SEE: https://docs.microsoft.com/en-us/windows/win32/procthread/numa-support
// #SEE: https://docs.microsoft.com/en-us/windows/win32/procthread/processor-groups
// #NOTE: This is not thread-safe on the first time is called
//-----------------------------------------------------------
const NumaInfo* SysHost::GetNUMAInfo()
{
    // #TODO: Check _WIN32_WINNT >= 0x0601
    //                              Build 20348
    // and support new NUMA method/API.

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
        
        // The above returns the highest node index (0-based), we want the count.
        nodeCount++;

        if( nodeCount < 2 )
            return nullptr;

        ZeroMem( &_info );

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

        // Allocate required buffers
        _info.cpuIds = ( Span<uint>* )malloc( sizeof( Span<uint> ) * nodeCount );
        FatalIf( !_info.cpuIds, "Failed to allocate NUMA node buffer." );
        memset( _info.cpuIds, 0, sizeof( Span<uint> ) * nodeCount );

        uint* cpuIds = (uint*)malloc( sizeof( uint ) * totalCpuCount );
        FatalIf( !cpuIds, "Failed to allocate CPU id buffer." );
        memset( cpuIds, 0, sizeof( uint ) * totalCpuCount );


        // Get nodes & process group information
        DWORD nodeInfoLength = 0, procInfoLength = 0;
        DWORD result = 0;

        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* nodeInfo = nullptr, *procInfo = nullptr;

        // Get node info size
        if( GetLogicalProcessorInformationEx( RelationNumaNode, nullptr, &nodeInfoLength ) )
            Fatal( "Unexpected result from GetLogicalProcessorInformationEx( RelationNumaNode )." );

        result = GetLastError();
        if( result != ERROR_INSUFFICIENT_BUFFER )
            Fatal( "GetLogicalProcessorInformationEx( RelationNumaNode, null ) with error: %d (0x%x).", result, result );
        
        ASSERT( nodeInfoLength >= ( sizeof( DWORD ) + sizeof( LOGICAL_PROCESSOR_RELATIONSHIP ) ) );


        // Get process info size
        if( GetLogicalProcessorInformationEx( RelationGroup, nullptr, &procInfoLength ) )
            Fatal( "Unexpected result from GetLogicalProcessorInformationEx( RelationGroup )." );

        result = GetLastError();
        if( result != ERROR_INSUFFICIENT_BUFFER )
            Fatal( "GetLogicalProcessorInformationEx( RelationGroup, null  ) with error: %d (0x%x).", result, result );

        ASSERT( procInfoLength >= ( sizeof( DWORD ) + sizeof( LOGICAL_PROCESSOR_RELATIONSHIP ) ) );

        // Allocate the buffers and fetch the actual info now
        nodeInfo = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)malloc( nodeInfoLength );
        procInfo = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)malloc( procInfoLength );

        if( !nodeInfo )
            Fatal( "Failed to allocate node info buffer." );
        if( !procInfo )
            Fatal( "Failed to allocate processor info buffer." );
        
        if( !GetLogicalProcessorInformationEx( RelationNumaNode, nodeInfo, &nodeInfoLength ) )
        {
            const DWORD err = GetLastError();
            Fatal( "GetLogicalProcessorInformationEx( RelationNumaNode ) failed with error: %d (0x%x).", err, err );
        }

        if( !GetLogicalProcessorInformationEx( RelationGroup, procInfo, &procInfoLength ) )
        {
            const DWORD err = GetLastError();
            Fatal( "GetLogicalProcessorInformationEx( RelationGroup ) failed with error: %d (0x%x).", err, err );
        }

        ASSERT( procInfo->Size == procInfoLength ); // Only expect a single instance here


        // Save an instance of process group info as we need to refer to it when setting thread affinity
        _procGroupInfo = &procInfo->Group;
        
        // Get info from each node
        for( PocessorInfoIter<NUMA_NODE_RELATIONSHIP> numaIter( nodeInfo, nodeInfoLength ); numaIter.HasNext(); )
        {
            const NUMA_NODE_RELATIONSHIP& node = numaIter.Next();
            ASSERT( node.NodeNumber < nodeCount );
            ASSERT( node.GroupMask.Group < procInfo->Group.ActiveGroupCount );

            const WORD                  targetGroupId = node.GroupMask.Group;
            const PROCESSOR_GROUP_INFO& targetGroup   = procInfo->Group.GroupInfo[targetGroupId];

            // Find the starting cpuId for this group by 
            // adding all the active processors on the groups before ours
            uint cpuBase = 0;
            for( WORD i = 0; i < targetGroupId; i++ )
                cpuBase += procInfo->Group.GroupInfo[i].ActiveProcessorCount;

            // Save CPUids for this node
            uint* nodeCpus = cpuIds;
            for( BYTE i = 0; i < targetGroup.MaximumProcessorCount; i++ )
            {
                if( targetGroup.ActiveProcessorMask & ( 1ull << i ) )
                    *nodeCpus++ = cpuBase + i;
            }

            ASSERT( (intptr_t)( nodeCpus - cpuIds ) == (intptr_t)targetGroup.ActiveProcessorCount );

            // Save node info
            _info.cpuIds[node.NodeNumber].length = targetGroup.ActiveProcessorCount;
            _info.cpuIds[node.NodeNumber].values = cpuIds;

            cpuIds += targetGroup.ActiveProcessorCount;
            ASSERT( cpuIds == nodeCpus );
        }

        // All done
        _info.nodeCount = nodeCount;
        _info.cpuCount  = totalCpuCount;
        _pInfo          = &_info;
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
    // Not a thing on windows
    // #TODO: Remove this function
    return false;
}

//-----------------------------------------------------------
bool SysHost::NumaSetMemoryInterleavedMode( void* ptr, size_t size )
{
    ASSERT( ptr && size );

    ULONG nodeCount = 0;
    
    if( !GetNumaHighestNodeNumber( (PULONG)&nodeCount ) )
    {
        const DWORD err = GetLastError();
        Log::Error( "Failed to get NUMA nodes to interleave memory with error %d (0x%x).", err, err );
        return false;
    }

    nodeCount++;

    const size_t pageSize   = GetPageSize();
    const size_t pageCount  = size / pageSize;

    const size_t blockStride = pageSize * nodeCount;
    const size_t blockCount  = pageCount / nodeCount;

    const byte* pages    = (byte*)ptr;
    const byte* endPage  = pages + pageCount  * pageSize;
    const byte* endBlock = pages + blockCount * blockStride;

    HANDLE gProcess = GetCurrentProcess();

    DWORD dwNodes = (DWORD)nodeCount;

    while( pages < endBlock )
    {
        for( DWORD i = 0; i < nodeCount; i++ )
        {
            LPVOID r = VirtualAllocExNuma( 
                            gProcess, 
                            (LPVOID)pages, pageSize,
                            MEM_COMMIT, 
                            PAGE_READWRITE,
                            i );

            pages += pageSize;
            if( !r )
            {
                const DWORD err = GetLastError();
                Log::Error( "Failed to assigned memory to NUMA node with error: %d (0x%x).", err, err );
                return false;
            }
        }
    }

    DWORD node = 0;
    while( pages < endPage )
    {
        LPVOID r = VirtualAllocExNuma( 
                            gProcess, 
                            (LPVOID)pages, pageSize,
                            MEM_COMMIT, 
                            PAGE_READWRITE,
                            node++ );

        if( !r )
        {
            const DWORD err = GetLastError();
            Log::Error( "Failed to assigned memory to NUMA node with error: %d (0x%x).", err, err );
            return false;
        }

        pages += pageSize;
    }

    return true;
}

//-----------------------------------------------------------
int SysHost::NumaGetNodeFromPage( void* ptr )
{
    // #TODO: Implement me
    // PSAPI_WORKING_SET_EX_INFORMATION info;

    // BOOL r = QueryWorkingSetEx( GetCurrentProcess(), (PVOID)&info, (DWORD)GetPageSize() );
    // if( !r )
    // {
    //     const DWORD err = GetLastError();
    //     Log::Error( "Failed to call QueryWorkingSetEx with error: %d (0x%x).", err, err );
    // }

    return -1;
}