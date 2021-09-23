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

template<typename T>
struct PocessorInfo
{
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* info;
    DWORD                                    size;

    //-----------------------------------------------------------
    PocessorInfo( SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* info, DWORD size )
        : info( info )
        , size( size )
    {}

    //-----------------------------------------------------------
    inline const T& Get( size_t index ) const
    {
        using ProcInfo = SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;

        const byte* ptr = (byte*)this->info;
        const byte* end = ptr + this->size;

        size_t i = 0;
        
        do
        {
            if( i == index )
                return *(T*)&((ProcInfo*)ptr)->Processor;
            
            i++;
            ptr += ((ProcInfo*)ptr)->Size;
        } while( ptr < end );

        Fatal( "PocessorInfo: Index out of range." );
        return *(T*)(nullptr);
    }

    //-----------------------------------------------------------
    inline T& operator[]( size_t  index ) const { return this->Get( index ); }
    inline T& operator[]( ssize_t index ) const
    {
        ASSERT( index >= 0 ); return this->Get( (size_t)index );
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