#pragma once

enum class VProtect : uint
{
    None = 0,                   // No protection / Clear protection

    Read     = 1 << 0,          // Allow reads. (Make pages read-only.)
    Write    = 1 << 1,          // Allow writes. (Make pages write-only.)
    NoAccess = 1 << 2,          // No memory access allowed at all

    ReadWrite = Read | Write    // Make pages read/writable
};
ImplementFlagOps( VProtect );

struct NumaInfo
{
    uint        nodeCount;  // How many NUMA nodes in the system
    uint        cpuCount;   // Total cpu count used by nodes
    Span<uint>* cpuIds;     // CPU ids of each node
};

class SysHost
{
public:

    // Get the system page size in bytes
    static size_t GetPageSize();

    // Get total physical system ram in bytes
    static size_t GetTotalSystemMemory();

    /// Gets the currently available (unused) system ram in bytes
    static size_t GetAvailableSystemMemory();

    /// Create an allocation in the virtual memory space
    /// If initialize == true, then all pages are touched so that
    /// the pages are actually assigned.
    static void* VirtualAlloc( size_t size, bool initialize = false );
    
    static void VirtualFree( void* ptr );

    static bool VirtualProtect( void* ptr, size_t size, VProtect flags = VProtect::NoAccess );

    /// Set the processor affinity mask for the current process
    static uint64 SetCurrentProcessAffinityMask( uint64 mask );

    /// Set the processor affinity mask for the current thread
    static uint64 SetCurrentThreadAffinityMask( uint64 mask );

    /// Set the processor affinity mask to a specific cpu id for the current thread
    static bool   SetCurrentThreadAffinityCpuId( uint32 cpuId );

    /// Install a crash handler to dump stack traces upon crash
    static void InstallCrashHandler();

    /// Generate random data
    /// (We basically do what libsodium does here)
    static void Random( byte* buffer, size_t size );

    /// Get system's NUMA info, if it has any
    static const NumaInfo* GetNUMAInfo();

    /// Assign memory pages to a NUMA node
    static void NumaAssignPages( void* ptr, size_t size, uint node );

    /// Set interleave NUMA mode for allocations in the calling thread
    static bool NumaSetThreadInterleavedMode();

    /// Set interleave NUMA mode for the specified memory regions.
    /// NOTE: Pages must not yet be faulted.
    static bool NumaSetMemoryInterleavedMode( void* ptr, size_t size );

    /// Get the node a memory page belongs to.
    /// Returns a negative value upon failure.
    /// NOTE: Pages must first be faulted on linuz.
    static int NumaGetNodeFromPage( void* ptr );

};