#pragma once
#include "threading/AutoResetSignal.h"
#include "util/SPCQueue.h"
#include "util/Array.h"
#include "plotdisk/DiskPlotConfig.h"

// A simple heap to be used as our working buffer provider
// for doing plotting work in memory and I/O operations.
// It is meant to have a very small amount of allocations, therefore
// allocations are tracked in a small table that is searched linearly.
class WorkHeap
{
private:
    // Represents a portion of unallocated space in our heap/work buffer
    struct HeapEntry
    {
        byte*  address;
        size_t size;

        inline byte* EndAddress() const { return address + size; }

        inline bool CanAllocate( size_t allocationSize, size_t alignment )
        {
            ASSERT( allocationSize == RoundUpToNextBoundaryT( allocationSize, alignment ) );
            ASSERT( RoundUpToNextBoundaryT( allocationSize, alignment ) <= 0x7FFFFFFFFFFFFFFF );
            
            const intptr_t alignedAddress = (intptr_t)alignment * CDivT( (intptr_t)address, (intptr_t)alignment );
            return ( (intptr_t)EndAddress() - alignedAddress ) >= (intptr_t)allocationSize;
        }
    };

public:
    WorkHeap( size_t size, byte* heapBuffer );
    ~WorkHeap();

    void ResetHeap( const size_t heapSize, void* heapBuffer );

    // Allocate a buffer on the heap.
    // If no space is available it will block until
    // some space has become available again.
    byte* Alloc( size_t size, size_t alignment = sizeof( intptr_t ), bool blockUntilFreeBuffer = true, Duration* accumulator = nullptr );

    // Add a to the pending release list.
    // The buffer won't actually be released until an allocation
    // attempt is called or an explicit call AddPendingReleases() to is made.
    // This is meant to be called by a producer thread.
    bool  Release( byte* buffer );

    bool CanAllocate( size_t size, size_t alignment = sizeof( intptr_t ) ) const;

    inline size_t FreeSize() const { return _usedHeapSize - _heapSize; }

    inline size_t HeapSize() const { return _heapSize; }

    inline const void* Heap() const { return _heap; }


    // Makes pending released allocations available to the heap for allocation again.
    void CompletePendingReleases();

private:

private:
    byte*                _heap;
    size_t               _heapSize;             // Size of our work heap
    size_t               _usedHeapSize;         // How much heap space is currently being used

    Array<HeapEntry>     _heapTable;            // Tracks unallocated space in our work heap
    Array<HeapEntry>     _allocationTable;      // Tracks sizes and for currently allocated buffers. Used when performing releases.

    // #TODO: Make SPCQueue dynamically allocated.
    SPCQueue<byte*, BB_DISK_QUEUE_MAX_CMDS> _pendingReleases;      // Released buffers waiting to be re-added to the heap table
    AutoResetSignal      _releaseSignal;        // Used to signal that there's pending released buffers

    // std::atomic<size_t>  _freeHeapSize = 0;     // Current free heap size from the perspective of the consumer thread (the allocating thread)
    // std::atomic<size_t>  _waitingSize  = 0;     // Required size for the next allocation. If the next release
                                                // does not add up to this size, then it won't signal it
};


