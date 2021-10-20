#include "WorkHeap.h"
#include "Util.h"

WorkHeap::WorkHeap( size_t size, byte* heapBuffer )
    : _heap             ( heapBuffer )
    , _heapSize         ( size       )
    , _usedHeapSize     ( 0          )
    , _heapTable        ( 256u       )
    , _allocationTable  ( 256u       )
{
    ASSERT( _heap     );
    ASSERT( _heapSize );

    // Initialize heap table
    HeapEntry& firstEntry = _heapTable.Push();
    firstEntry.address = _heap;
    firstEntry.size    = _heapSize;
}

WorkHeap::~WorkHeap()
{
}

byte* WorkHeap::Alloc( size_t size )
{
    ASSERT( size <= _heapSize );

    // If we have such a buffer available, grab it, if not, we will
    // have to wait for enough deallocations in order to continue
    for( ;; )
    {
        // First, add any pending released buffers back to the heap
        AddPendingReleases();

        byte* buffer = nullptr;

        for( size_t i = 0; i < _heapTable.Length(); i++ )
        {
            if( _heapTable[i].size >= size )
            {
                // Found a big enough slice to use
                _usedHeapSize += size;
                
                buffer = _heapTable[i].address;
                
                // We need to track the size of our buffers for when they are released.
                HeapEntry& allocation = _allocationTable.Push();
                allocation.address = buffer;
                allocation.size    = size;

                if( size < _heapTable[i].size )
                {
                    // Split/fragment the current entry
                    _heapTable[i].address += size;
                    _heapTable[i].size    -= size;
                }
                else
                {
                    // We took the whole entry, remove it from the table.
                    _heapTable.Remove( i );
                }

                break;
            }
        }

        if( buffer )
            return buffer;

        // No buffer found, we have to wait until buffers are released and then try again
        _releaseSignal.Wait();
    }
}

void WorkHeap::Release( byte* buffer )
{
    ASSERT( buffer );
    ASSERT( buffer >= _heap && buffer < _heap + _heapSize );

    _pendingReleases.Enqueue( buffer );
    _releaseSignal.Signal();
}

void WorkHeap::AddPendingReleases()
{
    const int BUFFER_SIZE = 128;
    byte* releases[BUFFER_SIZE];

    // Grab pending releases
    int count;
    while( count = _pendingReleases.Dequeue( releases, BUFFER_SIZE ) )
    {
        for( int i = 0; i < count; i++ )
        {
            byte* buffer = releases[i];

            // Find the buffer in the allocation table
            #if _DEBUG
                bool foundAllocation = false;
            #endif
            for( size_t j = 0; j < _allocationTable.Length(); j++ )
            {
                if( _allocationTable[j].address == buffer )
                {
                    #if _DEBUG
                        foundAllocation = true;
                    #endif

                    HeapEntry allocation = _allocationTable[j];
                    _allocationTable.UnorderedRemove( j );

                    _usedHeapSize -= allocation.size;

                    size_t insertIndex = 0;

                    // Find where to place the allocation back into the 
                    for( ; insertIndex < _heapTable.Length(); insertIndex++ )
                    {
                        if( _heapTable[insertIndex].address > buffer )
                            break;
                    }

                    // When adding the released buffer (entry) back to the heap table,
                    // one of 4 scenarios can happen:
                    // 1: Entry can be merged with the left entry, don't create a new entry, simply extend it.
                    // 2: Entry can be merged with the right entry, don't create a new entry, 
                    //      change the start address of the right entry and extend size.
                    // 3: Entry can be merged with both left and right entry, 
                    //      extend the left entry to cover the released entry and the right entry,
                    //      then remove the right entry.
                    // 4: Entry cannot be merged with the left or the right entry, create a new entry


                    // Case 1? Can we merge with the left entry?
                    if( insertIndex > 0 && _heapTable[insertIndex - 1].EndAddress() == buffer )
                    {
                        // Extend size to the released entry
                        HeapEntry& entry = _heapTable[insertIndex - 1];
                        entry.size += allocation.size;

                        // Case 3? See if we also need to merge with the right entry (fills a hole between 2 entries).
                        if( insertIndex < _heapTable.Length() && allocation.EndAddress() == _heapTable[insertIndex].address )
                        {
                            // Extend size to the right entry
                            entry.size += _heapTable[insertIndex].size;
                            _heapTable.Remove( insertIndex );
                        }
                    }
                    // Case 2? Can we merge with the right entry
                    else if( insertIndex < _heapTable.Length() && allocation.EndAddress() == _heapTable[insertIndex].address )
                    {
                        // Don't create a new allocation, merge with the right entry
                        _heapTable[insertIndex].address = allocation.address;
                        _heapTable[insertIndex].size   += allocation.size;
                    }
                    // Case 4: Insert a new entry, no merges
                    else
                    {
                        // We need to insert a new entry
                        _heapTable.Insert( allocation, insertIndex );
                    }

                    break;
                }
            }

            ASSERT( foundAllocation );
        }
    }
}

