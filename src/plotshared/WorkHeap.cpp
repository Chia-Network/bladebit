#include "WorkHeap.h"
#include "Util.h"

WorkHeap::WorkHeap( size_t size, byte* heapBuffer )
    : _heap             ( heapBuffer )
    , _heapSize         ( size       )
    , _usedHeapSize     ( 0          )
    , _heapTable        ( 128        )
    , _allocationTable  ( 64         )
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
        byte* buffer = nullptr;

        for( size_t i = 0; i < _heapTable.Length(); i++ )
        {
            if( _heapTable[i].size >= size )
            {
                // Found a big enough slice to use
                _usedHeapSize += size;
                
                buffer = _heapTable[i].address;

                // #TODO: We need to track the size of our buffers for when they are released.

                if( size < _heapTable[i].size )
                {
                    // Split/fragment the current entry
                    _heapTable[i].address += size;
                    _heapTable[i].size    -= size;
                }
                else
                {
                    // We took the whole entry, remove it from the table
                    // and move down any subsequent entries.
                    const size_t remainder = --_heapTable.length - i;
                    if( remainder > 0 )
                        bbmemcpy_t( _heapTable.values + i, _heapTable.values + i + 1, remainder );
                }

                break;
            }
        }

        if( buffer )
            return buffer;

        // No buffer found, we have to wait until buffers are released and then try again
        _releasedBuffers.WaitForProduction();
        ConsumeReleasedBuffers();
}

void WorkHeap::Release( byte* buffer )
{

}

void WorkHeap::AddPendingReleases()
{

}

