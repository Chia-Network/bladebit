#include "WorkHeap.h"
#include "util/Util.h"
#include "util/Log.h"

//-----------------------------------------------------------
WorkHeap::WorkHeap( size_t size, byte* heapBuffer )
    : _heap            ( heapBuffer )
    , _heapSize        ( size       )
    , _usedHeapSize    ( 0          )
    , _heapTable       ( 256u       )
    , _allocationTable ( 256u       )
{
    ASSERT( _heap     );
    ASSERT( _heapSize );

    // Initialize heap table
    HeapEntry& firstEntry = _heapTable.Push();
    firstEntry.address = _heap;
    firstEntry.size    = _heapSize;
}

//-----------------------------------------------------------
WorkHeap::~WorkHeap()
{
}

//-----------------------------------------------------------
void WorkHeap::ResetHeap( const size_t heapSize, void* heapBuffer )
{
    ASSERT( _allocationTable.Length() == 0 );
    ASSERT( _heapTable.Length()       == 1 );

    _heap      = (byte*)heapBuffer;
    _heapSize  = heapSize;
    _heapTable[0].address = (byte*)heapBuffer;
    _heapTable[0].size    = heapSize;
}

//-----------------------------------------------------------
byte* WorkHeap::Alloc( size_t size, size_t alignment, bool blockUntilFreeBuffer, Duration* accumulator )
{
    ASSERT( size );
    size = alignment * CDivT( size, alignment );

    ASSERT( size <= _heapSize );

    // If we have such a buffer available, grab it, if not, we will
    // have to wait for enough deallocations in order to continue
    for( ;; )
    {
        // First, add any pending released buffers back to the heap
        CompletePendingReleases();

        byte* buffer = nullptr;

        for( size_t i = 0; i < _heapTable.Length(); i++ )
        {
            HeapEntry& entry = _heapTable[i];
            ASSERT( entry.EndAddress() <= _heap + _heapSize );

            if( entry.CanAllocate( size, alignment ) )
            {
                // Found a big enough slice to use
                byte* alignedAddress = (byte*)( alignment * CDivT( (uintptr_t)entry.address, (uintptr_t)alignment ) );

                _usedHeapSize += size;
                
                // We need to track the size of our buffers for when they are released.
                HeapEntry& allocation = _allocationTable.Push();
                allocation.address     = alignedAddress;
                allocation.size        = size;

                const size_t addressOffset = (size_t)( alignedAddress - entry.address );
                Log::Debug( "+ Allocated @ 0x%p : %llu", alignedAddress, size );

                if( addressOffset )
                {
                    // If we did not occupy all of the remaining size in the entry, 
                    // we need to insert a new entry to the right of the current one.
                    const size_t remainder = (size_t)( entry.EndAddress() - allocation.EndAddress() );

                    // Adjust the current entry's size to the space between the aligned address and the base address
                    entry.size = addressOffset;
                    ASSERT( entry.size );

                    if( remainder )
                    {
                        HeapEntry& rightEntry = _heapTable.Insert( i + 1 );
                        rightEntry.address    = allocation.EndAddress();
                        rightEntry.size       = remainder;
                        ASSERT( rightEntry.size );
                    }
                }
                else
                {
                    if( size < _heapTable[i].size )
                    {
                        // Split/fragment the current entry
                        entry.address += size;
                        entry.size    -= size;
                        ASSERT( entry.size );
                    }
                    else
                    {
                        // We took the whole entry, remove it from the table.
                        _heapTable.Remove( i );
                    }
                }

                buffer = alignedAddress;
                break;
            }
        }

        if( buffer )
            return buffer;

        if( !blockUntilFreeBuffer )
            return nullptr;

        // No buffer found, we have to wait until buffers are released and then try again
        // Log::Line( "*************************** No Buffers available waiting..." );
        auto timer = TimerBegin();
        _releaseSignal.Wait();

        if( accumulator )
        {
            (*accumulator) += TimerEndTicks( timer );
        }


        // Log::Line( " *************************** Waited %.6lf seconds for a buffer.", TimerEnd( timer ) );
    }
}

//-----------------------------------------------------------
bool WorkHeap::Release( byte* buffer )
{
    ASSERT( buffer );
    ASSERT( buffer >= _heap && buffer < _heap + _heapSize );

    bool queued = _pendingReleases.Enqueue( buffer );
    ASSERT( queued );

    // _freeHeapSize.load( std::memory_order_acquire );
    _releaseSignal.Signal();
    return queued;
}

//-----------------------------------------------------------
bool WorkHeap::CanAllocate( size_t size, size_t alignment  ) const
{
    size = alignment * CDivT( size, alignment );

    for( size_t i = 0; i < _heapTable.Length(); i++ )
    {
        HeapEntry& entry = _heapTable[i];

        if( entry.CanAllocate( size, alignment ) )
            return true;
    }

    return false;
}

//-----------------------------------------------------------
void WorkHeap::CompletePendingReleases()
{
    const int BUFFER_SIZE = 128;
    byte* releases[BUFFER_SIZE];

    // Grab pending releases
    int count;
    while( ( count = _pendingReleases.Dequeue( releases, BUFFER_SIZE ) ) )
    {
        // Log::Debug( "Releasing %d buffers.", count );

        for( int i = 0; i < count; i++ )
        {
            byte* buffer = releases[i];
            Log::Debug( "-? Need Free %p", buffer );

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
                    
                    Log::Debug( "- Free %p : %llu", buffer, allocation.size );

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
                        ASSERT( entry.size );

                        // Case 3? See if we also need to merge with the right entry (fills a hole between 2 entries).
                        if( insertIndex < _heapTable.Length() && allocation.EndAddress() == _heapTable[insertIndex].address )
                        {
                            // Extend size to the right entry
                            entry.size += _heapTable[insertIndex].size;
                            _heapTable.Remove( insertIndex );
                            ASSERT( entry.size );
                        }
                    }
                    // Case 2? Can we merge with the right entry
                    else if( insertIndex < _heapTable.Length() && allocation.EndAddress() == _heapTable[insertIndex].address )
                    {
                        // Don't create a new allocation, merge with the right entry
                        _heapTable[insertIndex].address = allocation.address;
                        _heapTable[insertIndex].size   += allocation.size;
                        ASSERT( _heapTable[insertIndex].size );
                    }
                    // Case 4: Insert a new entry, no merges
                    else
                    {
                        // We need to insert a new entry
                        _heapTable.Insert( allocation, insertIndex );
                        ASSERT( _heapTable[insertIndex].size );
                    }

                    break;
                }
            }

            #if _DEBUG
                FatalIf( !foundAllocation, "Failed to find released buffer." );
            #endif
                
        }
    }
}


