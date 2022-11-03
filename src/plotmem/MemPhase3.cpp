#include "MemPhase3.h"
#include "util/Util.h"
#include "util/Log.h"
#include "algorithm/RadixSort.h"
#include "LPGen.h"
#include "ParkWriter.h"
#include <cmath>

#include "DbgHelper.h"
#include "SysHost.h"


//-----------------------------------------------------------
MemPhase3::MemPhase3( MemPlotContext& context )
    : _context( context )
{}

//-----------------------------------------------------------
void MemPhase3::Run()
{
    MemPlotContext& cx = _context;

    // These will become the park buffer once processed.
    Pair* rTables[7] = {
        nullptr,
        cx.t2LRBuffer,
        cx.t3LRBuffer,
        cx.t4LRBuffer,
        cx.t5LRBuffer,
        cx.t6LRBuffer,
        cx.t7LRBuffer,
    };

    // This table will always be used as the left table.
    // It will be re-written as a lookup table after each pass.
    uint32* lTable   = cx.t1XBuffer;

    // Use the same buffer for LPs, it will be serialized to 
    // park back in the rTable.
    // Therefore after each iteration rTable will be a park buffer
    uint64* lpBuffer = cx.metaBuffer0;

    for( uint i = (uint)TableId::Table1; i < (uint)TableId::Table7; i++ )
    {
        Pair*        rTable       = rTables[i+1];
        const uint64 rTableCount  = cx.entryCount[i+1];
        const byte*  rUsedEntries = i < (uint)TableId::Table6 ? (byte*)cx.usedEntries[i+1] : nullptr;

        Log::Line( "  Compressing tables %u and %u...", i+1, i+2 );
        auto tableTimer = TimerBegin();
        
        uint64 newCount;
        if( i == (uint)TableId::Table6 )
            newCount = ProcessTable<true> ( lTable, lpBuffer, rTable, rTableCount, rUsedEntries, (TableId)i );
        else
            newCount = ProcessTable<false>( lTable, lpBuffer, rTable, rTableCount, rUsedEntries, (TableId)i ); 

        double tElapsed = TimerEnd( tableTimer );
        Log::Line( "  Finished compressing tables %u and %u in %.2lf seconds", i+1, i+2, tElapsed );
        Log::Line( "  Table %d now has %llu / %llu entries ( %.2lf%% ).", 
            i+1, newCount, rTableCount, (newCount / (double)rTableCount) * 100 );
    }
}

//-----------------------------------------------------------
template<bool IsTable6>
uint64 MemPhase3::ProcessTable( uint32* lEntries, uint64* lpBuffer, Pair* rTable,
                                const uint64 rTableCount, const byte* markedEntries, TableId tableId )
{
    auto& cx = _context;

    const uint   threadCount      = cx.threadCount;
    const uint64 entriesPerThread = rTableCount / threadCount;
    const uint64 trailingEntries  = rTableCount - ( entriesPerThread * threadCount );

    if constexpr ( IsTable6 )
    {
        // ConverToLinePointThread reads fron lpBuffer,
        // but since we haven't pruned rTable and moved it to lpBuffer,
        // we need to swap it here, so that we read/write from/to it in ConverToLinePointThread
        uint64* tmp = (uint64*)rTable;
        rTable   = (Pair*)lpBuffer;
        lpBuffer = tmp;
    }

    uint32* map = (uint32*)cx.metaBuffer1;

    std::atomic<uint> threadSignal = 0;
    std::atomic<uint> releaseLock  = 0;
    
    LPJob jobs[MAX_THREADS];

    for( uint i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];
        job.Init( (uint)i, threadCount, threadSignal, releaseLock );

        job.lTable        = lEntries;
        job.length        = entriesPerThread;
        job.offset        = i * entriesPerThread;
        job.rTable        = (Pair*)rTable;
        job.lpBuffer      = lpBuffer;
        job.jobs          = jobs;

        job.markedEntries = markedEntries;
        job.map           = map;
    }

    jobs[threadCount-1].length += trailingEntries;

    constexpr bool PruneTable = !IsTable6;
    cx.threadPool->RunJob( ProcessTableThread<PruneTable>, jobs, threadCount );


    // Get the new total length after the prune
    uint64 newLength;
    if constexpr ( !IsTable6 )
    {
        newLength = jobs[0].length;
        
        for( uint i = 1; i < threadCount; i++ )
            newLength += jobs[i].length;
    }
    else
    {
        // No prunning for table 6, so same length
        newLength = rTableCount;
    }


    // Sort LinePoints, along with the map
    RadixSort256::SortWithKey<MAX_THREADS>( *cx.threadPool,
        lpBuffer, (uint64*)rTable,
        map,      map + newLength,  // This is meta1, so there's plenty of space to hold both buffers
        newLength );
    

    // Write lookup table (map it based on sort key)
    // After this step lEntries will contain the new index map into the LP's
    cx.threadPool->RunJob( WriteLookupTableThread, jobs, threadCount );


    if constexpr ( IsTable6 )
    {
        uint32* t7SortTmp       = (uint32*)cx.yBuffer0; // Don't need yBuffer0 at this point, safe to use
        uint32* lEntriesSortTmp = (uint32*)cx.yBuffer1;

        // We need to sort on f7 now, with lEntries with
        // contain now the index into table 6's LinePoints
        RadixSort256::SortWithKey<MAX_THREADS>( *cx.threadPool,
            cx.t7YBuffer, t7SortTmp,
            lEntries,     lEntriesSortTmp,
            newLength );

        cx.entryCount[(uint)TableId::Table7] = newLength;
    }

    #if DBG_WRITE_LINE_POINTS
    {
        char filePath[512];
        snprintf( filePath, sizeof( filePath ), "%slp.t%d.tmp", DBG_TABLES_PATH, (int)tableId+1 );
        DbgWriteTableToFile( *cx.threadPool, filePath, newLength, lpBuffer, true );
    }
    #endif

    // Write park for table (re-use rTable for it)
    // #NOTE: For table 6: rTable is meta0 here.
    byte*  parkBuffer     = _context.plotWriter->AlignPointerToBlockSize<byte>( (void*)rTable );
    size_t sizeTableParks = WriteParks<MAX_THREADS>( *cx.threadPool, newLength, lpBuffer, parkBuffer, tableId );
    
    // Send over the park for writing in the plot file in the background
    if( !cx.plotWriter->WriteTable( parkBuffer, sizeTableParks ) )
        Fatal( "Failed to write table %d to disk.", (int)tableId+1 );

    if constexpr ( IsTable6 )
    {
        #if DBG_WRITE_SORTED_F7_TABLE
        {
            DbgWriteTableToFile( *cx.threadPool, DBG_TABLES_PATH "f7.tmp", newLength, cx.t7YBuffer, true );
            DbgWriteTableToFile( *cx.threadPool, DBG_TABLES_PATH "t7indices.tmp", newLength, lEntries, true );
        }
        #endif
    }

    return newLength;
}

//-----------------------------------------------------------
template<bool PruneTable>
void ProcessTableThread( LPJob* job )
{
    // - Scan the table and determine how many valid entries we have
    // - Now we know where to place them in the new table
    // - Prune the table by copying valid entries to the new buffer
    //  - Write the original index of the entry into a map
    // - Convert pruned entries to LinePoint using the 'left' table.
    //      In table 1, these are the x's. In the others, it is the lookup table.
    // - Sort on line point, sorting along with it the map
    // - Using the map, write a lookup table by writing
    //      on the lookup table's original index (obtained from the map)
    //      the final index (current index of the map) of the entry.
    // - Use the lookup table as the 'left' table for the next table.
    //      Since the lookup table maps to the final indices, the 
    //      LinePoints can be generated from it.

    if constexpr ( PruneTable )
    {
        PruneAndMapThread( job );
        job->WaitForThreads();
    }

    // Convert to LinePoint
    ConverToLinePointThread( job );


    // If it's the last table pair, perform a few things differently.
    if constexpr ( !PruneTable )
    {
        job->WaitForThreads();
        
        // Generate a simple sequential sort key for sorting
        // on y, then sorting on line point. This will
        // finally be used to map the final index table
        // that maps from f7 to table 6's line points.

        uint32* map = job->map;
        
        uint64 i = job->offset;
        const uint64 end = i + job->length;

        for( ; i < end; i++ )
            map[i] = (uint32)i;
    }
}

//-----------------------------------------------------------
void PruneAndMapThread( LPJob* job )
{
    const byte*  markedEntries = job->markedEntries;

    uint64       length        = job->length;

    const uint64 srcOffset     = job->offset; 
    const uint64 end           = srcOffset + length;

    Pair* pairs = job->rTable;

    // Scan entries
    {
        uint64 newLength = 0;

        for( uint64 i = srcOffset; i < end; i++ )
        {
            if( markedEntries[i] )
                newLength ++;
        }

        length      = newLength;
        job->length = newLength;
    }

    // Wait for other entries so that can determine
    // the new position to copy to.
    job->WaitForThreads();

    ///
    /// Prune to new buffer
    ///

    // Get our new offset
    uint64 dstOffset = 0;

    const uint  threadId = job->_threadId;
    const auto* jobs     = job->jobs;

    for( uint i = 0; i < threadId; i++ )
        dstOffset += jobs[i].length;

    // Store new offset
    job->offset = dstOffset;

    // Copy our valid entries to the new buffer
    uint32* map      = job->map + dstOffset;
    Pair*   newPairs = (Pair*)(job->lpBuffer + dstOffset);

    uint64 dstI = 0;

    for( uint64 i = srcOffset; i < end; i++ )
    {
        if( !markedEntries[i] )
            continue;
        
        newPairs[dstI] = pairs[i];  // Copy to new location
        map     [dstI] = (uint32)i; // Map the entry back to its original location

        dstI++; 
    }

    ASSERT( dstI == length );
}

//-----------------------------------------------------------
void ConverToLinePointThread( LPJob* job )
{
    const uint64  length = job->length;

    Pair*         rTable = (Pair*)(job->lpBuffer + job->offset);
    const uint32* lTable = job->lTable;

    for( uint64 i = 0; i < length; i++ )
    {
        const Pair* rEntry = &rTable[i];
        
        const uint64 x = lTable[rEntry->left ];
        const uint64 y = lTable[rEntry->right];
        ASSERT( x || y );

        const uint64 lp = SquareToLinePoint( x, y );
        ASSERT( lp );
        *((uint64*)rEntry) = lp;//SquareToLinePoint( x, y );
    }
}

//-----------------------------------------------------------
void WriteLookupTableThread( LPJob* job )
{
    const uint64  length = job->length;
    const uint64  offset = job->offset;
    const uint64  end    = offset + length;

    const uint32* map    = job->map;
    uint32*       lookup = job->lTable;

    for( uint64 i = offset; i < end; i++ )
        lookup[map[i]] = (uint32)i;
}


