#include "MemPhase2.h"
#include "DbgHelper.h"

///
/// Job structs
///

struct ClearMarkingBufferJob
{
    byte*  buffer;
    size_t size;
};

struct MarkJob
{
    uint64      startIndex;
    uint64      rightEntryCount;
    const Pair* rightEntries;
    const byte* rightMarkedEntries;  // Used in tables <= 5
    byte*       leftMarkingBuffer;

    uint64      fieldPerMarkingBuffer;
};

///
/// Internal Functions
///
void ClearMarkedEntriesThread( ClearMarkingBufferJob* job );

template<bool HasRightTableMarkingBuffer>
void MarkEntriesThread( MarkJob* job );


void DbgReadPhase1TableFiles( MemPlotContext& cx );
void DbgCountMarkedEntries( MemPlotContext& cx );

///
/// Implementation
///

//-----------------------------------------------------------
MemPhase2::MemPhase2( MemPlotContext& context )
    : _context( context )
{
    
}

//-----------------------------------------------------------
void MemPhase2::Run()
{
    MemPlotContext& cx = _context;

    #if DBG_READ_PHASE_1_TABLES
        // #TODO: Move this to Phase1
        if( cx.plotCount == 0 )
            DbgReadPhase1TableFiles( cx );
    #endif
    
    #if DBG_READ_MARKED_TABLES
        if( cx.plotCount == 0 )
        {
            DbgReadPhase2MarkedEntries( cx );
            return;
        }
    #endif

    // Prep our marking buffers
    ClearMarkingBuffers();


    // Now mark the rest of the tables
    const Pair* rTables[7] = {
        nullptr,
        cx.t2LRBuffer,
        cx.t3LRBuffer,
        cx.t4LRBuffer,
        cx.t5LRBuffer,
        cx.t6LRBuffer,
        cx.t7LRBuffer
    };

    // #NOTE: We don't need to prune table 1. 
    //        Since it doesn't refer back to other values,
    //        pruning up to table 2 is enough.
    for( uint i = (int)TableId::Table7; i > 1; i-- )
    {
        const Pair*  rTable       = rTables[i];
        const uint64 rTableCount  = cx.entryCount[i];
        byte* lTableMarkingBuffer = (byte*)cx.usedEntries[i-1];


        Log::Line( "  Prunning table %d...", i );
        auto timer = TimerBegin();

        if( i == (int)TableId::Table7 )
        {
            // Table 6 which does not have a rightMarkedEntries buffer, as all of table 7's entries are valid
            MarkTable<false>( rTable, rTableCount, nullptr, lTableMarkingBuffer );
        }
        else
        {
            const byte* rTableMarkedEntries = (byte*)cx.usedEntries[i];

            MarkTable<true>( rTable, rTableCount, rTableMarkedEntries, lTableMarkingBuffer );
        }

        double elapsed = TimerEnd( timer );
        Log::Line( "  Finished prunning table %d in %.2lf seconds.", i, elapsed );
    }

    // DbgCountMarkedEntries( cx );
    DbgWritePhase2MarkedEntries( cx );
}

//-----------------------------------------------------------
void MemPhase2::ClearMarkingBuffers()
{
    MemPlotContext& cx = _context;

    const uint64 maxEntries    = 1ull << _K;
    byte*        markingBuffer = (byte*)cx.yBuffer0;

    const size_t totalSize     = maxEntries * 5;  // We need 5 buffers, for tables 2-6 
    const uint   threadCount   = cx.threadCount;

    const size_t sizePerThread = totalSize / threadCount;

    ClearMarkingBufferJob jobs[MAX_THREADS];

    for ( uint64 i = 0; i < threadCount; i++ )
    {
        auto& job  = jobs[i];
        job.buffer = markingBuffer + i * sizePerThread;
        job.size   = sizePerThread;
    }

    // Add trailing size
    jobs[threadCount-1].size += totalSize - (sizePerThread * threadCount);
    
    cx.threadPool->RunJob( ClearMarkedEntriesThread, jobs, threadCount );

    // Assign our table buffers
    cx.usedEntries[0] = nullptr;    // Table 1 has no need for marked entries

    for( uint i = 0; i < 5; i++ )
        cx.usedEntries[i+1] = markingBuffer + i * maxEntries;
}

//-----------------------------------------------------------
template<bool HasRightTableMarkingBuffer>
void MemPhase2::MarkTable( const Pair* rightTable, uint64 rightEntryCount, const byte* rMarkedEntries, byte* lMarkingBuffer )
{
    MemPlotContext& cx = _context;

    const uint   threadCount           = cx.threadCount;
    const uint64 rightEntriesPerThread = rightEntryCount / threadCount;

    MarkJob jobs[MAX_THREADS];

    for( uint i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];

        job.startIndex         = i * rightEntriesPerThread;
        job.rightEntryCount    = rightEntriesPerThread;
        job.rightEntries       = rightTable;
        job.rightMarkedEntries = rMarkedEntries;
        job.leftMarkingBuffer  = lMarkingBuffer;
    }

    // Add trailing entries to the last job
    jobs[threadCount-1].rightEntryCount += (rightEntryCount - ( rightEntriesPerThread  * threadCount ) );

    cx.threadPool->RunJob( MarkEntriesThread<HasRightTableMarkingBuffer>, jobs, threadCount );
}

//-----------------------------------------------------------
void ClearMarkedEntriesThread( ClearMarkingBufferJob* job )
{
    memset( job->buffer, 0, job->size );
}

//-----------------------------------------------------------
template<bool HasRightTableMarkingBuffer>
void MarkEntriesThread( MarkJob* job )
{
    const uint64 startIndex  = job->startIndex;
    const uint64 endIndex    = startIndex + job->rightEntryCount;

    const Pair* rightEntries = job->rightEntries;

    const byte* rightMarkedEntries = job->rightMarkedEntries;
    byte* markingBuffer            = job->leftMarkingBuffer;
    
    // #NOTE: This can easily cause false sharing.
    //        but since the region of data is so big, the thread acount
    //        no expected to be so great (maybe max 256?)
    //        and the data write locations are random, I don't
    //        think it will cause that many misses.
    //        In practice this method has been faster than others
    //        which take false-sharing into account.
    //        We also only write to the location, we don't care
    //        about reading it, so we won't have any invalid data.
    //       This means, thought, that this might not scale linearly, though.

    for( uint64 i = startIndex; i < endIndex; i++ )
    {
        if constexpr ( HasRightTableMarkingBuffer )
        {
            // If this entry is not marked as used 
            // in the right marked buffer, then skip it.
            // It did not contribute to the final f7 value,
            // so we don't need to consider it.
            if( !rightMarkedEntries[i] )
                continue;
        }

        const Pair& entry = rightEntries[i];

        markingBuffer[entry.left ] = 1;
        markingBuffer[entry.right] = 1;
    }
}


///
/// Debug
///

//-----------------------------------------------------------
void DbgCountMarkedEntries( MemPlotContext& cx )
{
    for( uint i = (uint)TableId::Table6; i > (uint)TableId::Table1; i-- )
    {
        uint64 originalCount = cx.entryCount[i];
        uint64 markedCount   = 0;
        
        const byte* markedEntries = cx.usedEntries[i];

        for( uint64 e = 0; e < originalCount; e++ )
        {
            if( markedEntries[e] )
                markedCount++;
        }
        
        const uint64 nDropped = originalCount - markedCount;
        Log::Line( "Table %d has now: %llu / %llu : %.2lf%% = %llu dropped.",
            i+1, markedCount, originalCount, 
            (markedCount/(f64)originalCount)*100,
            nDropped
        );
    }
}

//-----------------------------------------------------------
void DbgReadPhase1TableFiles( MemPlotContext& cx )
{
    #if DBG_READ_PHASE_1_TABLES && !DBG_WRITE_PHASE_1_TABLES
        DbgReadTableFromFile( *cx.threadPool, DBG_P1_TABLE1_FNAME  , cx.entryCount[0], cx.t1XBuffer , true );
        DbgReadTableFromFile( *cx.threadPool, DBG_P1_TABLE2_FNAME  , cx.entryCount[1], cx.t2LRBuffer, true );
        DbgReadTableFromFile( *cx.threadPool, DBG_P1_TABLE3_FNAME  , cx.entryCount[2], cx.t3LRBuffer, true );
        DbgReadTableFromFile( *cx.threadPool, DBG_P1_TABLE4_FNAME  , cx.entryCount[3], cx.t4LRBuffer, true );
        DbgReadTableFromFile( *cx.threadPool, DBG_P1_TABLE5_FNAME  , cx.entryCount[4], cx.t5LRBuffer, true );
        DbgReadTableFromFile( *cx.threadPool, DBG_P1_TABLE6_FNAME  , cx.entryCount[5], cx.t6LRBuffer, true );
        DbgReadTableFromFile( *cx.threadPool, DBG_P1_TABLE7_FNAME  , cx.entryCount[6], cx.t7LRBuffer, true );
        DbgReadTableFromFile( *cx.threadPool, DBG_P1_TABLE7_Y_FNAME, cx.entryCount[6], cx.t7YBuffer , true );
    #endif
}

//-----------------------------------------------------------
void DbgReadWritePhase2MarkedEntries( MemPlotContext& cx, bool write )
{
    const uint64 maxEntries    = 1ull << _K;
    byte*        markingBuffer = (byte*)cx.yBuffer0;
    const size_t sizePerTable  = maxEntries;

    const char* fileNames[6] = {
        nullptr,
        DBG_P2_TABLE2_FNAME,
        DBG_P2_TABLE3_FNAME,
        DBG_P2_TABLE4_FNAME,
        DBG_P2_TABLE5_FNAME,
        DBG_P2_TABLE6_FNAME,
    };

    cx.usedEntries[0] = nullptr;

    for( uint i = 1; i < 6; i++ )
    {
        cx.usedEntries[i] = markingBuffer + (i-1) * sizePerTable;

        if( write )
        {
            DbgWriteTableToFile( *cx.threadPool, fileNames[i], maxEntries, cx.usedEntries[i], true );
        }
        else
        {
            uint64 entryCount = 0;
          
            DbgReadTableFromFile( *cx.threadPool, fileNames[i], entryCount, cx.usedEntries[i], true );
            if( entryCount != maxEntries )
            {
                Log::Line( "Error: Invalid file. Wrong entry count." );
                exit( 1 );
            }
        }
    }
}

//-----------------------------------------------------------
void DbgWritePhase2MarkedEntries( MemPlotContext& cx )
{
    #if DBG_WRITE_MARKED_TABLES
    DbgReadWritePhase2MarkedEntries( cx, true );
    #endif
}

//-----------------------------------------------------------
void DbgReadPhase2MarkedEntries( MemPlotContext& cx )
{
    #if DBG_READ_MARKED_TABLES
    DbgReadWritePhase2MarkedEntries( cx, false );
    #endif
}