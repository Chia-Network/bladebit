#include "MemPhase1.h"
#include "b3/blake3.h"
#include "pos/chacha8.h"
#include "util/Util.h"
#include "util/Log.h"
#include "FxSort.h"
#include "algorithm/YSort.h"
#include "SysHost.h"
#include <cmath>

#include "DbgHelper.h"
    
    bool DbgVerifySortedY( const uint64 entryCount, const uint64* yBuffer );
    
#if _DEBUG
    // #define DBG_VALIDATE_KB_GROUPS 1

    #define DBG_FILE_T1_Y_PATH      DBG_TABLES_PATH "y.t1.tmp"
    #define DBG_FILE_T1_X_PATH      DBG_TABLES_PATH "x.t1.tmp"

#endif
    bool DbgTestPairs( uint64 entryCount, const Pair* pairs );
    bool DbgVerifyPairsKBCGroups( uint64 entryCount, const uint64* yBuffer, const Pair* pairs );


///
/// Internal data
///
struct F1GenJob
{
    // uint node;
    // uint threadCount;
    // uint cpuId;
    // uint startPage;
    // uint pageCount;
    
    const byte* key;

    uint32  blockCount;
    uint32  entryCount;
    uint32  x;
    byte*   blocks;
    uint64* yBuffer;
    uint32* xBuffer;
};

struct kBCJob
{
    const uint64* yBuffer;
    uint64        maxCount;         // Max group count for scan job, pair count for pair job.
    uint64        groupCount;
    uint32*       groupBoundaries;
    uint64        startIndex;

    // For scan job
    uint64 endIndex;

    // For scan job
    uint64 pairCount;        // Group count for scan job, pair count for pair job.
    Pair*  pairs;
    Pair*  copyDst;          // For second pass

#if DEBUG
    uint32 jobIdx;
#endif
};


template<typename TYOut, typename TMetaIn, typename TMetaOut>
struct FpFxJob
{
    uint64         entryCount;
    const TMetaIn* inMetaBuffer;
    const uint64*  inYBuffer;
    const Pair*    lrPairs;
    TMetaOut*      outMetaBuffer;
    TYOut*         outYBuffer;
};

/// Internal Funcs forwards-declares
void F1JobThread( F1GenJob* job );
void F1NumaJobThread( F1GenJob* job );

void FpScanThread( kBCJob* job );
void FpPairThread( kBCJob* job );

template<typename TYOut, typename TMetaIn, typename TMetaOut>
void ComputeFxJob( FpFxJob<TYOut, TMetaIn, TMetaOut>* job );

template<size_t metaKMultiplierIn, size_t metaKMultiplierOut, uint ShiftBits>
FORCE_INLINE uint64 ComputeFx( uint64 y, uint64* metaData, uint64* metaOut );



//----------------------------------------------------------
MemPhase1::MemPhase1( MemPlotContext& context )
    : _context( context )
{
    LoadLTargets();
}

//-----------------------------------------------------------
void MemPhase1::WaitForPreviousPlotWriter()
{
    // Wait until the current plot has finished writing
    if( !_context.plotWriter->WaitUntilFinishedWriting() )
        Fatal( "Failed to write previous plot file %s with error: %d", 
            _context.plotWriter->FilePath().c_str(),
            _context.plotWriter->GetError() );

    const char* curname = _context.plotWriter->FilePath().c_str();
    char* newname = new char[strlen(curname) - 3]();
    memcpy(newname, curname, strlen(curname) - 4);

    rename(curname, newname);

    // Print final pointer offsets
    Log::Line( "" );
    Log::Line( "Previous plot %s finished writing to disk:", _context.plotWriter->FilePath().c_str() );
    const uint64* tablePointers = _context.plotWriter->GetTablePointers();
    for( uint i = 0; i < 7; i++ )
    {
        const uint64 ptr = Swap64( tablePointers[i] );
        Log::Line( "  Table %u pointer  : %16lu ( 0x%016lx )", i+1, ptr, ptr );
    }

    for( uint i = 7; i < 10; i++ )
    {
        const uint64 ptr = Swap64( tablePointers[i] );
        Log::Line( "  C%u table pointer : %16lu ( 0x%016lx )", i+1-7, ptr, ptr);
    }
    Log::Line( "" );

    _context.p4WriteBuffer = nullptr;
}

//----------------------------------------------------------
void MemPhase1::Run()
{
    const uint64 entryCount = GenerateF1();

    ForwardPropagate( entryCount );


    #if DBG_WRITE_PHASE_1_TABLES
        WritePhaseTableFiles( _context );
    #endif

    // Test Proofs
    #if DBG_DUMP_PROOFS
    {
        const uint64 proofRange[2] = DBG_TEST_PROOF_RANGE;

        const uint64 f7Count = proofRange[1];//(proofRange[1]+1) - proofRange[0];
        Log::Line( "\nDumping Test Proofs from %llu - %llu", proofRange[0], proofRange[0]+f7Count );

        for( uint i = 0; i < f7Count; i++ )
        {
            const uint64 f7Idx = proofRange[0] + i;
            DumpTestProofs( _context, f7Idx );
            Log::Line( "" );
        }
    }
    #endif
}


//-----------------------------------------------------------
uint64 MemPhase1::GenerateF1()
{
    MemPlotContext& cx  = _context;

    ///
    /// Init chacha key
    ///
    // First byte is the table index
    byte key[32] = { 1 };
    memcpy( key + 1, cx.plotId, 31 );
    
    ///
    /// Prepare jobs
    ///
    const uint   k                  = _K;
    const size_t CHACHA_BLOCK_SIZE  = kF1BlockSizeBits / 8;
    const uint   numThreads         = cx.threadCount;

    const uint64 totalEntries       = 1ull << k;
    const uint64 entriesPerBlock    = CHACHA_BLOCK_SIZE / sizeof( uint32 );
    const uint64 totalBlocks        = totalEntries / entriesPerBlock;
    const uint64 blocksPerThread    = totalBlocks / numThreads;
    const uint64 entriesPerThread   = blocksPerThread * entriesPerBlock;

    const uint64 trailingEntries    = totalEntries - ( entriesPerThread * numThreads );
    const uint64 trailingBlocks     = CDiv( trailingEntries, entriesPerBlock );

    ASSERT( entriesPerBlock * sizeof( uint32 ) == CHACHA_BLOCK_SIZE );  // Must fit exactly within a block

    // Generate all of the y values to a metabuffer first
    byte*   blocks  = (byte*)cx.yBuffer0;
    uint64* yBuffer = cx.yBuffer0;
    uint32* xBuffer = cx.t1XBuffer;
    uint64* yTmp    = cx.metaBuffer1;
    uint32* xTmp    = (uint32*)(yTmp + totalEntries);

    ASSERT( numThreads <= MAX_THREADS );

    // const NumaInfo* numa = SysHost::GetNUMAInfo();

    // Gen all raw f1 values
    {
        // Prepare jobs
        F1GenJob jobs[MAX_THREADS];
        for( uint i = 0; i < numThreads; i++ )
        {
            uint64 offset      = i * entriesPerThread;
            uint64 blockOffset = i * blocksPerThread * CHACHA_BLOCK_SIZE;

            F1GenJob& job = jobs[i];
            // job.cpuId       = i;
            // job.threadCount = numThreads;

            job.key        = key;
            job.blockCount = (uint32)blocksPerThread;
            job.entryCount = (uint32)entriesPerThread;
            job.x          = (uint32)offset;
            job.blocks     = blocks  + blockOffset;
            job.yBuffer    = yTmp    + offset;
            job.xBuffer    = xTmp    + offset;
        }

        jobs[numThreads-1].entryCount += (uint32)trailingEntries;
        jobs[numThreads-1].blockCount += (uint32)trailingBlocks;

        // Initialize NUMA pages
        // if( numa )
        // {
        //     const uint pageSize       = (uint)SysHost::GetPageSize();
        //     const uint blocksPerPage  = (uint)( pageSize / CHACHA_BLOCK_SIZE );
        //     const uint pageCount      = (uint)( totalBlocks / blocksPerPage );
        //     const uint pagesPerThread = pageCount / numThreads;
        //     const uint nodeStride     = numa->nodeCount;

        //     for( uint i = 0; i < numa->nodeCount; i++ )
        //     {
        //         const auto& nodeCpus = numa->cpuIds[i];
        //         const uint  cpuCount = nodeCpus.length;

        //         // #TODO: Remove this. For now we hard-code it to
        //         //        the node of the first page.
        //         // const int pageOffset = (i + 1) & 1;

        //         for( uint j = 0; j < cpuCount; j++ )
        //         { 
        //             const uint cpuId = nodeCpus[j];

        //             auto& job = jobs[cpuId];

        //             job.node      = i;
        //             job.startPage = nodeStride * j;
        //             job.pageCount = pagesPerThread;
        //             job.blocks    = blocks;
        //             job.yBuffer   = yBuffer;
        //             job.xBuffer   = xBuffer;
        //         }
        //     }
        // }

        Log::Line( "Generating F1..." );
        auto timeStart = TimerBegin();

        // cx.threadPool->RunJob( numa ? F1NumaJobThread : F1JobThread, jobs, numThreads );
        cx.threadPool->RunJob( F1JobThread, jobs, numThreads );

        double elapsed = TimerEnd( timeStart );
        Log::Line( "Finished F1 generation in %.2lf seconds.", elapsed );
    }

    Log::Line( "Sorting F1..." );
    auto timeStart = TimerBegin();

    YSorter sorter( *cx.threadPool );
    sorter.Sort( totalEntries, yTmp, yBuffer, xTmp, xBuffer );

    double elapsed = TimerEnd( timeStart );
    Log::Line( "Finished F1 sort in %.2lf seconds.", elapsed );


    #if DBG_VERIFY_SORT_F1
        Log::Line( "Verifying that y is sorted..." );
        if( !DbgVerifySortedY( totalEntries, (uint64*)yBuffer ) )
        {
            Log::Line( "Failed." );
            exit( 1 );
        }
        Log::Line( "Ok!" );
    #endif

    #if DBG_WRITE_T1
        DbgWriteTableToFile( *cx.threadPool, DBG_FILE_T1_Y_PATH, totalEntries, (uint64*)yBuffer );
        DbgWriteTableToFile( *cx.threadPool, DBG_FILE_T1_X_PATH, totalEntries, xBuffer );
    #endif

    return totalEntries;
}


///
/// Perform forward propagation across all tables 
/// after F1 has been generated and sorted.
///
//-----------------------------------------------------------
void MemPhase1::ForwardPropagate( uint64 table1EntryCount )
{
    MemPlotContext& cx  = _context;

    // F1's y values are stored in yBuffer0, this will server
    // as the first y read buffer. They may be flipped afterwards.
    ReadWriteBuffer<uint64> yBuffer   ( cx.yBuffer0,    cx.yBuffer1    );
    ReadWriteBuffer<uint64> metaBuffer( cx.metaBuffer0, cx.metaBuffer1 );

    uint64 table2EntryCount = FpComputeTable<TableId::Table2>( table1EntryCount, yBuffer, metaBuffer );
    uint64 table3EntryCount = FpComputeTable<TableId::Table3>( table2EntryCount, yBuffer, metaBuffer );
    uint64 table4EntryCount = FpComputeTable<TableId::Table4>( table3EntryCount, yBuffer, metaBuffer );
    uint64 table5EntryCount = FpComputeTable<TableId::Table5>( table4EntryCount, yBuffer, metaBuffer );
    uint64 table6EntryCount = FpComputeTable<TableId::Table6>( table5EntryCount, yBuffer, metaBuffer );
    uint64 table7EntryCount = FpComputeTable<TableId::Table7>( table6EntryCount, yBuffer, metaBuffer );

    cx.entryCount[0] = table1EntryCount;
    cx.entryCount[1] = table2EntryCount;
    cx.entryCount[2] = table3EntryCount;
    cx.entryCount[3] = table4EntryCount;
    cx.entryCount[4] = table5EntryCount;
    cx.entryCount[5] = table6EntryCount;
    cx.entryCount[6] = table7EntryCount;
}

//-----------------------------------------------------------
template<TableId tableId>
uint64 MemPhase1::FpComputeTable( uint64 entryCount,
                                  ReadWriteBuffer<uint64>& yBuffer, 
                                  ReadWriteBuffer<uint64>& metaBuffer )
{
    static_assert( tableId >= TableId::Table2 && tableId <= TableId::Table7 );

    MemPlotContext& cx  = _context;

    Pair* pairBuffer;
    if      constexpr ( tableId == TableId::Table2 ) pairBuffer = cx.t2LRBuffer;
    else if constexpr ( tableId == TableId::Table3 ) pairBuffer = cx.t3LRBuffer;
    else if constexpr ( tableId == TableId::Table4 ) pairBuffer = cx.t4LRBuffer;
    else if constexpr ( tableId == TableId::Table5 ) pairBuffer = cx.t5LRBuffer;
    else if constexpr ( tableId == TableId::Table6 ) pairBuffer = cx.t6LRBuffer;
    else if constexpr ( tableId == TableId::Table7 ) pairBuffer = cx.t7LRBuffer;

    return FpComputeSingleTable<tableId>( entryCount, pairBuffer, yBuffer, metaBuffer );
}

//-----------------------------------------------------------
template<TableId tableId>
uint64 MemPhase1::FpComputeSingleTable(
    uint64 entryCount,
    Pair*  pairBuffer,
    ReadWriteBuffer<uint64>& yBuffer, 
    ReadWriteBuffer<uint64>& metaBuffer )
{
    using TMetaIn  = typename TableMetaType<tableId>::MetaIn;
    using TMetaOut = typename TableMetaType<tableId>::MetaOut;

    MemPlotContext& cx  = _context;
    Log::Line( "Forward propagating to table %d...", (int)tableId+1 );

    // yBuffer.read amd metaBuffer.read should always point
    // to the y and meta values generated from the previous table, respectively
    // That is the values generated from the previous' tables fx(), but sorted.
    
    Pair* unsortedPairBuffer = cx.t7LRBuffer; // On tables below 7, we use table 7's L/R
                                              // buffer as our temporary pair buffer.
                                              // We can't use a metadata one as we need to keep
                                              // the temp pairs around for sorting (which require the meta buffers).
    
    uint32* groupBoundaries = (uint32*)yBuffer.write;   // Temporarily use this buffer to store group boundaries

    if constexpr ( tableId == TableId::Table7 )
    {
        // Write y buffer to table 7's f7 buffer
        yBuffer.write = (uint64*)cx.t7YBuffer;
    }

    auto tableTimer = TimerBegin();

    uint64 pairCount;
    {
        kBCJob jobs[MAX_THREADS];

        // Scan for kBC groups
        const uint64 groupCount = FpScan( entryCount, yBuffer.read, groupBoundaries, jobs );
        
        // Generate L/R pairs from kBC groups (writes to unsorted pair buffer)
        Pair* tmpPairBuffer = (Pair*)metaBuffer.write;

        pairCount = FpPair( yBuffer.read, jobs, groupCount, tmpPairBuffer, unsortedPairBuffer );
    }

    // Compute fx values for this new table
    const uint64* inMetaBuffer = metaBuffer.read;

    // Special case: Use taable 1's x buffer as input metadata
    if constexpr ( tableId == TableId::Table2 )
    {
        inMetaBuffer = (uint64*)cx.t1XBuffer;
        ASSERT( metaBuffer.read == cx.metaBuffer0 );
    }

    FpComputeFx<tableId, TMetaIn, TMetaOut>( 
        pairCount, unsortedPairBuffer, 
        (TMetaIn*)inMetaBuffer, yBuffer.read,
        (TMetaOut*)metaBuffer.write, yBuffer.write );

    // DbgVerifyPairsKBCGroups( pairCount, yBuffer.read, unsortedPairBuffer );

    // If a previous plot was being written to disk, we need to ensure
    // it finished as we are about to use meta0 which is used to write the
    // last buffers written to disk.
    if( cx.p4WriteBuffer )
    {
        Log::Line( " Waiting for last plot to finish being written to disk..." );
        WaitForPreviousPlotWriter();
    }   

    // Swap buffers here, ready for sort
    yBuffer   .Swap();
    metaBuffer.Swap();

    // Sort our (y, metadata, L/R entries) on y
    if constexpr ( tableId != TableId::Table7 )
    {
        Log::Line( "  Sorting entries..." );
        auto timer = TimerBegin();

        // Use table 7's buffers as a temporary buffer
        uint32* sortKey    = cx.t7YBuffer;
        uint32* sortKeyTmp = (uint32*)( metaBuffer.write + ENTRIES_PER_TABLE ); // Use the output metabuffer for now as 
                                                                                // the temporary sortkey buffer.
        SortFx<MAX_THREADS>(
            *cx.threadPool,        pairCount,
            (uint64*)yBuffer.read, yBuffer.write,
            sortKeyTmp,            sortKey
        );
        yBuffer.Swap();

        // DbgVerifyPairsKBCGroups( pairCount, yBuffer.write, unsortedPairBuffer );

        MapFxWithSortKey<TMetaOut, MAX_THREADS>(
            *cx.threadPool, pairCount, sortKey,
            (TMetaOut*)metaBuffer.read, (TMetaOut*)metaBuffer.write,
            unsortedPairBuffer,         pairBuffer   // Write to the final pair buffer
        );

        // DbgVerifyPairsKBCGroups( pairCount, yBuffer.write, pairBuffer );

        // Use the sorted metabuffer as the read buffer for the next table
        metaBuffer.Swap();

        double elapsed = TimerEnd( timer );
        Log::Line( "  Finished sorting in %.2lf seconds.", elapsed );
        // Log::Line( "  Entries after sort: %llu/%llu (%.2lf%%).", entryCount, pairCount, (entryCount / (f64)pairCount) * 100 );

        #if DBG_VERIFY_SORT_FX
            Log::Line( "  Verifying that fx (y) is sorted..." );
            if( !DbgVerifySortedY( pairCount, (uint64*)yBuffer.read ) )
            {
                Log::Line( "  Failed." );
                exit( 1 );
            }
            Log::Line( "  Ok!" );
        #endif

        #if DBG_WRITE_Y_VALUES
        {
            char filePath[512];
            
            sprintf( filePath, DBG_TABLES_PATH "t%d.y.tmp", (int)tableId );
            DbgWriteTableToFile( *cx.threadPool, filePath, pairCount, yBuffer.read, true );
        }
        #endif
    }
    else
    {
        // We don't sort table 7 just yet, so leave it as is.
    }


    double tableElapsed = TimerEnd( tableTimer );
    Log::Line( "Finished forward propagating table %d in %.2lf seconds.", (int)tableId+1, tableElapsed );

    return pairCount;
}

//-----------------------------------------------------------
void F1JobThread( F1GenJob* job )
{
    const uint32 blockCount = job->blockCount;
    const uint32 entryCount = job->entryCount;
    const uint64 x          = job->x;

    uint32* blocks  = (uint32*)job->blocks;
    uint64* yBuffer = job->yBuffer;

    // Which block are we generating?
    const uint64 blockIdx = x * _K / kF1BlockSizeBits;

    chacha8_ctx chacha;
    ZeroMem( &chacha );

    chacha8_keysetup( &chacha, job->key, 256, NULL );
    chacha8_get_keystream( &chacha, blockIdx, blockCount, (byte*)blocks );

    // chacha output is treated as big endian, therefore swap, as required by chiapos
    for( uint64 i = 0; i < entryCount; i++ )
    {
        const uint64 y = Swap32( blocks[i] );
        yBuffer[i] = ( y << kExtraBits ) | ( (x+i) >> (_K - kExtraBits) );
    }

    // Gen the x that generated the y
    uint32* xBuffer = job->xBuffer;
    for( uint64 i = 0; i < entryCount; i++ )
        xBuffer[i] = (uint32)( x + i );
}


//-----------------------------------------------------------
// void F1NumaJobThread( F1GenJob* job )
// {
//     // const NumaInfo* numa = SysHost::GetNUMAInfo();

//     const uint32 pageSize           = (uint32)SysHost::GetPageSize();

//     // const uint   k                  = _K;
//     const size_t CHACHA_BLOCK_SIZE  = kF1BlockSizeBits / 8;
//     // const uint64 totalEntries       = 1ull << k;
//     const uint32 entriesPerBlock    = (uint32)( CHACHA_BLOCK_SIZE / sizeof( uint32 ) );
//     const uint32 blocksPerPage      = pageSize / CHACHA_BLOCK_SIZE;
//     const uint32 entriesPerPage32   = entriesPerBlock * blocksPerPage;
//     const uint32 entriesPerPage64   = entriesPerPage32 / 2;
    
//     const uint   pageOffset         = job->startPage;
//     const uint   pageCount          = job->pageCount;

//     const uint32 pageStride         = job->threadCount;
//     const uint32 blockStride        = pageSize         * pageStride;
//     const uint32 entryStride32      = entriesPerPage32 * pageStride;
//     const uint32 entryStride64      = entriesPerPage64 * pageStride;

//     // #TODO: Get proper offset depending on node count. Or, figure out if we can always have
//     //        the pages of the buffers simply start at the same location
//     const uint32 blockStartPage    = SysHost::NumaGetNodeFromPage( job->blocks  ) == job->node ? 0 : 1;
//     const uint32 yStartPage        = SysHost::NumaGetNodeFromPage( job->yBuffer ) == job->node ? 0 : 1;
//     const uint32 xStartPage        = SysHost::NumaGetNodeFromPage( job->xBuffer ) == job->node ? 0 : 1;

//     // const uint64 x                  = job->x;
//     const uint64 x                  = (blockStartPage + pageOffset) * entriesPerPage32;
//     // const uint32 xStride            = pageStride * entriesPerPage;

//     byte*   blockBytes = job->blocks + (blockStartPage + pageOffset) * pageSize;
//     uint32* blocks     = (uint32*)blockBytes;
    
//     uint64* yBuffer    = job->yBuffer + ( yStartPage + pageOffset ) * entriesPerPage64;
//     uint32* xBuffer    = job->xBuffer + ( xStartPage + pageOffset ) * entriesPerPage32;

//     chacha8_ctx chacha;
//     ZeroMem( &chacha );

//     chacha8_keysetup( &chacha, job->key, 256, NULL );

//     for( uint64 p = 0; p < pageCount/4; p+=4 )
//     {
//         ASSERT( SysHost::NumaGetNodeFromPage( blockBytes ) == job->node );

//         // blockBytes
//         // Which block are we generating?
//         const uint64 blockIdx = ( x + p * entryStride32 ) * _K / kF1BlockSizeBits;

//         chacha8_get_keystream( &chacha, blockIdx,  blocksPerPage,   blockBytes );
//         chacha8_get_keystream( &chacha, blockIdx + blocksPerPage,   blocksPerPage, blockBytes + blockStride     );
//         chacha8_get_keystream( &chacha, blockIdx + blocksPerPage*2, blocksPerPage, blockBytes + blockStride * 2 );
//         chacha8_get_keystream( &chacha, blockIdx + blocksPerPage*3, blocksPerPage, blockBytes + blockStride * 3 );
//         blockBytes += blockStride * 4;
//     }

//     for( uint64 p = 0; p < pageCount; p++ )
//     {
//         ASSERT( SysHost::NumaGetNodeFromPage( yBuffer ) == job->node );
//         ASSERT( SysHost::NumaGetNodeFromPage( blocks  ) == job->node );

//         const uint64 curX = x + p * entryStride32;

//         for( uint64 i = 0; i < entriesPerPage32; i++ )
//         {
//             // chacha output is treated as big endian, therefore swap, as required by chiapos
//             const uint64 y = Swap32( blocks[i] );
//             yBuffer[i] = ( y << kExtraBits ) | ( (curX+i) >> (_K - kExtraBits) );
//         }
        
//         // for( uint64 i = 0; i < 64; i++ )
//         // {
//         //     yBuffer[0] = ( Swap32( blocks[0] ) << kExtraBits ) | ( (curX+0) >> (_K - kExtraBits) );
//         //     yBuffer[1] = ( Swap32( blocks[1] ) << kExtraBits ) | ( (curX+1) >> (_K - kExtraBits) );
//         //     yBuffer[2] = ( Swap32( blocks[2] ) << kExtraBits ) | ( (curX+2) >> (_K - kExtraBits) );
//         //     yBuffer[3] = ( Swap32( blocks[3] ) << kExtraBits ) | ( (curX+3) >> (_K - kExtraBits) );
//         //     yBuffer[4] = ( Swap32( blocks[4] ) << kExtraBits ) | ( (curX+4) >> (_K - kExtraBits) );
//         //     yBuffer[5] = ( Swap32( blocks[5] ) << kExtraBits ) | ( (curX+5) >> (_K - kExtraBits) );
//         //     yBuffer[6] = ( Swap32( blocks[6] ) << kExtraBits ) | ( (curX+6) >> (_K - kExtraBits) );
//         //     yBuffer[7] = ( Swap32( blocks[7] ) << kExtraBits ) | ( (curX+7) >> (_K - kExtraBits) );

//         //     yBuffer += 8;
//         //     blocks  += 8;
//         // }

//         // #TODO: This is wrong. We need to fill more y's before w go to the next block page.
//         yBuffer += entryStride64;
//         blocks  += entryStride32;
//     }

//     // Gen the x that generated the y
//     for( uint64 p = 0; p < pageCount; p++ )
//     {
//         ASSERT( SysHost::NumaGetNodeFromPage( xBuffer ) == job->node );

//         const uint32 curX = (uint32)(x + p * entryStride32);

//         for( uint32 i = 0; i < entriesPerPage32; i++ )
//             xBuffer[i] = curX + i;

//         xBuffer += entryStride32;
//     }

//     // #TODO: Process last part
// }


///
/// kBC groups & matching
///

//-----------------------------------------------------------
uint64 MemPhase1::FpScan( const uint64 entryCount, const uint64* yBuffer, uint32* groupBoundaries, kBCJob jobs[MAX_THREADS] )
{
    MemPlotContext& cx  = _context;
    const uint32 threadCount        = cx.threadCount;
    const uint64 maxKBCGroups       = cx.maxKBCGroups;
    const uint64 maxGroupsPerThread = maxKBCGroups / threadCount;
    
    uint64 groupCount = 0;

    // auto timer = TimerBegin();
    // Log::Line( "  Scanning kBC groups..." );

    jobs[0].groupBoundaries = groupBoundaries;
    jobs[0].maxCount        = maxGroupsPerThread;
    jobs[0].yBuffer         = yBuffer;
    jobs[0].startIndex      = 0;
    jobs[0].endIndex        = entryCount;

    #if DEBUG
        jobs[0].jobIdx = 0;
    #endif

    // Find a starting position for each thread
    for( uint64 i = 1; i < threadCount; i++ )
    {
        auto& job = jobs[i];

        job.yBuffer    = yBuffer;
        job.groupCount = 0;
        job.copyDst    = nullptr;

        const uint64 idx      = entryCount / threadCount * i;
        const uint64 y        = yBuffer[idx];
        const uint64 curGroup = y / kBC;

        const uint32 groupLocalIdx = (uint32)(y - curGroup * kBC);

        uint64 targetGroup;

        // If we are already at the start of a group, just use this index
        if( groupLocalIdx == 0 )
        {
            job.startIndex = idx;
        }
        else
        {
            // Choose if we should find the upper boundary or the lower boundary
            const uint32 remainder = kBC - groupLocalIdx;
            
            #if _DEBUG
                bool foundBoundary = false;
            #endif
            if( remainder <= kBC / 2 )
            {
                // Look for the upper boundary
                for( uint64 j = idx+1; j < entryCount; j++ )
                {
                    targetGroup = yBuffer[j] / kBC;
                    if( targetGroup != curGroup )
                    {
                        #if _DEBUG
                            foundBoundary = true;
                        #endif
                        job.startIndex = j; break;
                    }   
                }
            }
            else
            {
                // Look for the lower boundary
                for( uint64 j = idx-1; j >= 0; j-- )
                {
                    targetGroup = yBuffer[j] / kBC;
                    if( targetGroup != curGroup )
                    {
                        #if _DEBUG
                            foundBoundary = true;
                        #endif
                        job.startIndex = j+1; break;
                    }  
                }
            }

            ASSERT( foundBoundary );
        }

        auto& lastJob = jobs[i-1];

        ASSERT( job.startIndex > lastJob.startIndex );

        lastJob.endIndex = job.startIndex+1;    // We must let it compare with
                                                // the start of this group so that 
                                                // the last job actually adds its last group.
                                                // #NOTE: We add +1 so that the next group boundary is added to the list,
                                                //        and we can tell where the R group ends.

        ASSERT( yBuffer[job.startIndex-1] / kBC != yBuffer[job.startIndex] / kBC );

        job.groupBoundaries = groupBoundaries + maxGroupsPerThread * i;
        job.maxCount        = maxGroupsPerThread;

        
        #if DEBUG
            job.jobIdx = (uint32)i;
        #endif
    }

    // Fill in missing data for the last job
    jobs[threadCount-1].endIndex = entryCount;

    // Run jobs
    cx.threadPool->RunJob( FpScanThread, jobs, threadCount );

    // Determine group count
    groupCount = 0;

    for( uint32 i = 0; i < threadCount-1; i++ )
    {
        auto& job = jobs[i];

        // Add a trailing end index so that we can test against it
        job.groupBoundaries[job.groupCount] = jobs[i+1].groupBoundaries[0];
        
        groupCount += job.groupCount;
    }
    groupCount += jobs[threadCount-1].groupCount;
        
    // Let the last job know where its R group
    auto& lastJob = jobs[threadCount-1];

    // This is an overflow if the last group ends @ k^32, if so,
    // then have it end just before that.
    if( entryCount == ENTRIES_PER_TABLE )
        lastJob.groupBoundaries[lastJob.groupCount] = (uint32)(entryCount-1);
    else
        lastJob.groupBoundaries[lastJob.groupCount] = (uint32)entryCount;

    // double elapsed = TimerEnd( timer );
    // Log::Line( "  Finished scanning kBC groups in %.2lf seconds.", elapsed );
    // Log::Line( "  Found %llu kBC groups.", groupCount );

    #if DBG_VALIDATE_KB_GROUPS
    {
        uint64 prevGroup = yBuffer[jobs[0].startIndex] / kBC;

        for( uint32 t = 0; t < threadCount; t++ )
        {
            const auto& job = jobs[t];

            // Ensure groups are ordered
            for( uint64 i = 0; i < job.groupCount; i++ )
            {
                const uint64 rIdx  = job.groupBoundaries[i];
                const uint64 group = yBuffer[rIdx] / kBC;

                if( group <= prevGroup )
                {
                    // ASSERT( group > prevGroup );
                    ASSERT( 0 );
                    Fatal( "Invalid group sequence. Job: %u Idx: %llu, Ridx: %llu, Group: %llu", t, i, rIdx, group );
                }
                prevGroup = group;
            }

            // Make sure all entries do belong to the same group
            uint64 groupStartL = job.startIndex;
            for( uint64 i = 0; i < job.groupCount; i++ )
            {
                const uint64 groupStartR = job.groupBoundaries[i];

                const uint64 groupL = yBuffer[groupStartL] / kBC;
                const uint64 groupR = yBuffer[groupStartR] / kBC;

                const uint64 groupDiff = groupR - groupL;
                if( groupDiff == 1 )
                {
                    for( uint64 j = groupStartL; j < groupStartR; j++ )
                    {
                        const uint64 group = yBuffer[j] / kBC;
                        if( group != groupL )
                        {
                            ASSERT( 0 );
                            Fatal( "Group validation failed @ job %llu index: %llu. L: %llu R: %llu", i, j, groupL, group );
                        }
                        ASSERT( group == groupL );
                    }
                }

                groupStartL = groupStartR;
            }
        }

    }
    #endif

    return groupCount;
}

//-----------------------------------------------------------
void FpScanThread( kBCJob* job )
{
    const uint64 maxGroups  = job->maxCount;

    uint32* groupBoundaries = job->groupBoundaries;
    uint64  groupCount      = 0;

    const uint64* yBuffer = job->yBuffer;
    const uint64  start   = job->startIndex;
    const uint64  end     = job->endIndex;

    uint64 lastGroup = yBuffer[start] / kBC;

    for( uint64 i = start+1; i < end; i++ )
    {
        uint64 group = yBuffer[i] / kBC;
        if( group != lastGroup )
        {
            ASSERT( group > lastGroup );

            groupBoundaries[groupCount++] = (uint32)i;
            lastGroup = group;

            if( groupCount == maxGroups )
            {
                ASSERT( 0 );    // We ought to always have enough space
                                // So this should be an error
                break;
            }
        }
    }

    job->groupCount = groupCount;
}

// Create pairs from y values
//-----------------------------------------------------------
uint64 MemPhase1::FpPair( const uint64* yBuffer, kBCJob jobs[MAX_THREADS],
                          const uint64 groupCount, Pair* tmpPairBuffer, Pair* outPairBuffer )
{
    MemPlotContext& cx = _context;

    const uint32 threadCount = cx.threadCount;

    uint64 pairCount = 0;

    Log::Line( "  Pairing L/R groups..." );
    auto timer = TimerBegin();

    const uint64 maxTotalpairs     = cx.maxPairs;
    const uint64 maxPairsPerThread = maxTotalpairs / threadCount;

    for( uint32 i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];

        job.pairs     = tmpPairBuffer + i * maxPairsPerThread;
        job.maxCount  = maxPairsPerThread;
        job.pairCount = 0;
        job.copyDst   = nullptr;

        // Test
        // job.groupBoundariesIdx = i * groupsPerThread;
        // job.jobIdx          = (uint32)i;
    }

    cx.threadPool->RunJob( FpPairThread, jobs, threadCount );

    // Count the total pairs and copy the pair buffers
    // to the actual destination pair buffer.
    pairCount = jobs[0].pairCount;
    jobs[0].copyDst = outPairBuffer;
    ASSERT( pairCount > 0 );

    for( uint32 i = 1; i < threadCount; i++ )
    {
        auto& job = jobs[i];
        ASSERT( job.pairCount > 0 );

        job.copyDst = outPairBuffer + pairCount;
        pairCount += job.pairCount;
    }

    // Sometimes we get more pairs than we support, so cap it.
    if( pairCount > ENTRIES_PER_TABLE )
    {
        const uint64 overflowEntries = pairCount - ENTRIES_PER_TABLE;

        auto& lastJob = jobs[threadCount-1];
        ASSERT( lastJob.pairCount >= overflowEntries );
        lastJob.pairCount -= overflowEntries;
       
        pairCount = ENTRIES_PER_TABLE;
    }

    cx.threadPool->RunJob( (JobFunc)[]( void* pdata ) {

        auto* job = (kBCJob*)pdata;
        memcpy( job->copyDst, job->pairs, job->pairCount * sizeof( Pair ) );

    }, jobs, threadCount, sizeof( kBCJob ) );

    auto elapsed = TimerEnd( timer );
    Log::Line( "  Finished pairing L/R groups in %.4lf seconds. Created %llu pairs.", elapsed, pairCount );
    Log::Line( "  Average of %.4lf pairs per group.", pairCount / (float64)groupCount );

    ASSERT( pairCount <= ENTRIES_PER_TABLE );

    #if DBG_TEST_PAIRS
        DbgTestPairs( pairCount, outPairBuffer, yBuffer );
    #endif

    return pairCount;
}

//-----------------------------------------------------------
void FpPairThread( kBCJob* job )
{
    const uint64  maxPairs        = job->maxCount;
    const uint32  groupCount      = (uint32)job->groupCount;
    const uint32* groupBoundaries = job->groupBoundaries;
    const uint64* yBuffer         = job->yBuffer;

    Pair*  pairs     = job->pairs;
    uint64 pairCount = 0;

    uint8  rMapCounts [kBC];
    uint16 rMapIndices[kBC];

    uint64 groupLStart = job->startIndex;
    uint64 groupL      = yBuffer[groupLStart] / kBC;

    for( uint32 i = 0; i < groupCount; i++ )
    {
        const uint64 groupRStart = groupBoundaries[i];
        const uint64 groupR      = yBuffer[groupRStart] / kBC;

        if( groupR - groupL == 1 )
        {
            // Groups are adjacent, calculate matches
            const uint16 parity           = groupL & 1;
            const uint64 groupREnd        = groupBoundaries[i+1];

            const uint64 groupLRangeStart = groupL * kBC;
            const uint64 groupRRangeStart = groupR * kBC;
            
            ASSERT( groupREnd - groupRStart <= 350 );
            ASSERT( groupLRangeStart == groupRRangeStart - kBC );

            // Prepare a map of range kBC to store which indices from groupR are used
            // For now just iterate our bGroup to find the pairs
           
            // #NOTE: memset(0) works faster on average than keeping a separate a clearing buffer
            memset( rMapCounts, 0, sizeof( rMapCounts ) );

            for( uint64 iR = groupRStart; iR < groupREnd; iR++ )
            {
                uint64 localRY = yBuffer[iR] - groupRRangeStart;
                ASSERT( yBuffer[iR] / kBC == groupR );

                if( rMapCounts[localRY] == 0 )
                    rMapIndices[localRY] = (uint16)( iR - groupRStart );

                rMapCounts[localRY] ++;
            }

            // For each group L entry
            for( uint64 iL = groupLStart; iL < groupRStart; iL++ )
            {
                const uint64 yL     = yBuffer[iL];
                const uint64 localL = yL - groupLRangeStart;

                // Iterate kExtraBitsPow = 1 << kExtraBits = 1 << 6 == 64
                // So iterate 64 times for each L entry.
                for( int iK = 0; iK < kExtraBitsPow; iK++ )
                {
                    const uint64 targetR = L_targets[parity][localL][iK];

                    for( uint j = 0; j < rMapCounts[targetR]; j++ )
                    {
                        const uint64 iR = groupRStart + rMapIndices[targetR] + j;

                        ASSERT( iL < iR );

                        // Add a new pair
                        Pair& pair = pairs[pairCount++];
                        pair.left  = (uint32)iL;
                        pair.right = (uint32)iR;
                        
                        ASSERT( pairCount <= maxPairs );
                        if( pairCount == maxPairs )
                            goto RETURN;
                    }
                }
            }
        }
        // Else: Not an adjacent group, skip to next one.

        // Go to next group
        groupL      = groupR;
        groupLStart = groupRStart;
    }

RETURN:
    job->pairCount = pairCount;
}

///
/// Fx Computation
///
template<TableId tableId, typename TMetaIn, typename TMetaOut>
void MemPhase1::FpComputeFx( const uint64 entryCount, const Pair* lrPairs,
                             const TMetaIn* inMetaBuffer, const uint64* inYBuffer,
                             TMetaOut* outMetaBuffer, uint64* outYBuffer )
{
    using TYOut = typename YOut<tableId>::Type;


    MemPlotContext& cx = _context;

    Log::Line( "  Computing Fx..." );
    auto timer = TimerBegin();
    
    const uint   threadCount     = cx.threadCount;
    const uint64 entriesPerThred = entryCount / threadCount;
    const uint64 trailingEntries = entryCount - (entriesPerThred * threadCount);

    // Table 7 needs 32-bit y outputs, so we have to change it here
    TYOut* tYOut = (TYOut*)outYBuffer;

    using Job = FpFxJob<TYOut, TMetaIn, TMetaOut>;
    Job jobs[MAX_THREADS];

    for( uint i = 0; i < threadCount; i++ )
    {
        Job& job = jobs[i];

        const size_t offset = entriesPerThred * i;

        job.entryCount    = entriesPerThred;
        job.inMetaBuffer  = inMetaBuffer;             // These should NOT be offseted as we 
        job.inYBuffer     = inYBuffer;                // use them as lookup tables based on the lrPairs
        job.lrPairs       = lrPairs       + offset;
        job.outMetaBuffer = outMetaBuffer + offset;
        job.outYBuffer    = tYOut         + offset;
    }

    // Add trailing entries to the last job
    jobs[threadCount-1].entryCount += trailingEntries;

    // Calculate Fx
    cx.threadPool->RunJob( ComputeFxJob<TYOut, TMetaIn, TMetaOut>, jobs, threadCount );

    auto elapsed = TimerEnd( timer );
    Log::Line( "  Finished computing Fx in %.4lf seconds.", elapsed );
}

//-----------------------------------------------------------
template<typename TYOut, typename TMetaIn, typename TMetaOut>
void ComputeFxJob( FpFxJob<TYOut, TMetaIn, TMetaOut>* job )
{
    const size_t metaKMultiplierIn  = SizeForMeta<TMetaIn >::Value;
    const size_t metaKMultiplierOut = SizeForMeta<TMetaOut>::Value;

    // Table 7 (identified by 0 metadata output) we don't have k + kExtraBits sized y's.
    // so we need to shift by 32 bits, instead of 26.
    constexpr size_t extraBitsShift = metaKMultiplierOut == 0 ? 0 : kExtraBits; 

    const uint64   entryCount    = job->entryCount;
    const Pair*    lrPairs       = job->lrPairs;
    const TMetaIn* inMetaBuffer  = job->inMetaBuffer;
    const uint64*  inYBuffer     = job->inYBuffer;
    TMetaOut*      outMetaBuffer = job->outMetaBuffer;
    TYOut*         outYBuffer    = job->outYBuffer;

    #if _DEBUG
        uint64 lastLeft = 0;
    #endif

    // Intermediate metadata holder
    uint64 lrMetadata[4];

    for( uint64 i = 0; i < entryCount; i++ )
    {
        const Pair& pair = lrPairs[i];

        #if _DEBUG
            ASSERT( pair.left >= lastLeft );
            lastLeft = pair.left;
        #endif

        // Read y
        const uint64 y = inYBuffer[pair.left];

        // Read metadata
        if constexpr( metaKMultiplierIn == 1 )
        {
            uint32* meta32 = (uint32*)lrMetadata;

            meta32[0] = inMetaBuffer[pair.left ];    // Metadata( l and r x's)
            meta32[1] = inMetaBuffer[pair.right];
        }
        else if constexpr( metaKMultiplierIn == 2 )
        {
            lrMetadata[0] = inMetaBuffer[pair.left ];
            lrMetadata[1] = inMetaBuffer[pair.right];
        }
        else
        {
            // For 3 and 4 we just use 16 bytes (2 64-bit entries)
            const Meta4* inMeta4 = static_cast<const Meta4*>( inMetaBuffer );
            const Meta4& meta4L  = inMeta4[pair.left ];
            const Meta4& meta4R  = inMeta4[pair.right];

            lrMetadata[0] = meta4L.m0;
            lrMetadata[1] = meta4L.m1;
            lrMetadata[2] = meta4R.m0;
            lrMetadata[3] = meta4R.m1;
        }

        TYOut f = (TYOut)ComputeFx<metaKMultiplierIn, metaKMultiplierOut, extraBitsShift>( y, lrMetadata, (uint64*)outMetaBuffer );

        outYBuffer[i] = f;

        if constexpr( metaKMultiplierOut != 0 )
            outMetaBuffer ++;
    }
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"

//-----------------------------------------------------------
template<size_t metaKMultiplierIn, size_t metaKMultiplierOut, uint ShiftBits>
FORCE_INLINE uint64 ComputeFx( uint64 y, uint64* metaData, uint64* metaOut )
{
    static_assert( metaKMultiplierIn != 0, "Invalid metaKMultiplier" );

    // Helper consts
    const uint   k           = _K;
    const uint32 ySize       = k + kExtraBits;         // = 38
    const uint32 yShift      = 64 - (k + ShiftBits);   // = 26 or 32
    const size_t metaSize    = k * metaKMultiplierIn;
    const size_t metaSizeLR  = metaSize * 2;

    const size_t bufferSize  = CDiv( ySize + metaSizeLR, 8 );


    // Hashing input and output buffers
    uint64 input [5];       // y + L + R
    uint64 output[4];       // blake3 hashed output

    blake3_hasher hasher;


    // Prepare the input buffer depending on the metadata size
    if constexpr( metaKMultiplierIn == 1 )
    {
        /**
         * 32-bit per metadata (1*32) L == 4 R == 4
         * Metadata: L: [32] R: [32]
         * 
         * Serialized:
         *  y  L     L  R -
         * [38|26]  [6|32|-]
         *    0        1
         */

        const uint64 l = reinterpret_cast<uint32*>( metaData )[0];
        const uint64 r = reinterpret_cast<uint32*>( metaData )[1];

        input[0] = Swap64( y << 26 | l >> 6  );
        input[1] = Swap64( l << 58 | r << 26 );

        // Metadata is just L + R of 8 bytes
        if constexpr( metaKMultiplierOut == 2 )
            metaOut[0] = l << 32 | r;
    }
    else if constexpr( metaKMultiplierIn == 2 )
    {
        /**
         * 64-bit per metadata (2*32) L == 8 R == 8
         * Metadata: L: [64] R: [64]
         * 
         * Serialized:
         *  y   L    L  R     R  -
         * [38|26]  [38|26]  [38|-]
         *    0        1       2
         */
        const uint64 l = metaData[0];
        const uint64 r = metaData[1];

        input[0] = Swap64( y << 26 | l >> 38 );
        input[1] = Swap64( l << 26 | r >> 38 );
        input[2] = Swap64( r << 26 );

        // Metadata is just L + R again of 16 bytes
        if constexpr( metaKMultiplierOut == 4 )
        {
            metaOut[0] = l;
            metaOut[1] = r;
        }
    }
    else if constexpr( metaKMultiplierIn == 3 )
    {
        /**
        * 96-bit per metadata (3*32) L == 12 bytes R == 12 bytes
        * Metadata: L: [64][32] R: [64][32]
        *               L0  L1      R0  R1 
        * Serialized:
        *  y  L0    L0 L1   L1 R0   R0 R1 -
        * [38|26]  [38|26]  [6|58]  [6|32|-]
        *    0        1       2        3
        */

        const uint64 l0 = metaData[0];
        const uint64 l1 = metaData[1] & 0xFFFFFFFF;
        const uint64 r0 = metaData[2];
        const uint64 r1 = metaData[3] & 0xFFFFFFFF;
        
        input[0] = Swap64( y  << 26 | l0 >> 38 );
        input[1] = Swap64( l0 << 26 | l1 >> 6  );
        input[2] = Swap64( l1 << 58 | r0 >> 6  );
        input[3] = Swap64( r0 << 58 | r1 << 26 );
    }
    else if constexpr( metaKMultiplierIn == 4 )
    {
        /**
        * 128-bit per metadata (4*32) L == 16 bytes R == 16 bytes
        * Metadata  : L [64][64] R: [64][64]
        *                L0  L1      R0  R1
        * Serialized: 
        *  y  L0    L0 L1    L1 R0    R0 R1    R1 -
        * [38|26]  [38|26]  [38|26]  [38|26]  [38|-]
        *    0        1        2        3        4
        */
        
        const uint64 l0 = metaData[0];
        const uint64 l1 = metaData[1];
        const uint64 r0 = metaData[2];
        const uint64 r1 = metaData[3];

        input[0] = Swap64( y  << 26 | l0 >> 38 );
        input[1] = Swap64( l0 << 26 | l1 >> 38 );
        input[2] = Swap64( l1 << 26 | r0 >> 38 );
        input[3] = Swap64( r0 << 26 | r1 >> 38 );
        input[4] = Swap64( r1 << 26 );
    }


    // Hash the input
    blake3_hasher_init( &hasher );
    blake3_hasher_update( &hasher, input, bufferSize );
    blake3_hasher_finalize( &hasher, (uint8_t*)output, sizeof( output ) );

    uint64 f = Swap64( *output ) >> yShift;


    ///
    /// Calculate metadata for tables >= 4
    ///
    // Only table 6 do we output size 2 with an input of size 3.
    // Otherwise for output == 2 we calculate the output above
    // as it is just L + R, and it is not taken from the output
    // of the blake3 hash.
    if constexpr ( metaKMultiplierOut == 2 && metaKMultiplierIn == 3 )
    {
        const uint64 h0 = Swap64( output[0] );
        const uint64 h1 = Swap64( output[1] );

        metaOut[0] = h0 << ySize | h1 >> 26;
    }
    else if constexpr ( metaKMultiplierOut == 3 )
    {
        const uint64 h0 = Swap64( output[0] );
        const uint64 h1 = Swap64( output[1] );
        const uint64 h2 = Swap64( output[2] );

        metaOut[0] = h0 << ySize | h1 >> 26;
        metaOut[1] = ((h1 << 6) & 0xFFFFFFC0) | h2 >> 58;
    }
    else if constexpr ( metaKMultiplierOut == 4 && metaKMultiplierIn != 2 ) // In = 2 is calculated above with L + R
    {
        const uint64 h0 = Swap64( output[0] );
        const uint64 h1 = Swap64( output[1] );
        const uint64 h2 = Swap64( output[2] );

        metaOut[0] = h0 << ySize | h1 >> 26;
        metaOut[1] = h1 << 38    | h2 >> 26;
    }
    
    return f;
}

#pragma GCC diagnostic pop



///
/// Some Debug Helpers
///

//-----------------------------------------------------------
bool DbgTestPairs( uint64 entryCount, const Pair* pairs )
{
    ASSERT( entryCount > 1 );

    // Test that for all pairs the left entry
    // are in sequential order. From least to greatest.
    uint64 lastLeft  = pairs[0].left;

    for( uint64 i = 1; i < entryCount; i++ )
    {
        const uint64 left = pairs[i].left;

        if( left < lastLeft )
        {
            ASSERT( left >= lastLeft );
            return false;
        }

        lastLeft = left;
    }

    return true;
}

//-----------------------------------------------------------
bool DbgVerifySortedY( const uint64 entryCount, const uint64* yBuffer )
{
    ASSERT( entryCount );
    
    uint64 lastY = 0;
    for( uint64 i = 0; i < entryCount; i++ )
    {
        const uint64 y = yBuffer[i];
        
        if( lastY > y )
        {
            ASSERT(0);
            return false;
        }
        lastY = y;
    }

    return true;
}

//-----------------------------------------------------------
bool DbgVerifyPairsKBCGroups( uint64 entryCount, const uint64* yBuffer, const Pair* pairs )
{
    Log::Line( "Verifying kBC groups from pairs..." );
    
    for( uint64 i = 0; i < entryCount; i++ )
    {
        const Pair& pair = pairs[i];

        const uint64 yl = yBuffer[pair.left ];
        const uint64 yr = yBuffer[pair.right];

        const uint64 lGroup = yl / kBC;
        const uint64 rGroup = yr / kBC;

        if( rGroup - lGroup != 1 )
        {
            ASSERT( 0 );
            return false;
        }
    }
    Log::Line( "OK!" );

    return true;
}

