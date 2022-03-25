#include "DiskPlotPhase2.h"
#include "util/BitField.h"
#include "algorithm/RadixSort.h"
#include "transforms/FxTransform.h"
#include "plotdisk/DiskPlotInfo.h"
#include "util/StackAllocator.h"
#include "DiskPlotInfo.h"
#include "plotdisk/DiskPairReader.h"


// Fence ids used when loading buckets
struct FenceId
{
    enum
    {
        None = 0,
        MapLoaded,
        PairsLoaded,

        FenceCount
    };
};

struct MarkJob : MTJob<MarkJob>
{
    TableId          table;
    uint32           entryCount;
    Pairs            pairs;
    const uint32*    map;

    DiskPlotContext* context;

    uint64*          lTableMarkedEntries;
    uint64*          rTableMarkedEntries;

    uint64           lTableOffset;
    uint32           pairBucket;
    uint32           pairBucketOffset;


public:
    void Run() override;

    template<TableId table>
    void MarkEntries();

    template<TableId table>
    inline int32 MarkStep( int32 i, const int32 entryCount, BitField lTable, const BitField rTable,
                           uint64 lTableOffset, const Pairs& pairs, const uint32* map );
};

struct StripMapJob : MTJob<StripMapJob>
{
    uint32        entryCount;
    const uint64* inMap;
    uint32*       outKey;
    uint32*       outMap;

    void Run() override;
};

//-----------------------------------------------------------
DiskPlotPhase2::DiskPlotPhase2( DiskPlotContext& context )
    : _context( context )
{
    memset( _bucketBuffers, 0, sizeof( _bucketBuffers ) );

    DiskBufferQueue& ioQueue = *context.ioQueue;

    // #TODO: Give the cache to the marks? Probably not needed for sucha small write...
    //        Then we would need to re-distribute the cache on Phase 3.
    // #TODO: We need to specify the temporary file location
    ioQueue.InitFileSet( FileId::MARKED_ENTRIES_2, "table_2_marks", 1, FileSetOptions::DirectIO, nullptr );
    ioQueue.InitFileSet( FileId::MARKED_ENTRIES_3, "table_3_marks", 1, FileSetOptions::DirectIO, nullptr );
    ioQueue.InitFileSet( FileId::MARKED_ENTRIES_4, "table_4_marks", 1, FileSetOptions::DirectIO, nullptr );
    ioQueue.InitFileSet( FileId::MARKED_ENTRIES_5, "table_5_marks", 1, FileSetOptions::DirectIO, nullptr );
    ioQueue.InitFileSet( FileId::MARKED_ENTRIES_6, "table_6_marks", 1, FileSetOptions::DirectIO, nullptr );

    ioQueue.SeekFile( FileId::T1, 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile( FileId::T2, 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile( FileId::T3, 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile( FileId::T4, 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile( FileId::T5, 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile( FileId::T6, 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile( FileId::T7, 0, 0, SeekOrigin::Begin );

    ioQueue.SeekBucket( FileId::MAP2, 0, SeekOrigin::Begin );
    ioQueue.SeekBucket( FileId::MAP3, 0, SeekOrigin::Begin );
    ioQueue.SeekBucket( FileId::MAP4, 0, SeekOrigin::Begin );
    ioQueue.SeekBucket( FileId::MAP5, 0, SeekOrigin::Begin );
    ioQueue.SeekBucket( FileId::MAP6, 0, SeekOrigin::Begin );
    ioQueue.SeekBucket( FileId::MAP7, 0, SeekOrigin::Begin );

    ioQueue.CommitCommands();
}

//-----------------------------------------------------------
DiskPlotPhase2::~DiskPlotPhase2() {}

//-----------------------------------------------------------
void DiskPlotPhase2::Run()
{
    switch( _context.numBuckets )
    {
        case 128 : RunWithBuckets<128 >(); break;
        case 256 : RunWithBuckets<256 >(); break;
        case 512 : RunWithBuckets<512 >(); break;
        case 1024: RunWithBuckets<1024>(); break;
        
        default:
        ASSERT( 0 );
            break;
    }
}

//-----------------------------------------------------------
template<uint32 _numBuckets>
void DiskPlotPhase2::RunWithBuckets()
{
    DiskPlotContext& context = _context;
    DiskBufferQueue& queue   = *context.ioQueue;

    StackAllocator allocator( context.heapBuffer, context.heapSize );
    ////
    Fence fence;
    

    const uint64 maxBucketEntries = (uint64)DiskPlotInfo<TableId::Table1, _numBuckets>::MaxBucketEntries;
    Pair*   pairs = allocator.CAlloc<Pair>  ( maxBucketEntries );
    uint64* map   = allocator.CAlloc<uint64>( maxBucketEntries );

    uint64 maxEntries = context.entryCounts[0];
    for( TableId table = TableId::Table2; table <= TableId::Table7; table++ )
        maxEntries = std::max( maxEntries, context.entryCounts[(int)table] );

    const size_t blockSize     = _context.tmp2BlockSize;
    const size_t markfieldSize = RoundUpToNextBoundaryT( maxEntries / 8, (uint64)blockSize );

    // Prepare 2 marking bitfields for dual-buffering
    uint64* bitFields[2];
    bitFields[0] = allocator.AllocT<uint64>( markfieldSize, blockSize );
    bitFields[1] = allocator.AllocT<uint64>( markfieldSize, blockSize );


    // reader.LoadNextBucket();
    // reader.LoadNextBucket();
    // reader.UnpackBucket( 0, pairs, map );
    // reader.UnpackBucket( 1, pairs, map );

    // _phase3Data.bitFieldSize   = bitFieldSize;
    // _phase3Data.maxTableLength = maxEntries;

    #if _DEBUG && BB_DP_DBG_SKIP_PHASE_2
        return;
    #endif

    // // Prepare map buffer
    // _tmpMap = (uint32*)( context.heapBuffer + bitFieldSize*2 );

    // // Prepare our fences
    // Fence bitFieldFence, bucketLoadFence, mapWriteFence;
    // _bucketReadFence = &bucketLoadFence;
    // _mapWriteFence   = &mapWriteFence;

    // // Set write fence as signalled initially
    // _mapWriteFence->Signal();

    // Mark all tables
    // FileId lTableFileId = FileId::MARKED_ENTRIES_6;

    for( TableId table = TableId::Table7; table > TableId::Table2; table = table-1 )
    {
        const auto timer = TimerBegin();

        uint64* lMarkingTable = bitFields[0];
        uint64* rMarkingTable = bitFields[1];
        
        const size_t stackMarker = allocator.Size();
        DiskPairAndMapReader<_numBuckets> reader( context, fence, table, allocator );

        ASSERT( allocator.Size() < context.heapSize );

        // Log::Line( "Allocated work heap of %.2lf GiB out of %.2lf GiB.", 
        //     (double)(allocator.Size() + markfieldSize*2 ) BtoGB, (double)context.heapSize BtoGB );

        MarkTable<_numBuckets>( table, reader, lMarkingTable, rMarkingTable );

    //     //
    //     // #TEST
    //     //
    //     #if 0
    //     if( 0 )
    //     {
    //         uint32* lPtrBuf = bbcvirtalloc<uint32>( 1ull << _K );
    //         uint16* rPtrBuf = bbcvirtalloc<uint16>( 1ull << _K );

    //         byte* rMarkedBuffer = bbcvirtalloc<byte>( 1ull << _K );
    //         byte* lMarkedBuffer = bbcvirtalloc<byte>( 1ull << _K );
            
    //         for( TableId rTable = TableId::Table7; rTable > TableId::Table1; rTable = rTable-1 )
    //         {
    //             const TableId lTable = rTable-1;

    //             const uint64 rEntryCount = context.entryCounts[(int)rTable];
    //             const uint64 lEntryCount = context.entryCounts[(int)lTable];

    //             // BitField rMarkedEntries( (uint64*)rMarkedBuffer );
    //             // BitField lMarkedEntries( (uint64*)lMarkedBuffer );

    //             Log::Line( "Reading R table %u...", rTable+1 );
    //             {
    //                 const FileId rTableIdL = TableIdToBackPointerFileId( rTable );
    //                 const FileId rTableIdR = (FileId)((int)rTableIdL + 1 );

    //                 Fence fence;
    //                 queue.ReadFile( rTableIdL, 0, lPtrBuf, sizeof( uint32 ) * rEntryCount );
    //                 queue.ReadFile( rTableIdR, 0, rPtrBuf, sizeof( uint16 ) * rEntryCount );
    //                 queue.SignalFence( fence );
    //                 queue.CommitCommands();
    //                 fence.Wait();
    //             }


    //             uint32* lPtr = lPtrBuf;
    //             uint16* rPtr = rPtrBuf;
                
    //             uint64 lEntryOffset = 0;
    //             uint64 rTableOffset = 0;

    //             Log::Line( "Marking entries..." );
    //             for( uint32 bucket = 0; bucket < BB_DP_BUCKET_COUNT; bucket++ )
    //             {
    //                 const uint32 rBucketCount = context.ptrTableBucketCounts[(int)rTable][bucket];
    //                 const uint32 lBucketCount = context.bucketCounts[(int)lTable][bucket];

    //                 for( uint e = 0; e < rBucketCount; e++ )
    //                 {
    //                     // #NOTE: The bug is related to this.
    //                     //        Somehow the entries we get from the R table
    //                     //        are not filtering properly...
    //                     //        We tested without this and got the exact same
    //                     //        results from the reference implementation
    //                     if( rTable < TableId::Table7 )
    //                     {
    //                         const uint64 rIdx = rTableOffset + e;
    //                         // if( !rMarkedEntries.Get( rIdx ) )
    //                         if( !rMarkedBuffer[rIdx] )
    //                             continue;
    //                     }

    //                     uint64 l = (uint64)lPtr[e] + lEntryOffset;
    //                     uint64 r = (uint64)rPtr[e] + l;

    //                     ASSERT( l < ( 1ull << _K ) );
    //                     ASSERT( r < ( 1ull << _K ) );

    //                     lMarkedBuffer[l] = 1;
    //                     lMarkedBuffer[r] = 1;
    //                     // lMarkedEntries.Set( l );
    //                     // lMarkedEntries.Set( r );
    //                 }

    //                 lPtr += rBucketCount;
    //                 rPtr += rBucketCount;

    //                 lEntryOffset += lBucketCount;
    //                 rTableOffset += context.bucketCounts[(int)rTable][bucket];
    //             }

    //             uint64 prunedEntryCount = 0;
    //             Log::Line( "Counting entries." );
    //             for( uint64 e = 0; e < lEntryCount; e++ )
    //             {
    //                 if( lMarkedBuffer[e] )
    //                     prunedEntryCount++;
    //                 // if( lMarkedEntries.Get( e ) )
    //             }

    //             Log::Line( " %llu/%llu (%.2lf%%)", prunedEntryCount, lEntryCount,
    //                 ((double)prunedEntryCount / lEntryCount) * 100.0 );
    //             Log::Line("");

    //             // Swap marking tables and zero-out the left one.
    //             std::swap( lMarkedBuffer, rMarkedBuffer );
    //             memset( lMarkedBuffer, 0, 1ull << _K );
    //         }
    //     }
    //     #endif

        // // Ensure the last table finished writing to the bitfield
        // if( table < TableId::Table7 )
        //     bitFieldFence.Wait( _context.writeWaitTime );

    //     // Submit l marking table for writing
    //     queue.WriteFile( lTableFileId, 0, lMarkingTable, bitFieldSize );
    //     queue.SignalFence( bitFieldFence );
    //     queue.CommitCommands();

    //     // Swap marking tables
    //     std::swap( bitFields[0], bitFields[1] );
    //     lTableFileId = (FileId)( (int)lTableFileId - 1 );

        const double elapsed = TimerEnd( timer );
        Log::Line( "Finished marking table %d in %.2lf seconds.", table, elapsed );

        allocator.PopToMarker( stackMarker );
        ASSERT( allocator.Size() == stackMarker );
        // Log::Line( " Table %u IO Aggregate Wait Time | READ: %.4lf | WRITE: %.4lf | BUFFERS: %.4lf", table,
        //     TicksToSeconds( context.readWaitTime ), TicksToSeconds( context.writeWaitTime ), context.ioQueue->IOBufferWaitTime() );

    //     // #TEST:
    //     // if( table < TableId::Table7 )
    //     // if( 0 )
    //     // {
    //     //     BitField markedEntries( bitFields[1] );
    //     //     uint64 lTableEntries = context.entryCounts[(int)table-1];

    //     //     uint64 bucketsTotalCount = 0;
    //     //     for( uint64 e = 0; e < BB_DP_BUCKET_COUNT; ++e )
    //     //         bucketsTotalCount += context.ptrTableBucketCounts[(int)table-1][e];

    //     //     ASSERT( bucketsTotalCount == lTableEntries );

    //     //     uint64 lTablePrunedEntries = 0;

    //     //     for( uint64 e = 0; e < lTableEntries; ++e )
    //     //     {
    //     //         if( markedEntries.Get( e ) )
    //     //             lTablePrunedEntries++;
    //     //     }

    //     //     Log::Line( "Table %u entries: %llu/%llu (%.2lf%%)", table,
    //     //                lTablePrunedEntries, lTableEntries, ((double)lTablePrunedEntries / lTableEntries ) * 100.0 );
    //     //     Log::Line( "" );

    //     // }
    }

    // bitFieldFence.Wait( _context.writeWaitTime );
    // queue.CompletePendingReleases();

    // // Unpack table 2 and 7's map here to to make Phase 3 easier, though this will issue more read/writes
    // UnpackTableMap( TableId::Table7 );
    // UnpackTableMap( TableId::Table2 );

    // Log::Line( " Phase 2 Total IO Aggregate Wait Time | READ: %.4lf | WRITE: %.4lf | BUFFERS: %.4lf", 
    //         TicksToSeconds( context.readWaitTime ), TicksToSeconds( context.writeWaitTime ), context.ioQueue->IOBufferWaitTime() );
}

//-----------------------------------------------------------
template<uint32 _numBuckets>
void DiskPlotPhase2::MarkTable( TableId table, DiskPairAndMapReader<_numBuckets> reader, uint64* lTableMarks, uint64* rTableMarks )
{
    // Load initial bucket
    reader.LoadNextBucket();

    for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
    {
        reader.LoadNextBucket();

        // reader.UnpackBucket()
        
    }
}

//-----------------------------------------------------------
// void DiskPlotPhase2::MarkTable( TableId table, uint64* lTableMarks, uint64* rTableMarks )
// {
//     DiskPlotContext& context = _context;
//     DiskBufferQueue& queue   = *context.ioQueue;

//     const FileId rMapId       = table < TableId::Table7 ? TableIdToMapFileId( table ) : FileId::None;
//     const FileId rTableLPtrId = TableIdToBackPointerFileId( table );
//     const FileId rTableRPtrId = (FileId)((int)rTableLPtrId + 1 );

//     // Seek the table files back to the beginning
//     queue.SeekFile( rTableLPtrId, 0, 0, SeekOrigin::Begin );
//     queue.SeekFile( rTableRPtrId, 0, 0, SeekOrigin::Begin );
//     queue.CommitCommands();

//     if( rMapId != FileId::None )
//     {
//         queue.SeekBucket( rMapId, 0, SeekOrigin::Begin );
//         queue.CommitCommands();
//     }

//     const uint64 maxEntries          = 1ull << _K;
//     const uint32 maxEntriesPerBucket = (uint32)( maxEntries / (uint64)BB_DP_BUCKET_COUNT );
//     const uint64 tableEntryCount     = context.entryCounts[(int)table];
    
//     _bucketsLoaded = 0;
//     _bucketReadFence->Reset( 0 );

//     const uint32 threadCount = context.p2ThreadCount;

//     uint64 lTableEntryOffset     = 0;
//     uint32 pairBucket            = 0;   // Pair bucket we are processing (may be different than 'bucket' which refers to the map bucket)
//     uint32 pairBucketEntryOffset = 0;   // Offset in the current pair bucket 

//     for( uint32 bucket = 0; bucket - BB_DP_BUCKET_COUNT; bucket++ )
//     {
//         uint32  bucketEntryCount;
//         uint64* unsortedMapBuffer;
//         Pairs   pairs;

//         // Load as many buckets as we can in the background
//         LoadNextBuckets( table, bucket, unsortedMapBuffer, pairs, bucketEntryCount );

//         uint32* map = nullptr;
//         const uint32 waitFenceId = bucket * FenceId::FenceCount;
        
//         if( rMapId != FileId::None )
//         {
//             // Wait for the map to finish loading
//             _bucketReadFence->Wait( FenceId::MapLoaded + waitFenceId, _context.readWaitTime );

//             // Ensure the map buffer isn't being used anymore (we use the tmp map buffer for this)
//             _mapWriteFence->Wait( _context.writeWaitTime );

//             // Sort the lookup map and strip out the origin index
//             // const auto stripTimer = TimerBegin();
//             map = UnpackMap( unsortedMapBuffer, bucketEntryCount, bucket );
//             // const auto stripElapsed = TimerEnd( stripTimer );
//             // Log::Line( "  Stripped bucket %u in %.2lf seconds.", bucket, stripElapsed );

//             // Write the map back to disk & release the buffer
//             queue.ReleaseBuffer( _bucketBuffers[bucket].map );
            
//             if( bucket > 0 )
//                 queue.DeleteFile( rMapId, bucket );

//             queue.WriteFile( rMapId, 0, map, bucketEntryCount * sizeof( uint32 ) );
//             queue.SignalFence( *_mapWriteFence );
//             queue.CommitCommands();
//         }

//         // Wait for the pairs to finish loading
//         _bucketReadFence->Wait( FenceId::PairsLoaded + waitFenceId, _context.readWaitTime );

//         // Mark the entries on this bucket
//         MTJobRunner<MarkJob> jobs( *context.threadPool );

//         for( uint i = 0; i < threadCount; i++ )
//         {
//             MarkJob& job = jobs[i];

//             job.table               = (TableId)table;
//             job.entryCount          = bucketEntryCount;
//             job.pairs               = pairs;
//             job.map                 = map;
//             job.context             = &context;

//             job.lTableMarkedEntries = lTableMarks;
//             job.rTableMarkedEntries = rTableMarks;

//             job.lTableOffset        = lTableEntryOffset;
//             job.pairBucket          = pairBucket;
//             job.pairBucketOffset    = pairBucketEntryOffset;
//         }

//         jobs.Run( threadCount );

//         // Release the paiors buffer we just used
//         ASSERT( _bucketBuffers[bucket].pairs.left < queue.Heap().Heap() + queue.Heap().HeapSize() );
//         queue.ReleaseBuffer( _bucketBuffers[bucket].pairs.left );
//         queue.CommitCommands();

//         // Update our offsets
//         lTableEntryOffset     = jobs[0].lTableOffset;
//         pairBucket            = jobs[0].pairBucket;
//         pairBucketEntryOffset = jobs[0].pairBucketOffset;
//     }
// }

//-----------------------------------------------------------
void DiskPlotPhase2::LoadNextBuckets( TableId table, uint32 bucket, uint64*& outMapBuffer, Pairs& outPairsBuffer, uint32& outBucketEntryCount )
{
    DiskPlotContext& context = _context;
    DiskBufferQueue& queue   = *context.ioQueue;

    const FileId rMapId              = table < TableId::Table7 ? TableIdToMapFileId( table ) : FileId::None;
    const FileId rTableLPtrId        = TableIdToBackPointerFileId( table );
    const FileId rTableRPtrId        = (FileId)((int)rTableLPtrId + 1 );

    const uint64 maxEntries          = 1ull << _K;
    const uint32 maxEntriesPerBucket = (uint32)( maxEntries / (uint64)BB_DP_BUCKET_COUNT );
    const uint64 tableEntryCount     = context.entryCounts[(int)table];

    // Load as many buckets as we're able to
    const uint32 maxBucketsToLoad = _bucketsLoaded + 2;   // Only load 2 buckets per pass max for now (Need to allow space for map and table writes as well)

    while( _bucketsLoaded < BB_DP_BUCKET_COUNT )
    {
        const uint32 bucketToLoadEntryCount = _bucketsLoaded < BB_DP_BUCKET_COUNT - 1 ?
                                              maxEntriesPerBucket :
                                              (uint32)( tableEntryCount - maxEntriesPerBucket * ( BB_DP_BUCKET_COUNT - 1 ) ); // Last bucket

        // #TODO: I think we need to ne loading a different amount for the L table and the R table on the last bucket.
        // #TODO: Block-align size?
        // Reserve a buffer to load both a map bucket and the same amount of entries worth of pairs.
        const size_t mapReadSize  = rMapId != FileId::None ? sizeof( uint64 ) * bucketToLoadEntryCount : 0;
        const size_t lReadSize    = sizeof( uint32 ) * bucketToLoadEntryCount;
        const size_t rReadSize    = sizeof( uint16 ) * bucketToLoadEntryCount;
        const size_t pairReadSize = lReadSize + rReadSize;
        const size_t totalSize    = mapReadSize + pairReadSize;

        // Break out if a buffer isn't available, and we don't actually require one
        if( !queue.Heap().CanAllocate( totalSize ) && _bucketsLoaded > bucket )
            break;
        
        PairAndMap& buffer = _bucketBuffers[_bucketsLoaded];  // Store the buffer for the other threads to use
        ZeroMem( &buffer );
        
        if( mapReadSize > 0 )
            buffer.map = (uint64*)queue.GetBuffer( mapReadSize , true );

        buffer.pairs.left  = (uint32*)queue.GetBuffer( pairReadSize, true );
        buffer.pairs.right = (uint16*)( buffer.pairs.left + bucketToLoadEntryCount );

        const uint32 loadFenceId = _bucketsLoaded * FenceId::FenceCount;

        if( mapReadSize > 0 )
        {
            queue.ReadFile( rMapId, _bucketsLoaded, buffer.map, mapReadSize );
            queue.SignalFence( *_bucketReadFence, FenceId::MapLoaded + loadFenceId );

            // Seek the file back to origin, and over-write it.
            // If it's not the origin bucket, then just delete the file, don't need it anymore
            if( _bucketsLoaded == 0 )
                queue.SeekFile( rMapId, 0, 0, SeekOrigin::Begin );
            // else
            //     queue.DeleteFile( rMapId, _bucketsLoaded );
        }

        queue.ReadFile( rTableLPtrId, 0, buffer.pairs.left , lReadSize );
        queue.ReadFile( rTableRPtrId, 0, buffer.pairs.right, rReadSize );
        queue.SignalFence( *_bucketReadFence, FenceId::PairsLoaded + loadFenceId );

        queue.CommitCommands();
        _bucketsLoaded++;

        if( _bucketsLoaded >= maxBucketsToLoad )
            break;
    }

    {
        ASSERT( _bucketsLoaded > bucket );
        
        const uint32 entryCount = bucket < BB_DP_BUCKET_COUNT - 1 ?
                                    maxEntriesPerBucket :
                                    (uint32)( tableEntryCount - maxEntriesPerBucket * ( BB_DP_BUCKET_COUNT - 1 ) ); // Last bucket

        const PairAndMap& buffer = _bucketBuffers[bucket];

        outMapBuffer        = rMapId != FileId::None ? buffer.map : nullptr;
        outPairsBuffer      = buffer.pairs;
        outBucketEntryCount = entryCount;
    }
}


// #TODO: Consolidate this job w/ Phase3 again
struct UnpackMapJob : MTJob<UnpackMapJob>
{
    uint32        bucket;
    uint32        entryCount;
    const uint64* mapSrc;
    uint32*       mapDst;

    //-----------------------------------------------------------
    static void RunJob( ThreadPool& pool, const uint32 threadCount, const uint32 bucket,
                        const uint32 entryCount, const uint64* mapSrc, uint32* mapDst )
    {
        MTJobRunner<UnpackMapJob> jobs( pool );

        for( uint32 i = 0; i < threadCount; i++ )
        {
            auto& job = jobs[i];
            job.bucket     = bucket;
            job.entryCount = entryCount;
            job.mapSrc     = mapSrc;
            job.mapDst     = mapDst;
        }

        jobs.Run( threadCount );
    }

    //-----------------------------------------------------------
    void Run() override
    {
        const uint64 maxEntries         = 1ull << _K ;
        const uint32 fixedBucketLength  = (uint32)( maxEntries / BB_DP_BUCKET_COUNT );
        const uint32 bucketOffset       = fixedBucketLength * this->bucket;


        const uint32 threadCount = this->JobCount();
        uint32 entriesPerThread = this->entryCount / threadCount;

        const uint32 offset = entriesPerThread * this->JobId();

        if( this->IsLastThread() )
            entriesPerThread += this->entryCount - entriesPerThread * threadCount;

        const uint64* mapSrc = this->mapSrc + offset;
        uint32*       mapDst = this->mapDst;

        // Unpack with the bucket id
        for( uint32 i = 0; i < entriesPerThread; i++ )
        {
            const uint64 m   = mapSrc[i];
            const uint32 idx = (uint32)m - bucketOffset;
            
            ASSERT( idx < this->entryCount );

            mapDst[idx] = (uint32)(m >> 32);
        }
    }
};

//-----------------------------------------------------------
uint32* DiskPlotPhase2::UnpackMap( uint64* map, uint32 entryCount, const uint32 bucket )
{
    auto& context = _context;

    UnpackMapJob::RunJob( 
            *context.threadPool, context.p2ThreadCount,
            bucket, entryCount, map, _tmpMap );

    return _tmpMap;
}

//-----------------------------------------------------------
void DiskPlotPhase2::UnpackTableMap( TableId table )
{
    auto& context = _context;
    auto& ioQueue = *context.ioQueue;

    Log::Line( "Unpacking table %u's map...", table + 1 );
    const auto timer = TimerBegin();

    const FileId mapId = TableIdToMapFileId( table );

    const uint64 maxEntries = 1ull << _K;

    uint32 bucketEntryCount = (uint32)( maxEntries / BB_DP_BUCKET_COUNT);
    uint32 lastBucketCount  = (uint32)( context.entryCounts[(int)table] - bucketEntryCount * (BB_DP_BUCKET_COUNT-1) );
    
    size_t bucketSize = bucketEntryCount * sizeof( uint64 );

    // Take some heap space for the tmp stripping buffer
    // const size_t tmpBucketSize = std::max( bucketEntryCount, lastBucketCount ) * sizeof( uint32);
    const size_t totalHeapSize = context.heapSize + context.heapSize;
    
    byte*   heap      = context.heapBuffer;
    // uint32* tmpBucket = (uint32*)heap;  

    // Reset our heap to use what remains
    ioQueue.ResetHeap( totalHeapSize, heap );
    ioQueue.SeekBucket( mapId, 0, SeekOrigin::Begin );
    ioQueue.CommitCommands();

    // Load the buckets
    uint32 maxBucketsToLoadPerIter = 2;
    uint32 bucketsLoaded           = 1;

    // Load first bucket
    Fence fence;

    uint64* buckets[BB_DP_BUCKET_COUNT];

    buckets[0] = (uint64*)ioQueue.GetBuffer( bucketSize );

    ioQueue.ReadFile( mapId, 0, buckets[0], bucketSize );
    ioQueue.SeekFile( mapId, 0, 0, SeekOrigin::Begin );      // Seek back since we will re-write to the first bucket
    ioQueue.SignalFence( fence, 1 );
    ioQueue.CommitCommands();

    for( uint32 bucket = 0; bucket < BB_DP_BUCKET_COUNT; bucket++ )
    {
        if( bucket == BB_DP_BUCKET_COUNT - 1)
            bucketEntryCount = lastBucketCount;

        // Obtain a bucket for writing
        uint32* writeBucket = (uint32*)ioQueue.GetBuffer( bucketEntryCount * sizeof( uint32 ) );

        // Load buckets in the background if we need to
        if( bucketsLoaded < BB_DP_BUCKET_COUNT )
        {
            uint32 bucketsToLoadCount = std::min( BB_DP_BUCKET_COUNT - bucketsLoaded, maxBucketsToLoadPerIter );

            for( uint32 i = 0; i < bucketsToLoadCount; i++ )
            {
                const uint32 bucketToLoadEntryCount = bucketsLoaded < BB_DP_BUCKET_COUNT-1 ?
                                                      bucketEntryCount : lastBucketCount;

                // If we don't currently have a bucket, we need to force to load a bucket now
                const bool   blockForBuffer   = bucketsLoaded == bucket;
                const size_t bucketToLoadSize = sizeof( uint64 ) * bucketToLoadEntryCount;

                byte* buffer = ioQueue.GetBuffer( bucketToLoadSize, blockForBuffer );
                if( !buffer )
                    break;

                ioQueue.ReadFile( mapId, bucketsLoaded, buffer, bucketToLoadSize );
                ioQueue.SignalFence( fence, bucketsLoaded+1 );
                ioQueue.DeleteFile( mapId, bucketsLoaded );
                ioQueue.CommitCommands();

                buckets[bucketsLoaded++] = (uint64*)buffer;
            }
        }

        // Ensure the bucket has finished loading
        fence.Wait( bucket + 1, _context.readWaitTime );

        // Unpack the map to its target position given its origin index
        uint64* map = buckets[bucket];

        // const auto jobTimer = TimerBegin();

        UnpackMapJob::RunJob( 
            *context.threadPool, context.p2ThreadCount,
            bucket, bucketEntryCount, map, writeBucket );
        
        // const auto jobElapsed = TimerEnd( jobTimer );
        // Log::Line( " Unpacked bucket %u in %.2lf seconds.", bucket, jobElapsed );

        // Write back to disk
        ioQueue.ReleaseBuffer( map );
        ioQueue.WriteFile( mapId, 0, writeBucket, sizeof( uint32 ) * bucketEntryCount );
        ioQueue.ReleaseBuffer( writeBucket );
        ioQueue.CommitCommands();
    }

    // Wait for last write to finish
    fence.Reset( 0 );
    ioQueue.SignalFence( fence, 1 );
    ioQueue.CommitCommands();
    fence.Wait( 1 );
    ioQueue.CompletePendingReleases();

    const double elapsed = TimerEnd( timer );
    Log::Line( "Finished unpacking table %u map in %.2lf seconds.", table + 1, elapsed );

}


//-----------------------------------------------------------
void MarkJob::Run()
{
    switch( this->table )
    {
        case TableId::Table7: this->MarkEntries<TableId::Table7>(); return;
        case TableId::Table6: this->MarkEntries<TableId::Table6>(); return;
        case TableId::Table5: this->MarkEntries<TableId::Table5>(); return;
        case TableId::Table4: this->MarkEntries<TableId::Table4>(); return;
        case TableId::Table3: this->MarkEntries<TableId::Table3>(); return;

        default:
            ASSERT( 0 );
            return;
    }

    ASSERT( 0 );
}

//-----------------------------------------------------------
template<TableId table>
void MarkJob::MarkEntries()
{
    DiskPlotContext& context = *this->context;
    DiskBufferQueue& queue   = *context.ioQueue;

    const uint32 jobId               = this->JobId();
    const uint32 threadCount         = this->JobCount();
    const uint64 maxEntries          = 1ull << _K;
    const uint32 maxEntriesPerBucket = (uint32)( maxEntries / (uint64)BB_DP_BUCKET_COUNT );
    const uint64 tableEntryCount     = context.entryCounts[(int)table];

    BitField lTableMarkedEntries( this->lTableMarkedEntries );
    BitField rTableMarkedEntries( this->rTableMarkedEntries );

    // Zero-out our portion of the bit field and sync, do this only on the first run
    if( this->pairBucket == 0 && this->pairBucketOffset == 0 )
    {
        const size_t bitFieldSize  = RoundUpToNextBoundary( (size_t)maxEntries / 8, 8 );  // Round up to 64-bit boundary

              size_t sizePerThread = bitFieldSize / threadCount;
        const size_t sizeRemainder = bitFieldSize - sizePerThread * threadCount;

        byte* buffer = ((byte*)this->lTableMarkedEntries) + sizePerThread * jobId;

        if( jobId == threadCount - 1 )
            sizePerThread += sizeRemainder;

        memset( buffer, 0, sizePerThread );
        this->SyncThreads();
    }

    const uint32* map   = this->map;
    Pairs         pairs = this->pairs;
    
    uint32 bucketEntryCount = this->entryCount;

    // Determine how many passes we need to run for this bucket.
    // Passes are determined depending on the range were currently processing
    // on the pairs buffer. Since they have different starting offsets after each
    // L table bucket length that generated its pairs, we need to update that offset
    // after we reach the boundary of the buckets that generated the pairs.
    while( bucketEntryCount )
    {
        uint32 pairBucket           = this->pairBucket;
        uint32 pairBucketOffset     = this->pairBucketOffset;
        uint64 lTableOffset         = this->lTableOffset;

        uint32 pairBucketEntryCount = context.ptrTableBucketCounts[(int)table][pairBucket];

        uint32 passEntryCount       = std::min( pairBucketEntryCount - pairBucketOffset, bucketEntryCount );

        // Prune the table
        {
            // We need a minimum number of entries per thread to ensure that we don't,
            // write to the same qword in the bit field. So let's ensure that each thread
            // has at least more than 2 groups worth of entries.
            // There's an average of 284,190 entries per bucket, which means each group
            // has an about 236.1 entries. We round up to 280 entries.
            // We use minimum 3 groups and round up to 896 entries per thread which gives us
            // 14 QWords worth of area each threads can reference.
            const uint32 minEntriesPerThread = 896;
            
            uint32 threadsToRun     = threadCount;
            uint32 entriesPerThread = passEntryCount / threadsToRun;
            
            while( entriesPerThread < minEntriesPerThread && threadsToRun > 1 )
                entriesPerThread = passEntryCount / --threadsToRun;

            // Only run with as many threads as we have filtered
            if( jobId < threadsToRun )
            {
                const uint32* jobMap   = map; 
                Pairs         jobPairs = pairs;

                jobMap         += entriesPerThread * jobId;
                jobPairs.left  += entriesPerThread * jobId;
                jobPairs.right += entriesPerThread * jobId;

                // Add any trailing entries to the last thread
                // #NOTE: Ensure this is only updated after we get the pairs offset
                uint32 trailingEntries = passEntryCount - entriesPerThread * threadsToRun;
                uint32 lastThreadId    = threadsToRun - 1;
                if( jobId == lastThreadId )
                    entriesPerThread += trailingEntries;

                // Mark entries in 2 steps to ensure the previous thread does NOT
                // write to the same QWord at the same time as the current thread.
                // (May happen when the prev thread writes to the end entries & the 
                //   current thread is writing to its beginning entries)
                const int32 fistStepEntryCount = (int32)( entriesPerThread / 2 );
                int32 i = 0;

                // 1st step
                i = this->MarkStep<table>( i, fistStepEntryCount, lTableMarkedEntries, rTableMarkedEntries, lTableOffset, jobPairs, jobMap );
                this->SyncThreads();

                // 2nd step
                this->MarkStep<table>( i, entriesPerThread, lTableMarkedEntries, rTableMarkedEntries, lTableOffset, jobPairs, jobMap );
            }
            else
            {
                this->SyncThreads();    // Sync for 2nd step
            }

            this->SyncThreads();    // Sync after marking finished
        }


        // Update our position on the pairs table
        bucketEntryCount -= passEntryCount;
        pairBucketOffset += passEntryCount;

        map         += passEntryCount;
        pairs.left  += passEntryCount;
        pairs.right += passEntryCount;

        if( pairBucketOffset < pairBucketEntryCount )
        {
            this->pairBucketOffset = pairBucketOffset;
        }
        else
        {
            // Update our left entry offset by adding the number of entries in the
            // l table bucket index that matches our paid bucket index
            this->lTableOffset += context.bucketCounts[(int)table-1][pairBucket];

            // Move to next pairs bucket
            this->pairBucket ++;
            this->pairBucketOffset = 0;
        }
    }
}

//-----------------------------------------------------------
template<TableId table>
inline int32 MarkJob::MarkStep( int32 i, const int32 entryCount, BitField lTable, const BitField rTable,
                                uint64 lTableOffset, const Pairs& pairs, const uint32* map )
{
    for( ; i < entryCount; i++ )
    {
        if constexpr ( table < TableId::Table7 )
        {
            // #TODO: This map needs to support overflow addresses...
            const uint64 rTableIdx = map[i];
            if( !rTable.Get( rTableIdx ) )
                continue;
        }

        const uint64 left  = lTableOffset + pairs.left [i];
        const uint64 right = left         + pairs.right[i];

        // #TODO Test with atomic sets so that we can write into the
        //       mapped index, and not have to do mapped readings when
        //       reading from the R table here, or in Phase 3.
        lTable.Set( left  );
        lTable.Set( right );
    }

    return i;
}

