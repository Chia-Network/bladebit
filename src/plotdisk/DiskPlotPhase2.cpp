#include "DiskPlotPhase2.h"
#include "util/BitField.h"
#include "algorithm/RadixSort.h"
#include "transforms/FxTransform.h"
#include "plotdisk/DiskPlotInfo.h"
#include "util/StackAllocator.h"
#include "DiskPlotInfo.h"
#include "plotdisk/DiskPairReader.h"

// #DEBUG
#include "jobs/IOJob.h"
#include "io/FileStream.h"
#include "DiskPlotDebug.h"

//-----------------------------------------------------------
template<TableId table>
inline void MarkTableEntries( int64 i, const int64 entryCount, BitField lTable, const BitField rTable,
                              uint64 lTableOffset, const Pair* pairs, const uint64* map );


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
    
    Fence readFence;
    Fence bitFieldFence;
    

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
    FileId  lTableFileId  = FileId::MARKED_ENTRIES_6;

    uint64* lMarkingTable = bitFields[0];
    uint64* rMarkingTable = bitFields[1];

    for( TableId table = TableId::Table7; table > TableId::Table2; table = table-1 )
    {
        readFence.Reset( 0 );

        const auto timer = TimerBegin();
        
        const size_t stackMarker = allocator.Size();
        DiskPairAndMapReader<_numBuckets> reader( context, context.p2ThreadCount, readFence, table, allocator, table == TableId::Table7 );

        ASSERT( allocator.Size() < context.heapSize );

        // Log::Line( "Allocated work heap of %.2lf GiB out of %.2lf GiB.", 
        //     (double)(allocator.Size() + markfieldSize*2 ) BtoGB, (double)context.heapSize BtoGB );

        MarkTable<_numBuckets>( table, reader, pairs, map, lMarkingTable, rMarkingTable );

        //
        // #TEST
        //
        // #if 0
        if( 0 )
        {
            // Debug::ValidatePairs<_numBuckets>( _context, TableId::Table7 );

            uint64* readbuffer    = bbcvirtalloc<uint64>( maxEntries*2 );
            Pair*   pairBuf       = bbcvirtalloc<Pair>( maxEntries );
            byte*   rMarkedBuffer = bbcvirtalloc<byte>( maxEntries );
            byte*   lMarkedBuffer = bbcvirtalloc<byte>( maxEntries );

            Pair*   pairRef       = bbcvirtalloc<Pair>( 1ull << _K );
            Pair*   pairRefTmp    = bbcvirtalloc<Pair>( 1ull << _K );

            uint64* rMap          = bbcvirtalloc<uint64>( maxEntries );
            uint64* refMap        = bbcvirtalloc<uint64>( maxEntries );

            // Load from the reader as a reference
            if( 0 )
            {
                Log::Line( "Loading with map reader..." );
                uint64 readCount = 0;
                uint64* mapReader = refMap;
                for( uint32 b = 0; b < _numBuckets; b++ )
                {
                    reader.LoadNextBucket();
                    const uint64 readLength = reader.UnpackBucket( b, pairBuf, mapReader );
                    mapReader += readLength;
                    readCount += readLength;
                }
                ASSERT( readCount ==  context.entryCounts[(int)TableId::Table7] );
            }

            for( TableId rTable = TableId::Table7; rTable > TableId::Table2; rTable = rTable-1 )
            {
                const TableId lTable = rTable-1;

                const uint64 rEntryCount = context.entryCounts[(int)rTable];
                const uint64 lEntryCount = context.entryCounts[(int)lTable];

                // BitField rMarkedEntries( (uint64*)rMarkedBuffer );
                // BitField lMarkedEntries( (uint64*)lMarkedBuffer );

                Log::Line( "Reading R table %u...", rTable+1 );
                {
                    const uint32 savedBits    = bblog2( _numBuckets );
                    const uint32 pairBits     = _K + 1 - savedBits + 9;
                    const size_t blockSize    = queue.BlockSize( FileId::T1 );
                    const size_t pairReadSize = (size_t)CDiv( rEntryCount * pairBits, (int)blockSize*8 ) * blockSize;
                    const FileId rTableId     = FileId::T1 + (FileId)rTable;

                    Fence fence;
                    queue.SeekFile( rTableId, 0, 0, SeekOrigin::Begin );
                    queue.ReadFile( rTableId, 0, readbuffer, pairReadSize );
                    queue.SignalFence( fence );
                    queue.CommitCommands();
                    fence.Wait();

                    AnonMTJob::Run( *_context.threadPool, [=]( AnonMTJob* self ) {

                        uint64 count, offset, end;
                        GetThreadOffsets( self, rEntryCount, count, offset, end );

                        BitReader reader( readbuffer, rEntryCount * pairBits, offset * pairBits );
                        const uint32 lBits  = _K - savedBits + 1;
                        const uint32 rBits  = 9;
                        for( uint64 i = offset; i < end; i++ )
                        {
                            const uint32 left  = (uint32)reader.ReadBits64( lBits );
                            const uint32 right = left +  (uint32)reader.ReadBits64( rBits );
                            pairBuf[i] = { .left = left, .right = right };
                        }
                    });
                }

                // if( rTable < TableId::Table7 )
                {
                    Log::Line( "Reading R map" );
                    const uint32 _k              = _K;
                    const uint32 _savedBits      = bblog2( _numBuckets );
                    const uint32 _mapBits        = _k + 1 + _k - _savedBits;

                    const FileId mapId           = FileId::MAP2 + (FileId)rTable - 1;

                    const uint64 kEntryCount     = ( 1ull << _K );
                    const uint64 bucketLength    = kEntryCount / _numBuckets;

                    // Last, non-overflow bucket
                    const uint64 lastBucketLength = rEntryCount > kEntryCount ? 
                                 bucketLength : bucketLength - ( kEntryCount - rEntryCount );
                    
                    // Overflow bucket
                    const uint64 overflowBucketLength = rEntryCount > kEntryCount ? rEntryCount - kEntryCount : 0;

                    Fence fence;
                    queue.SeekBucket( mapId, 0, SeekOrigin::Begin );
                    queue.CommitCommands();

                    uint64* mapReader = rMap;
                    for( uint32 b = 0; b <= _numBuckets; b++ )
                    {
                        const uint64 length = b < _numBuckets - 1 ? bucketLength :
                                              b < _numBuckets     ? lastBucketLength : overflowBucketLength;

                        if( length < 1 )
                            break;
                        const size_t bucketReadSize = RoundUpToNextBoundary( length * _mapBits, (int)blockSize * 8 ) / 8;
                        
                        queue.ReadFile( mapId, b, readbuffer, bucketReadSize );
                        queue.SignalFence( fence );
                        queue.CommitCommands();
                        fence.Wait();

                        AnonMTJob::Run( *_context.threadPool, [=]( AnonMTJob* self ) {

                            int64 count, offset, end;
                            GetThreadOffsets( self, (int64)length, count, offset, end );
                            ASSERT( count > 0 );

                            BitReader reader( (uint64*)readbuffer, _mapBits * length, offset * _mapBits );
                            
                            const uint32 idxShift     = _k+1;
                            const uint64 finalIdxMask = ( 1ull << idxShift ) - 1;
                            for( int64 i = offset; i < end; i++ )
                            {
                                const uint64 packedMap = reader.ReadBits64( (uint32)_mapBits );
                                const uint64 map       = packedMap & finalIdxMask;
                                const uint32 dstIdx    = (uint32)( packedMap >> idxShift );
                                ASSERT( dstIdx < length );

                                mapReader[dstIdx] = map;
                                // mapReader[dstIdx] = dstIdx;
                            }
                        });
                        // for( uint64 i = 1; i < length; i++ )
                        //     ASSERT( mapReader[i] == mapReader[i-1]+1 );

                        mapReader += length;
                    }

                    // Test against ref
                    AnonMTJob::Run( *_context.threadPool, 1, [=]( AnonMTJob* self ) {
                        
                        uint64 count, offset, end;
                        GetThreadOffsets( self, rEntryCount, count, offset, end );

                        for( uint64 i = offset; i < end; i++ )
                            ASSERT( rMap[i] == refMap[i] );
                    });
                }
 
                // uint64 refEntryCount = 0;
                // {
                //     char path[1024];
                //     sprintf( path, "%sp1.t%d.tmp", BB_DP_DBG_REF_DIR, (int)rTable+1 );
                //     Log::Line( " Loading reference pairs '%s'.", path );

                //     FatalIf( !Debug::LoadRefTable( path, pairRef, refEntryCount ), "Failed to load reference pairs." );

                //     ASSERT( refEntryCount == rEntryCount );
                //     RadixSort256::Sort<BB_DP_MAX_JOBS,uint64,4>( *_context.threadPool, (uint64*)pairRef, (uint64*)pairRefTmp, refEntryCount );
                // }
                // ASSERT( refEntryCount == rEntryCount );

                Log::Line( "Marking entries..." );
                std::atomic<uint64> prunedEntryCount = 0;

                AnonMTJob::Run( *context.threadPool, [&]( AnonMTJob* self ) {

                    const Pair* pairPtr = pairBuf;
                    
                    uint64 lEntryOffset = 0;
                    uint64 rTableOffset = 0;

                    for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
                    {
                        const uint32 rBucketCount = context.ptrTableBucketCounts[(int)rTable][bucket];
                        const uint32 lBucketCount = context.bucketCounts[(int)lTable][bucket];

                        uint32 count, offset, end;
                        GetThreadOffsets( self, rBucketCount, count, offset, end );

                        for( uint e = offset; e < end; e++ )
                        {
                            if( rTable < TableId::Table7 )
                            {
                                const uint64 rIdx = rMap[rTableOffset + e];
                                if( !rMarkedBuffer[rIdx] )
                                    continue;
                            }

                            uint64 l = (uint64)pairPtr[e].left  + lEntryOffset;
                            uint64 r = (uint64)pairPtr[e].right + lEntryOffset;

                            ASSERT( l < lEntryCount );
                            ASSERT( r < lEntryCount );

                            lMarkedBuffer[l] = 1;
                            lMarkedBuffer[r] = 1;
                        }

                        pairPtr += rBucketCount;

                        lEntryOffset += lBucketCount;
                        rTableOffset += rBucketCount;
                    }

                    self->SyncThreads();


                    if( self->IsControlThread() )
                        Log::Line( "Counting entries." );

                    uint64 count, offset, end;
                    GetThreadOffsets( self, lEntryCount, count, offset, end );
                    for( uint64 e = offset; e < end; e++ )
                    {
                        if( lMarkedBuffer[e] )
                            prunedEntryCount++;
                    }
                });

                Log::Line( " %llu/%llu (%.2lf%%)", prunedEntryCount.load(), lEntryCount,
                    ((double)prunedEntryCount.load() / lEntryCount) * 100.0 );
                Log::Line("");

                // Swap marking tables and zero-out the left one.
                std::swap( lMarkedBuffer, rMarkedBuffer );
                memset( lMarkedBuffer, 0, 1ull << _K );
            }
        }
        // #endif

        // Ensure the last table finished writing to the bitfield
        Duration writeWaitTime = Duration::zero();

        if( table < TableId::Table7 )
            bitFieldFence.Wait( writeWaitTime );

        // Submit l marking table for writing
        queue.WriteFile( lTableFileId, 0, lMarkingTable, markfieldSize );
        queue.SignalFence( bitFieldFence );
        queue.CommitCommands();

        // Swap marking tables
        std::swap( lMarkingTable, rMarkingTable );
        lTableFileId = (FileId)( (int)lTableFileId - 1 );

        const double elapsed = TimerEnd( timer );
        Log::Line( "Finished marking table %d in %.2lf seconds.", table, elapsed );
        Log::Line( " IO write wait time: %.2lf seconds.", TicksToSeconds( writeWaitTime ) );
        _context.writeWaitTime += writeWaitTime;

        allocator.PopToMarker( stackMarker );
        ASSERT( allocator.Size() == stackMarker );

        // Log::Line( " Table %u IO Aggregate Wait Time | READ: %.4lf | WRITE: %.4lf | BUFFERS: %.4lf", table,
        //     TicksToSeconds( context.readWaitTime ), TicksToSeconds( context.writeWaitTime ), context.ioQueue->IOBufferWaitTime() );

        // #TEST:
        // if( table < TableId::Table7 )
        // if( 0 )
        {
            Log::Line( "Counting marked entries..." );
            std::atomic<uint64> lTablePrunedEntries = 0;

            AnonMTJob::Run( *_context.threadPool, [&]( AnonMTJob* self ) {
                BitField markedEntries( rMarkingTable );
                
                const uint64 lTableEntries = context.entryCounts[(int)table-1];

                uint64 count, offset, end;
                GetThreadOffsets( self, lTableEntries, count, offset, end );

                uint64 prunedCount = 0;
                for( uint64 e = offset; e < end; ++e )
                {
                    if( markedEntries.Get( e ) )
                        prunedCount++;
                }

                lTablePrunedEntries += prunedCount;
            });

            const uint64 lTableEntries = context.entryCounts[(int)table-1];
            Log::Line( "Table %u entries: %llu/%llu (%.2lf%%)", table,
                       lTablePrunedEntries.load(), lTableEntries, ((double)lTablePrunedEntries.load() / lTableEntries ) * 100.0 );
            Log::Line( "" );
        }
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
void DiskPlotPhase2::MarkTable( const TableId rTable, DiskPairAndMapReader<_numBuckets> reader,
                                Pair* pairs, uint64* map, uint64* lTableMarks, uint64* rTableMarks )
{
    switch( rTable )
    {
        case TableId::Table7: MarkTableBuckets<TableId::Table7, _numBuckets>( reader, pairs, map, lTableMarks, rTableMarks ); break;
        case TableId::Table6: MarkTableBuckets<TableId::Table6, _numBuckets>( reader, pairs, map, lTableMarks, rTableMarks ); break;
        case TableId::Table5: MarkTableBuckets<TableId::Table5, _numBuckets>( reader, pairs, map, lTableMarks, rTableMarks ); break;
        case TableId::Table4: MarkTableBuckets<TableId::Table4, _numBuckets>( reader, pairs, map, lTableMarks, rTableMarks ); break;
        case TableId::Table3: MarkTableBuckets<TableId::Table3, _numBuckets>( reader, pairs, map, lTableMarks, rTableMarks ); break;
    
        default:
            ASSERT( 0 );
            break;
    }
}

//-----------------------------------------------------------
template<TableId table, uint32 _numBuckets>
void DiskPlotPhase2::MarkTableBuckets( DiskPairAndMapReader<_numBuckets> reader, 
                                       Pair* pairs, uint64* map, uint64* lTableMarks, uint64* rTableMarks )
{
    // Load initial bucket
    reader.LoadNextBucket();

    uint64 lTableOffset = 0;

    for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
    {
        reader.LoadNextBucket();
        reader.UnpackBucket( bucket, pairs, map );

        AnonMTJob::Run( *_context.threadPool, _context.p2ThreadCount, [=]( AnonMTJob* self ) { 
            
            BitField lTableMarkedEntries( lTableMarks );
            BitField rTableMarkedEntries( rTableMarks );

            const uint64 bucketEntryCount = _context.ptrTableBucketCounts[(int)table][bucket];

            if( bucket == 0 )
            {
                // Zero-out l marking table
                const uint64 lTableEntryCount = _context.entryCounts[(int)table];
                const size_t bitFieldSize     = RoundUpToNextBoundary( (size_t)lTableEntryCount / 8, 8 );  // Round up to 64-bit boundary

                size_t count, offset, _;
                GetThreadOffsets( self, bitFieldSize, count, offset, _ );

                if( count )
                    memset( ((byte*)lTableMarks) + offset, 0, count );

                self->SyncThreads();
            }

            // Mark entries
            int64 count, offset, _;
            GetThreadOffsets( self, (int64)bucketEntryCount, count, offset, _ );

            // We need to do 2 passes to ensure no 2 threads attempt to write to the same field at the same time
            const int64 firstPassCount = count / 2;
            
            MarkTableEntries<table>( offset, firstPassCount, lTableMarkedEntries, rTableMarkedEntries, lTableOffset, pairs, map );
            self->SyncThreads();
            MarkTableEntries<table>( offset + firstPassCount, count - firstPassCount, lTableMarkedEntries, rTableMarkedEntries, lTableOffset, pairs, map );

        });

        lTableOffset += _context.bucketCounts[(int)table-1][bucket];
    }
}

//-----------------------------------------------------------
template<TableId table>
inline void MarkTableEntries( int64 i, const int64 entryCount, BitField lTable, const BitField rTable,
                              uint64 lTableOffset, const Pair* pairs, const uint64* map )
{
    for( const int64 end = i + entryCount ; i < end; i++ )
    {
        if constexpr ( table < TableId::Table7 )
        {
            const uint64 rTableIdx = map[i];
            if( !rTable.Get( rTableIdx ) )
                continue;
        }

        const Pair& pair = pairs[i];
        
        const uint64 left  = lTableOffset + pair.left;
        const uint64 right = lTableOffset + pair.right;

        // #TODO Test with atomic sets so that we can write into the
        //       mapped index, and not have to do mapped readings when
        //       reading from the R table here, or in Phase 3.
        lTable.Set( left  );
        lTable.Set( right );
    }
}
