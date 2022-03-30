#include "DiskPlotPhase3.h"
#include "util/BitField.h"
#include "plotmem/LPGen.h"

#define P3_EXTRA_L_ENTRIES_TO_LOAD 1024     // Extra L entries to load per bucket to ensure we
                                            // have cross bucket entries accounted for


//-----------------------------------------------------------                        
DiskPlotPhase3::DiskPlotPhase3( DiskPlotContext& context )
    : _context( context )
{}

//-----------------------------------------------------------
DiskPlotPhase3::~DiskPlotPhase3() {}

//-----------------------------------------------------------
void DiskPlotPhase3::Run()
{
    DiskBufferQueue& ioQueue = *_context.ioQueue;

    ioQueue.SeekFile( FileId::T1, 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile( FileId::T2, 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile( FileId::T3, 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile( FileId::T4, 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile( FileId::T5, 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile( FileId::T6, 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile( FileId::T7, 0, 0, SeekOrigin::Begin );
    
    ioQueue.SeekFile( FileId::MAP2, 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile( FileId::MAP3, 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile( FileId::MAP4, 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile( FileId::MAP5, 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile( FileId::MAP6, 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile( FileId::MAP7, 0, 0, SeekOrigin::Begin );

    ioQueue.SeekFile( FileId::MARKED_ENTRIES_2, 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile( FileId::MARKED_ENTRIES_3, 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile( FileId::MARKED_ENTRIES_4, 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile( FileId::MARKED_ENTRIES_5, 0, 0, SeekOrigin::Begin );
    ioQueue.SeekFile( FileId::MARKED_ENTRIES_6, 0, 0, SeekOrigin::Begin );

    ioQueue.CommitCommands();

    switch( _context.numBuckets )
    {
        case 128 : RunBuckets<128 >(); break;
        case 256 : RunBuckets<256 >(); break;
        case 512 : RunBuckets<512 >(); break;
        case 1024: RunBuckets<1024>(); break;
    
        default:
            ASSERT( 0 );
            break;
    }
}

//-----------------------------------------------------------
template<uint32 _numBuckets>
void DiskPlotPhase3::RunBuckets()
{
    for( TableId rTable = TableId::Table2; rTable < TableId::Table7; rTable++ )
    {
        switch( rTable )
        {
            case TableId::Table2: ProcessTable<TableId::Table2, _numBuckets>(); break;
            case TableId::Table3: ProcessTable<TableId::Table3, _numBuckets>(); break;
            case TableId::Table4: ProcessTable<TableId::Table4, _numBuckets>(); break;
            case TableId::Table5: ProcessTable<TableId::Table5, _numBuckets>(); break;
            case TableId::Table6: ProcessTable<TableId::Table6, _numBuckets>(); break;
            case TableId::Table7: ProcessTable<TableId::Table7, _numBuckets>(); break;
            default:
                ASSERT( 0 );
                break;
        }
    }
}

//-----------------------------------------------------------
template<TableId rTable, uint32 _numBuckets>
void DiskPlotPhase3::ProcessTable()
{
    TableFirstStep <rTable, _numBuckets>();
    TableSecondStep<rTable, _numBuckets>();
}

//-----------------------------------------------------------
template<TableId rTable, uint32 _numBuckets>
void DiskPlotPhase3::TableFirstStep()
{
    DiskPlotContext& context = _context;
    DiskBufferQueue& ioQueue = *context.ioQueue;

    const uint32  _k               = _K;
    const uint32  _lMapBits        = _k + _k - bblog2( _numBuckets );    // For tables > 1, the map is k + k - log2( buckets )
    const TableId lTable           = rTable - 1;
    const uint64  maxBucketEntries = (uint64)DiskPlotInfo<TableId::Table1, _numBuckets>::MaxBucketEntries;
    const size_t  rMarksSize       = RoundUpToNextBoundary( context.entryCounts[(int)rTable] / 8, (int)context.tmp1BlockSize );
    const size_t  lBlockSize       = lTable == TableId::Table1 ? context.tmp1BlockSize : context.tmp2BlockSize;

    // Table 1 is loaded differently than the other tables as it is serialized simply as a single file of uint32's
    // L Tables > 1 are serialized as a reverse map lookup.
    // const size_t  lTabble1Size     = RoundUpToNextBoundary( maxBucketEntries * sizeof( uint32 ), (int)context.tmp1BlockSize ) +
    //                                   RoundUpToNextBoundary( P3_EXTRA_L_ENTRIES_TO_LOAD * sizeof( uint32 ), (int)context.tmp1BlockSize );

    // const size_t  lTableNSize      = RoundUpToNextBoundaryT( ((size_t)maxBucketEntries + P3_EXTRA_L_ENTRIES_TO_LOAD) * _lMapBits / 8, context.tmp2BlockSize );
    
    // const size_t  lTableSize       = lTable == TableId::Table1 ? t1TableSize : lTableNSize;

    StackAllocator allocator( context.heapBuffer, context.heapSize );

    void* rMarks = allocator.Alloc( rMarksSize, context.tmp1BlockSize );

    // if( lTable == TableId::Table1 )
    // {
    //     _lMap[0] = allocator.AllocT<uint32>( lTableSize, lBlockSize );
    //     _lMap[1] = allocator.AllocT<uint32>( lTableSize, lBlockSize );
    // }

    DiskPairAndMapReader<_numBuckets> rTableReader( context, context.p3ThreadCount, _readFence, rTable, allocator, false );

    if( lTable == TableId::Table1 )
        _lMap = BlockReader<uint32>( FileId::T1, &ioQueue, maxBucketEntries + P3_EXTRA_L_ENTRIES_TO_LOAD, 
                                      allocator, context.tmp1BlockSize );

    Pair*   pairs = allocator.CAlloc<Pair>  ( maxBucketEntries );
    uint64* map   = allocator.CAlloc<uint64>( maxBucketEntries );

    _rPrunedPairs = allocator.CAlloc<Pair>  ( maxBucketEntries );
    _rPrunedMap   = allocator.CAlloc<uint64>( maxBucketEntries );
    _linePoints   = allocator.CAlloc<uint64>( maxBucketEntries );

    auto LoadBucket = [&]( const uint32 bucket ) {
        
        LoadLBucket( lTable, bucket );
        rTableReader.LoadNextBucket();
    };

    // Load initial bucket and the whole marking table
    _readFence.Reset();
    ioQueue.ReadFile( FileId::MARKED_ENTRIES_2 + (FileId)rTable - 1, 0, rMarks, rMarksSize );
    LoadBucket( 0 );

    Log::Line( "Allocated %.2lf / %.2lf MiB", (double)allocator.Size() BtoMB, (double)allocator.Capacity() BtoMB );

    for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
    {
        const bool isLastbucket = bucket + 1 == _numBuckets;

        // Load next bucket in the background
        if( !isLastbucket )
            LoadBucket( bucket + 1 );

        const uint64  bucketLength = rTableReader.UnpackBucket( bucket, pairs, map );   // This will wait on the read fence
        const uint32* lEntries     = UnpackLBucket( lTable, bucket );

        // Convert to line points
        ConvertToLinePoints<rTable>( bucket, bucketLength, lEntries, rMarks, pairs, map );

        // allocator.PopToMarker( stackMarker );
    }
}

//-----------------------------------------------------------
inline uint64 DiskPlotPhase3::LT1EntryCountToLoad( const uint32 bucket ) const
{
    const uint64 bucketLength = _context.bucketCounts[(int)TableId::Table1][bucket];
          uint64 loadCount    = bucketLength;

    if( bucket == 0 )
        loadCount += P3_EXTRA_L_ENTRIES_TO_LOAD;
    else
        loadCount = std::min( loadCount, _context.entryCounts[(int)TableId::Table1] - _lEntriesLoaded );    // Don't overflow the load

    return loadCount;
}

//-----------------------------------------------------------
void DiskPlotPhase3::LoadLBucket( const TableId table, const uint32 bucket )
{
    if( table == TableId::Table1 )
    {
        const uint64 loadCount = LT1EntryCountToLoad( bucket );
        _lMap.LoadEntries( loadCount );
        _lEntriesLoaded += loadCount;
    }
    else
    {
    }

    _context.ioQueue->CommitCommands();
}

//-----------------------------------------------------------
uint32* DiskPlotPhase3::UnpackLBucket( const TableId table, const uint32 bucket )
{
    if( table == TableId::Table1 )
    {
        return _lMap.ReadEntries();
    }
    else
    {

    }

    return nullptr;
}

//-----------------------------------------------------------
template<TableId rTable, uint32 _numBuckets>
void DiskPlotPhase3::TableSecondStep()
{

}

//-----------------------------------------------------------
template<TableId rTable>
void DiskPlotPhase3::ConvertToLinePoints( 
    const uint32 bucket, const int64 bucketLength, const uint32* leftEntries, 
    const void* rightMarkedEntries, const Pair* rightPairs, const uint64* rightMap )
{

    int64 __prunedEntryCount[BB_DP_MAX_JOBS];
    int64*  _prunedEntryCount = __prunedEntryCount;


    AnonMTJob::Run( *_context.threadPool, _context.p3ThreadCount, [=]( AnonMTJob* self ) {

        int64 count, offset, end;
        GetThreadOffsets( self, bucketLength, count, offset, end );

        const uint32*  lMap     = leftEntries;
        const Pair*    pairs    = rightPairs;
        const uint64*  rMap     = rightMap;
        const BitField markedEntries( (uint64*)rightMarkedEntries );

        // First, scan our entries in order to prune them
        int64  prunedLength     = 0;
        int64* allPrunedLengths = _prunedEntryCount;

        if constexpr ( rTable < TableId::Table7 )
        {
            for( int64 i = offset; i < end; i++)
            {
                if( markedEntries.Get( rMap[i] ) )
                    prunedLength ++;
            }
        }
        else
        {
            prunedLength = bucketLength;
        }

        allPrunedLengths[self->_jobId] = prunedLength;
        self->SyncThreads();

        // #NOTE: Not necesarry for T7, but let's avoid code duplication for now.
        // Set our destination offset
        int64 dstOffset = 0;

        for( int32 i = 0; i < (int32)self->_jobId; i++ )
            dstOffset += allPrunedLengths[i];

        // Copy pruned entries into new buffer and expend R pointers to absolute address
        // #TODO: check if doing 1 pass per buffer performs better

        Pair*   outPairsStart = (Pair*)(_linePoints + dstOffset);
        Pair*   outPairs      = outPairsStart;
        uint64* outRMap       = _rPrunedMap + dstOffset;
        uint64* mapWriter     = outRMap;

        for( int64 i = offset; i < end; i++ )
        {
            const uint32 mapIdx = rMap[i];

            if constexpr ( rTable < TableId::Table7 )
            {
                if( !markedEntries.Get( mapIdx ) )
                    continue;
            }

            outPairs->left  = pairs[i].left;
            outPairs->right = outPairs->left + pairs[i].right;

            *mapWriter      = mapIdx;

            outPairs++;
            mapWriter++;
        }

        // Now we can convert our pruned pairs to line points
        uint64* outLinePoints = _linePoints + dstOffset;
        {
            const uint32* lTable = lMap;
            
            for( int64 i = 0; i < prunedLength; i++ )
            {
                Pair p = outPairsStart[i];
                const uint64 x = lTable[p.left ];
                const uint64 y = lTable[p.right];

                outLinePoints[i] = SquareToLinePoint( x, y );
            }

            // const uint64* lpEnd = outPairsStart + prunedLength;
            // do
            // {
            //     Pair p = *((Pair*)outPairsStart);
            //     const uint64 x = lTable[p.left ];
            //     const uint64 y = lTable[p.right];
                
            //     *outLinePoints = SquareToLinePoint( x, y );

            // } while( ++outLinePoints < lpEnd );
        }

        // DistributeToBuckets( prunedLength, outLinePoints, outRMap );
        
    });
}

// //-----------------------------------------------------------
// void DiskPlotPhase3::WriteLinePointsToBuckets( 
//     LPWriteJob* self, const int64 entryCount, const uint64* linePoints, const uint32* key )
// {
//     uint32 totalCounts[LP_BUCKET_COUNT];
//     uint32 counts     [LP_BUCKET_COUNT];
//     uint32 pfxSum     [LP_BUCKET_COUNT];

//     memset( counts, 0, sizeof( counts ) );

//     int64 count, offset, end;
//     GetThreadOffsets( self, entryCount, count, offset, end );

//     // Count entries per bucket
//     for( const uint64* lp = linePoints + offset, *lpEnd = lp + end; lp < lpEnd; lp++ )
//     {
//         const uint64 bucket = (*lp) >> 56; ASSERT( bucket < entryCount );
//         counts[bucket]++;
//     }
    
//     self->CalculatePrefixSum( LP_BUCKET_COUNT, counts, pfxSum, totalCounts );

//     uint64* lpOutBuffer  = nullptr;
//     uint32* keyOutBuffer = nullptr;

//     // Grab write buffers for distribution
//     if( self->IsControlThread() )
//     {
//         self->LockThreads();

//         uint64 bitCounts[LP_BUCKET_COUNT];

//         for( uint32 i = 0; i < LP_BUCKET_COUNT; i++ )
//             bitCounts[i] = 8ull * totalCounts[i];

//         _lpWriter.BeginWriteBuckets( bitCounts );

//         self->ReleaseThreads();
//     }
//     else
//         self->WaitForRelease();

//     // Distribute entries to their respective buckets
//     for( int64 i = offset; i < end; i++ )
//     {
//         const uint64 lp       = linePoints[i];
//         const uint64 bucket   = lp >> 56;           ASSERT( bucket < BB_DPP3_LP_BUCKET_COUNT );
//         const uint32 dstIndex = --pfxSum[bucket];

//         ASSERT( dstIndex < entryCount );

//         // lpOutBuffer [dstIndex] = lp;
//         // keyOutBuffer[dstIndex] = key[i];
//     }

//     if( self->IsControlThread() )
//     {
//         DiskBufferQueue& ioQueue = *context->ioQueue;

//         uint32* lpSizes  = (uint32*)ioQueue.GetBuffer( BB_DPP3_LP_BUCKET_COUNT * sizeof( uint32 ) );
//         uint32* keySizes = (uint32*)ioQueue.GetBuffer( BB_DPP3_LP_BUCKET_COUNT * sizeof( uint32 ) );

//         const uint32* bucketCounts = this->bucketCounts;

//         for( int64 i = 0; i < (int)BB_DPP3_LP_BUCKET_COUNT; i++ )
//             lpSizes[i] = bucketCounts[i] * sizeof( uint64 );

//         for( int64 i = 0; i < (int)BB_DPP3_LP_BUCKET_COUNT; i++ )
//             keySizes[i] = bucketCounts[i] * sizeof( uint32 );

//         const FileId lpFileId   = TableIdToLinePointFileId   ( this->rTable );
//         const FileId lpKeyFilId = TableIdToLinePointKeyFileId( this->rTable );

//         // Wait for all threads to finish writing
//         self->LockThreads();

//         ioQueue.WriteBuckets( lpFileId, lpOutBuffer, lpSizes );
//         ioQueue.ReleaseBuffer( lpOutBuffer );
//         ioQueue.ReleaseBuffer( lpSizes );

//         ioQueue.WriteBuckets( lpKeyFilId, keyOutBuffer, keySizes );
//         ioQueue.ReleaseBuffer( keyOutBuffer );
//         ioQueue.ReleaseBuffer( keySizes );

//         ioQueue.CommitCommands();

//         self->ReleaseThreads();
//     }
//     else
//         self->WaitForRelease();

//     // #NOTE: If we move the write from here, we still need to sync the 
//     //        threads before existing to ensure the counts[] buffer
//     //        doesn't go out of scope.
// }

