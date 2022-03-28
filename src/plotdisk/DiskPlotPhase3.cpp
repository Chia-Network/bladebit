#include "DiskPlotPhase3.h"
#include "plotdisk/DiskPairReader.h"
#include "util/BitField.h"

#define P3_EXTRA_L_ENTRIES_TO_LOAD 1024     // Extra L entries to load per bucket to ensure we
                                            // have cross bucket entries accounted for
                        
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
    TableThirdStep <rTable, _numBuckets>();
}

//-----------------------------------------------------------
template<TableId rTable, uint32 _numBuckets>
void DiskPlotPhase3::TableFirstStep()
{
    DiskPlotContext& context   = _context;
    DiskBufferQueue& ioQueue   = *context.ioQueue;
    Fence&           readFence = _readFence;

    const TableId lTable           = rTable - 1;
    const uint64  maxBucketEntries = (uint64)DiskPlotInfo<TableId::Table1, _numBuckets>::MaxBucketEntries;

    // Prepare files
    _readFence.Reset();

    const FileId lTableId = FileId::T1 + (FileId)lTable;

    StackAllocator allocator( context.heapBuffer, context.heapSize );

    // allocator.AllocT<uint32>

    DiskPairAndMapReader<_numBuckets> rTableReader( context, context.p3ThreadCount, readFence, rTable, allocator, false );

    uint32* lTableBuckets[2] = {
        allocator.AllocT<uint32>( maxBucketEntries + P3_EXTRA_L_ENTRIES_TO_LOAD ),
        allocator.AllocT<uint32>( maxBucketEntries + P3_EXTRA_L_ENTRIES_TO_LOAD )
    };

    Pair*   pairs = allocator.CAlloc<Pair>  ( maxBucketEntries );
    uint64* map   = allocator.CAlloc<uint64>( maxBucketEntries );

    auto LoadBucket = [&]( const uint32 bucket ) {

        const uint64 lBucketLength = context.bucketCounts[(int)lTable][bucket] + 
            ( bucket > 0 ? 0 : P3_EXTRA_L_ENTRIES_TO_LOAD );

        const uint32* lEntries = lTableBuckets[bucket & 1];

        // ioQueue.

    
        rTableReader.LoadNextBucket();   // This will call ioQueue.CommitCommands()
        
    };

    LoadBucket( 0 );
    
    for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
    {
        // const size_t stackMarker = allocator.Size();
        
        const uint64  bucketLength = rTableReader.UnpackBucket( bucket, pairs, map );
        const uint32* lEntries     = lTableBuckets[bucket & 1];

        // Convert to line points
        // ConvertToLinePoints( bucketLength, lEntries, pairs, map );

        // allocator.PopToMarker( stackMarker );
    }
}


//-----------------------------------------------------------
template<TableId rTable, uint32 _numBuckets>
void DiskPlotPhase3::TableSecondStep()
{

}

//-----------------------------------------------------------
template<TableId rTable, uint32 _numBuckets>
void DiskPlotPhase3::TableThirdStep()
{

}

//-----------------------------------------------------------
template<TableId rTable>
void DiskPlotPhase3::ConvertToLinePoints( 
    const int64 bucketLength, const uint32* leftEntries, 
    const void* rightMarkedEntries, const Pair* rightPairs, const uint64* rightMap )
{

    int64 _prunedEntryCount[BB_DP_MAX_JOBS];

    AnonMTJob::Run( *_context.threadPool, _context.p3ThreadCount, [=]( AnonMTJob* self ) {

        int64 count, offset, end;
        GetThreadOffsets( self, bucketLength, count, offset, end );

        const uint32   lMap     = leftEntries;
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

            outPairs->left  = pairs.left[i];
            outPairs->right = outPairs->left + pairs.right[i];

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
