#include "DiskPlotPhase3.h"
#include "util/BitField.h"
#include "plotmem/LPGen.h"

// Extra L entries to load per bucket to ensure we
// have cross bucket entries accounted for
#define P3_EXTRA_L_ENTRIES_TO_LOAD BB_DP_CROSS_BUCKET_MAX_ENTRIES

template<TableId rTable, uint32 _numBuckets>
class P3StepOne
{
    static constexpr uint32 _k             = _K;
    static constexpr uint32 _bucketBits    = bblog2( _numBuckets );
    static constexpr uint32 _lpBits        = _k * 2 - _bucketBits;
    static constexpr uint32 _idxBits       = _k + 1;
    static constexpr uint32 _entrySizeBits = _lpBits + _idxBits; // LP, origin index

public:

    //-----------------------------------------------------------
    P3StepOne( DiskPlotContext& context, Fence& readFence, Fence& writeFence )
        : _context    ( context )
        , _ioQueue    ( *context.ioQueue )
        , _threadCount( context.p3ThreadCount )
        , _readFence  ( readFence  )
        , _writeFence ( writeFence )
    {
        _readFence .Reset();
        _writeFence.Reset();

    }

    //-----------------------------------------------------------
    ~P3StepOne() {}

    //-----------------------------------------------------------
    inline Duration GetReadWaitTime() const { return _readWaitTime; }
    inline Duration GetWriteWaitTime() const { return _writeWaitTime; }

    //-----------------------------------------------------------
    uint64 Run()
    {
        DiskPlotContext& context = _context;
        DiskBufferQueue& ioQueue = *context.ioQueue;

        ioQueue.SeekBucket( FileId::LP, 0, SeekOrigin::Begin );
        ioQueue.CommitCommands();

        const TableId lTable           = rTable - 1;
        const uint64  maxBucketEntries = (uint64)DiskPlotInfo<TableId::Table1, _numBuckets>::MaxBucketEntries;
        const size_t  rMarksSize       = RoundUpToNextBoundary( context.entryCounts[(int)rTable] / 8, (int)context.tmp1BlockSize );
        const size_t  writeBufferSize  = RoundUpToNextBoundary( CDiv( maxBucketEntries * _entrySizeBits, 8 ), (int)context.tmp2BlockSize );

        // Allocate buffers
        StackAllocator allocator( context.heapBuffer, context.heapSize );
        
        void* rMarks = allocator.Alloc( rMarksSize, context.tmp1BlockSize );

        DiskPairAndMapReader<_numBuckets> rTableReader( context, context.p3ThreadCount, _readFence, rTable, allocator, false );

        using LReader = SingleFileMapReader<_numBuckets, P3_EXTRA_L_ENTRIES_TO_LOAD, uint32>;
        LReader lTableReader = LReader( FileId::T1, &ioQueue, allocator, maxBucketEntries, context.tmp1BlockSize, context.bucketCounts[(int)TableId::Table1] );

        {
            _lpWriteBuffer[0] = allocator.AllocT<byte>( writeBufferSize, context.tmp2BlockSize );
            _lpWriteBuffer[1] = allocator.AllocT<byte>( writeBufferSize, context.tmp2BlockSize );

            byte* blockBuffers = (byte*)allocator.CAlloc( _numBuckets, context.tmp2BlockSize, context.tmp2BlockSize );
            _lpWriter = BitBucketWriter<_numBuckets>( ioQueue, FileId::LP, blockBuffers );
        }

        Pair*   pairs = allocator.CAlloc<Pair>  ( maxBucketEntries );
        uint64* map   = allocator.CAlloc<uint64>( maxBucketEntries );

        _rPrunedLinePoints = allocator.CAlloc<uint64>( maxBucketEntries );
        _rPrunedMap        = allocator.CAlloc<uint64>( maxBucketEntries );

        auto LoadBucket = [&]( const uint32 bucket ) {

            lTableReader.LoadNextBucket();
            rTableReader.LoadNextBucket();
        };


        // Load initial bucket and the whole marking table
        if( rTable < TableId::Table7 )
            ioQueue.ReadFile( FileId::MARKED_ENTRIES_2 + (FileId)rTable - 1, 0, rMarks, rMarksSize );

        LoadBucket( 0 );

        Log::Line( "Allocated %.2lf / %.2lf MiB", (double)allocator.Size() BtoMB, (double)allocator.Capacity() BtoMB );

        for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
        {
            // Load next bucket in the background
            const bool isLastbucket = bucket + 1 == _numBuckets;

            if( !isLastbucket )
                LoadBucket( bucket + 1 );

            // Wait for and unpack our current bucket
            const uint64  bucketLength = rTableReader.UnpackBucket( bucket, pairs, map );   // This will wait on the read fence
            const uint32* lEntries     = lTableReader.ReadLoadedBucket();
            ASSERT( bucketLength <= maxBucketEntries );

            // Convert to line points
            uint64 prunedEntryCount = ConvertToLinePoints( bucket, bucketLength, lEntries, rMarks, pairs, map );    ASSERT( prunedEntryCount <= bucketLength );

            WriteLinePointsToBuckets( bucket, (int64)prunedEntryCount, _rPrunedLinePoints, _rPrunedMap, (uint64*)pairs, map );

            _prunedEntryCount += prunedEntryCount;
        }

        return _prunedEntryCount;
    }

private:
    //-----------------------------------------------------------
    uint64 ConvertToLinePoints( 
        const uint32 bucket, const int64 bucketLength, const uint32* leftEntries, 
        const void* rightMarkedEntries, const Pair* rightPairs, const uint64* rightMap )
    {
        int64 __prunedEntryCount[BB_DP_MAX_JOBS];
        int64* _prunedEntryCount = __prunedEntryCount;

        AnonMTJob::Run( *_context.threadPool, _threadCount, [=, this]( AnonMTJob* self ) {

            int64 count, offset, end;
            GetThreadOffsets( self, bucketLength, count, offset, end );

            const uint32*  lMap     = leftEntries;
            const Pair*    pairs    = rightPairs;
            const uint64*  rMap     = rightMap;
            const BitField markedEntries( (uint64*)rightMarkedEntries );

            // First, scan our entries in order to prune them
            int64  prunedLength     = 0;
            int64* allPrunedLengths = (int64*)_prunedEntryCount;

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

            // Set our destination offset
            // #NOTE: Not necesarry for T7, but let's avoid code duplication for now.
            int64 dstOffset = 0;

            for( int32 i = 0; i < (int32)self->_jobId; i++ )
                dstOffset += allPrunedLengths[i];

            // Copy pruned entries into a new, contiguous buffer
            // #TODO: check if doing 1 pass per buffer performs better
            Pair*   outPairsStart = (Pair*)(_rPrunedLinePoints + dstOffset);
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

                *outPairs  = pairs[i];
                *mapWriter = mapIdx;

                outPairs++;
                mapWriter++;
            }

            // Now we can convert our pruned pairs to line points
            uint64* outLinePoints = _rPrunedLinePoints + dstOffset;
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
        });

        int64 prunedEntryCount = 0;
        for( int32 i = 0; i < (int32)_threadCount; i++ )
            prunedEntryCount += _prunedEntryCount[i];

        return (uint64)prunedEntryCount;
    }

    //-----------------------------------------------------------
    void WriteLinePointsToBuckets( const uint32 bucket, const int64 entryCount, const uint64* linePoints, const uint64* indices,
                                   uint64* tmpLPs, uint64* tmpIndices )
    {
        using LPJob = AnonPrefixSumJob<uint32>;

        uint32 _totalCounts[_numBuckets]; uint32* totalCounts = _totalCounts;
        uint64 _bitCounts  [_numBuckets]; uint64* bitCounts   = _bitCounts;

        LPJob::Run( *_context.threadPool, _threadCount, [=, this]( LPJob* self ) {

            const uint32 entrySizeBits = _entrySizeBits;
            const uint32 bucketShift   = _lpBits;

            const uint64* srcLinePoints = linePoints;
            const uint64* srcIndices    = indices;

            uint32 counts[_numBuckets];
            uint32 pfxSum[_numBuckets];

            memset( counts, 0, sizeof( counts ) );

            int64 count, offset, end;
            GetThreadOffsets( self, entryCount, count, offset, end );

            /// Count entries per bucket
            for( const uint64* lp = srcLinePoints + offset, *lpEnd = srcLinePoints + end; lp < lpEnd; lp++ )
            {
                const uint64 bucket = (*lp) >> bucketShift; ASSERT( bucket < _numBuckets );
                counts[bucket]++;
            }

            self->CalculatePrefixSum( _numBuckets, counts, pfxSum, (uint32*)totalCounts );

            /// Distribute entries to their respective buckets
            uint64* lpBuckets  = tmpLPs;
            uint64* idxBuckets = tmpIndices;

            for( int64 i = offset; i < end; i++ )
            {
                const uint64 lp       = srcLinePoints[i];
                const uint64 bucket   = lp >> bucketShift;  ASSERT( bucket < _numBuckets );
                const uint32 dstIndex = --pfxSum[bucket];

                ASSERT( dstIndex < entryCount );

                lpBuckets [dstIndex] = lp;
                idxBuckets[dstIndex] = srcIndices[i];
            }

            /// Prepare to write to disk
            auto&   bucketWriter         = _lpWriter;
            uint64* totalBucketBitCounts = (uint64*)bitCounts;

            if( self->IsControlThread() )
            {
                self->LockThreads();

                for( uint32 i = 0; i < _numBuckets; i++ )
                    totalBucketBitCounts[i] = totalCounts[i] * (uint64)entrySizeBits;

                bucketWriter.BeginWriteBuckets( totalBucketBitCounts, GetLPWriteBuffer( bucket ) );
                self->ReleaseThreads();
            }
            else
                self->WaitForRelease();

            /// Bit-pack entries and write to disk
            uint64 bitsWritten = 0;

            for( uint32 i = 0; i < _numBuckets; i++ )
            {
                const uint64 writeOffset = pfxSum[i];
                const uint64 bitOffset   = writeOffset * entrySizeBits - bitsWritten;
                bitsWritten += totalBucketBitCounts[i];

                ASSERT( bitOffset + counts[i] * entrySizeBits <= totalBucketBitCounts[i] );

                BitWriter writer = bucketWriter.GetWriter( i, bitOffset );

                const uint64* lp  = lpBuckets  + writeOffset;
                const uint64* idx = idxBuckets + writeOffset;

                // Compress a couple of entries first, so that we don't get any simultaneaous writes to the same fields
                const int64 firstWrite  = (int64)std::min( 2u, counts[i] );
                const int64 secondWrite = (int64)counts[i] >= 2 ? counts[i] - 2 : 0;

                PackEntries( firstWrite, writer, lp, idx );
                self->SyncThreads();
                PackEntries( secondWrite, writer, lp+2, idx+2 );
            }

            /// Write buckets to disk
            if( self->IsControlThread() )
            {
                self->LockThreads();
                bucketWriter.Submit();
                _ioQueue.SignalFence( _writeFence, bucket );
                _ioQueue.CommitCommands();
                self->ReleaseThreads();
            }
            else
                self->WaitForRelease(); // Don't go out of scope
        });

        for( int32 b = 0; b < (int32)_numBuckets; b++ )
            _lpBucketCounts[b] += _totalCounts[b];
    }

    //-----------------------------------------------------------
    inline void PackEntries( const int64 count, BitWriter& writer, const uint64* lps, const uint64* indices )
    {
        for( int64 i = 0; i < count; i++ )
        {
            writer.Write( lps    [i], _lpBits  );
            writer.Write( indices[i], _idxBits );
        }
    }

    //-----------------------------------------------------------
    byte* GetLPWriteBuffer( const uint32 bucket )
    {
        if( bucket > 1 )
            _writeFence.Wait( bucket - 2, _writeWaitTime );

        return _lpWriteBuffer[bucket & 1];
    }

private:
    DiskPlotContext& _context;
    DiskBufferQueue& _ioQueue;
    uint32           _threadCount;

    Fence&           _readFence;
    Fence&           _writeFence;

    // Work buffers
    // Temporary buffer for storing the pruned pairs/linePoints and map
    uint64*          _rPrunedLinePoints = nullptr;
    uint64*          _rPrunedMap        = nullptr;

    BitBucketWriter<_numBuckets> _lpWriter;
    byte* _lpWriteBuffer[2] = { nullptr };


    Duration _writeWaitTime = Duration::zero();
    Duration _readWaitTime  = Duration::zero();

    uint64   _prunedEntryCount = 0;
    uint32   _lpBucketCounts[_numBuckets] = { 0 };
};


//-----------------------------------------------------------                        
DiskPlotPhase3::DiskPlotPhase3( DiskPlotContext& context )
    : _context( context )
    , _ioQueue( *context.ioQueue )
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

    // Use up any cache for our line points and map
    const size_t cacheSize = ( _context.cacheSize / 3 / _context.numBuckets / _context.tmp2BlockSize ) * _context.numBuckets * _context.tmp2BlockSize;
          byte*  cache     = _context.cache;
    ASSERT( cacheSize / _context.tmp2BlockSize * _context.tmp2BlockSize == cacheSize );

    FileSetOptions opts = FileSetOptions::DirectIO;

    if( _context.cache )
        opts |= FileSetOptions::Cachable;

    FileSetInitData fdata = {
        .cache     = cache,
        .cacheSize = cacheSize
    };

    ioQueue.InitFileSet( FileId::LP      , "lp"      , _context.numBuckets, opts, &fdata );   // LP+origin idx buckets
    fdata.cache = (cache += cacheSize);
    
    ioQueue.InitFileSet( FileId::LP_MAP_0, "lp_map_0", _context.numBuckets, opts, &fdata );   // Reverse map write/read
    fdata.cache = (cache += cacheSize);

    ioQueue.InitFileSet( FileId::LP_MAP_1, "lp_map_1", _context.numBuckets, opts, &fdata );   // Reverse map read/write


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
            // case TableId::Table3: ProcessTable<TableId::Table3, _numBuckets>(); break;
            // case TableId::Table4: ProcessTable<TableId::Table4, _numBuckets>(); break;
            // case TableId::Table5: ProcessTable<TableId::Table5, _numBuckets>(); break;
            // case TableId::Table6: ProcessTable<TableId::Table6, _numBuckets>(); break;
            // case TableId::Table7: ProcessTable<TableId::Table7, _numBuckets>(); break;
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
    uint64 prunedEntryCount;

    // Step 1: Converts pairs to line points whilst prunning the entries,
    //         then writes them to buckets, alog with their source index,
    //         for sorting in the second step.
    {
        P3StepOne<rTable, _numBuckets> stepOne( _context, _readFence, _writeFence );
        prunedEntryCount = stepOne.Run();

        Log::Line( "Table %u now has %llu / %llu ( %.2lf%% ) entries.", 
            rTable, prunedEntryCount, _context.entryCounts[(int)rTable], 
            prunedEntryCount / (double)_context.entryCounts[(int)rTable] * 100 );
    }

    _ioQueue.SignalFence( _stepFence );
    _ioQueue.CommitCommands();
    _stepFence.Wait();

    
    // Step 1: Loads line points & their source indices from buckets,
    //         sorts them on the line points and then writes the line points 
    //         as parks into the plot file. The sorted indices are then written as
    //         a reverse map into their origin buckets. This reverse map serves
    //         as the L table input for the next table.
    {

    }
}
