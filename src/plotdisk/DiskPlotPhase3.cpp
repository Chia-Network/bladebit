#include "DiskPlotPhase3.h"
#include "util/BitField.h"
#include "plotdisk/BitBucketWriter.h"
#include "plotmem/LPGen.h"
#include "algorithm/RadixSort.h"

#if _DEBUG
    #include "DiskPlotDebug.h"
    #include "jobs/IOJob.h"

    uint64* _linePointRef = nullptr;
    uint64* _indexRef     = nullptr;
    uint64* _lpRefWriter  = nullptr;
    uint64* _idxRefWriter = nullptr;
#endif

// Extra L entries to load per bucket to ensure we
// have cross bucket entries accounted for
#define P3_EXTRA_L_ENTRIES_TO_LOAD BB_DP_CROSS_BUCKET_MAX_ENTRIES

#if _DEBUG
static void ValidateLinePoints( const TableId table, const DiskPlotContext& context, const uint32 bucket, const uint64* linePoints, const uint64 length );

static void ValidateIndices( const TableId table, const DiskPlotContext& context, const uint32 bucket, const uint64* indices, const uint64 length );
#endif

class EntrySort
{
public:
    //-----------------------------------------------------------
    template<uint32 _numBuckets, uint32 entryBitSize, typename TEntry, typename TKey, typename BucketT = uint32>
    inline static void SortEntries( ThreadPool& pool, const uint32 threadCount, const int64 entryCount, 
                                    TEntry* entries, TEntry* tmpEntries, TKey* keys, TKey* tmpKeys )
    {
        ASSERT( entries );
        ASSERT( tmpEntries );
        ASSERT( entryCount > 0 );
        ASSERT( keys );
        ASSERT( tmpKeys );

        using Job = AnonPrefixSumJob<BucketT>;

        Job::Run( pool, threadCount, [=]( Job* self ) {

            const uint32 remainderBits =  CDiv( entryBitSize, 8 ) * 8 - entryBitSize;

            int64 count, offset, endIdx;
            GetThreadOffsets( self, entryCount, count, offset, endIdx );

            constexpr uint   Radix      = 256;
            constexpr uint32 MaxIter    = CDiv( entryBitSize, 8 );
            constexpr int32  iterations = MaxIter - remainderBits / 8;
            constexpr uint32 shiftBase  = 8;

            BucketT counts     [Radix];
            BucketT pfxSum     [Radix];
            BucketT totalCounts[Radix];

            const TEntry* input    = entries;
                  TEntry* output   = tmpEntries;

            const TKey*   keyInput = keys;
                  TKey*   keyOut   = tmpKeys;

            uint32 shift = 0;

            const uint32 lastByteMask = 0xFF >> remainderBits;
                  uint32 masks[8]     = { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
            
            masks[MaxIter-1] = lastByteMask;

            for( int32 iter = 0; iter < iterations ; iter++, shift += shiftBase )
            {
                const uint32 mask = masks[iter];

                // Zero-out the counts
                memset( counts, 0, sizeof( BucketT ) * Radix );

                const TEntry* src    = input    + offset;
                const TEntry* end    = input    + endIdx;
                const TKey*   keySrc = keyInput + offset;
                ASSERT( (int64)(intptr_t)(end - src) == count );

                do {
                    counts[(*src >> shift) & mask]++;
                } while( ++src < end );

                 self->CalculatePrefixSum( Radix, counts, pfxSum, totalCounts );

                src = input + offset;
                for( int64 i = count; i > 0; )
                {
                    const TEntry  value  = src[--i];
                    const uint64  bucket = (value >> shift) & mask;

                    const BucketT dstIdx = --pfxSum[bucket];
                    ASSERT( dstIdx < entryCount );

                    output[dstIdx] = value;
                    keyOut[dstIdx] = keySrc[i];
                }

                std::swap( (TEntry*&)input , output );
                std::swap( (TKey*&)keyInput, keyOut );

                self->SyncThreads();
            }
        });
    }

    //-----------------------------------------------------------
    // template<uint32 remainderBits, typename T, typename TKey, typename TJob, typename BucketT>
    // inline static void EntrySort( PrefixSumJob<TJob, BucketT>* self, const int64 entryCount, const int64 offset, 
    //                               T* entries, T* tmpEntries, TKey* keys, TKey* tmpKeys )
    // {
    //     ASSERT( self );
    //     ASSERT( entries );
    //     ASSERT( tmpEntries );
    //     ASSERT( entryCount > 0 );

    //     constexpr uint Radix = 256;

    //     constexpr uint32 MaxIter    = 4;
    //     constexpr int32  iterations = MaxIter - remainderBits / 8;
    //     constexpr uint32 shiftBase  = 8;

    //     BucketT counts     [Radix];
    //     BucketT pfxSum     [Radix];
    //     BucketT totalCounts[Radix];

    //     uint32 shift = 0;
    //     T* input  = entries;
    //     T* output = tmpEntries;

    //     TKey* keyInput = keys;
    //     TKey* keyTmp   = tmpKeys;

    //     const uint32 lastByteMask   = 0xFF >> remainderBits;
    //           uint32 masks[MaxIter] = { 0xFF, 0xFF, 0xFF, lastByteMask };

    //     for( int32 iter = 0; iter < iterations ; iter++, shift += shiftBase )
    //     {
    //         const uint32 mask = masks[iter];

    //         // Zero-out the counts
    //         memset( counts, 0, sizeof( BucketT ) * Radix );

    //         T*       src    = input + offset;
    //         const T* start  = src;
    //         const T* end    = start + entryCount;
    //         T*       keySrc = keyInput + offset;

    //         do {
    //             counts[(src->ykey >> shift) & mask]++;
    //         } while( ++src < end );

    //         self->CalculatePrefixSum( Radix, counts, pfxSum, totalCounts );

    //         while( --src >= start )
    //         {
    //             const T       value  = *src;
    //             const uint64  bucket = (value.ykey >> shift) & mask;

    //             const BucketT dstIdx = --pfxSum[bucket];
                
    //             output[dstIdx] = value;
    //             keyTmp[dstIdx] = keySrc[i];

    //         }

    //         std::swap( input   , output );
    //         std::swap( keyInput, keyTmp );
    //         self->SyncThreads();
    //     }
    // }
};

template<uint32 _numBuckets>
class MapWriter
{
public:
    static constexpr uint32 _k             = _K;
    static constexpr uint32 BucketBits     = bblog2( _numBuckets );
    static constexpr uint32 AddressBitSize = _k;
    static constexpr uint32 EntryBitSize   = _k - BucketBits + AddressBitSize;  // ( origin address | final address ) ( k-log2(buckets) | 32 )

public:

    //-----------------------------------------------------------
    MapWriter() {}

    //-----------------------------------------------------------
    MapWriter( DiskBufferQueue& ioQueue, const FileId fileId, IAllocator& allocator, const uint64 maxEntries, const size_t blockSize,
               Fence& writeFence, Duration& writeWaitTime )
        : _ioQueue      ( &ioQueue )
        , _bucketWriter ( ioQueue, fileId, (byte*)allocator.CAlloc( _numBuckets+1, blockSize, blockSize ) )
        , _writeFence   ( &writeFence )
        , _writeWaitTime( &writeWaitTime )
    {
        const size_t writeBufferSize = RoundUpToNextBoundary( CDiv( maxEntries * EntryBitSize, 8 ), (int)blockSize );

        _writebuffers[0] = allocator.AllocT<byte>( writeBufferSize, blockSize );
        _writebuffers[1] = allocator.AllocT<byte>( writeBufferSize, blockSize );
    }

    //-----------------------------------------------------------
    void Write( ThreadPool& pool, const uint32 threadCount,
                const uint32 bucket, const int64 entryCount, const uint64 mapOffset, 
                const uint64* map, uint64* outMap, uint64 outMapBucketCounts[_numBuckets+1] )
    {
        // Write the key as a map to the final entry location.
        // We do this by writing into buckets to final sorted (current) location
        // into its original bucket, with the offset in that bucket.
        using Job = AnonPrefixSumJob<uint32>;

        uint32 _totalCounts   [_numBuckets+1]; uint32* totalCounts    = _totalCounts;
        uint64 _totalBitCounts[_numBuckets+1]; uint64* totalBitCounts = _totalBitCounts;

        Job::Run( pool, threadCount, [=]( Job* self ) {
            
            const uint32 bucketBits   = BucketBits;
            const uint32 bucketShift  = _k - bucketBits;
            const uint32 bitSize      = EntryBitSize;
            const uint32 encodeShift  = _k;
            const uint32 numBuckets   = _numBuckets + 1;

            int64 count, offset, end;
            GetThreadOffsets( self, entryCount, count, offset, end );

            uint32 counts[numBuckets] = { 0 };
            uint32 pfxSum[numBuckets];

            const uint64* inIdx    = map + offset;
            const uint64* inIdxEnd = map + end;

            // Count buckets
            do {
                const uint64 b = *inIdx >> bucketShift;
                ASSERT( b < numBuckets );
                counts[b]++;
            } while( ++inIdx < inIdxEnd );

            self->CalculatePrefixSum( numBuckets, counts, pfxSum, totalCounts );

            // Convert map entries from source index to reverse map
            const uint64 tableOffset = mapOffset + (uint64)offset;

            const uint64* reverseMap    = map + offset;
                  uint64* outMapBuckets = outMap; 

            for( int64 i = 0; i < count; i++ )
            {
                const uint64 origin = reverseMap[i];            ASSERT( origin < 0x100000000 + (1ull<<_k) / _numBuckets );
                const uint64 b      = origin >> bucketShift;    ASSERT( b <= _numBuckets );
                const uint32 dstIdx = --pfxSum[b];              ASSERT( (int64)dstIdx < entryCount );

                const uint64 finalIdx = tableOffset + (uint64)i;
                ASSERT( finalIdx < ( 1ull << encodeShift ) );

                outMapBuckets[dstIdx] = (origin << encodeShift) | finalIdx;
            }

#if _DEBUG
            {
                self->SyncThreads();

                const uint32 idxShift      = _k;
                const uint64 finalIdxMask  = ( 1ull << idxShift    ) - 1;
                const uint64 bucketIdxMask = ( 1ull << bucketShift ) - 1;

                for( int64 i = offset; i < end; i++ )
                {
                    const uint64 packedMap = outMapBuckets[i];
                    const uint64 bucketIdx = ( packedMap >> idxShift ) & bucketIdxMask;
                    const uint64 map       = packedMap & finalIdxMask;
// if( i == 12109888 ) BBDebugBreak();
                    ASSERT( bucketIdx < (1ull<<_k) / _numBuckets );
                    ASSERT( map <= 0xFFFFFFFF );
                }

                self->SyncThreads();
            }
#endif

            auto&   bitWriter = _bucketWriter;
            uint64* bitCounts = totalBitCounts;

            if( self->IsControlThread() )
            {
                self->LockThreads();

                // Convert counts to bit sizes
                for( uint32 i = 0; i < numBuckets; i++ )
                    bitCounts[i] = (uint64)totalCounts[i] * bitSize;

                byte* writeBuffer = GetWriteBuffer( bucket );

                // Wait for the buffer to be available first
                if( bucket > 1 )
                    _writeFence->Wait( bucket - 2, *_writeWaitTime );

                bitWriter.BeginWriteBuckets( bitCounts, writeBuffer );

                self->ReleaseThreads();
            }
            else
                self->WaitForRelease();

            // Bit-compress/pack each bucket entries
            uint64 bitsWritten = 0;

            for( uint32 i = 0; i < numBuckets; i++ )
            {
                if( counts[i] < 1 )
                {
                    self->SyncThreads();
                    continue;
                }

                const uint64 writeOffset = pfxSum[i];
                const uint64 bitOffset   = writeOffset * bitSize - bitsWritten;
                bitsWritten += bitCounts[i];

                ASSERT( bitOffset + counts[i] * bitSize <= bitCounts[i] );

                BitWriter writer = bitWriter.GetWriter( i, bitOffset );

                const uint64* mapToWrite    = outMapBuckets + writeOffset; 
                const uint64* mapToWriteEnd = mapToWrite + counts[i]; 

                // Compress a couple of entries first, so that we don't get any simultaneaous writes to the same fields
                const uint64* mapToWriteEndPass1 = mapToWrite + std::min( counts[i], 2u ); 
                ASSERT( counts[i] > 2 );

                while( mapToWrite < mapToWriteEndPass1 )
                    writer.Write( *mapToWrite++, bitSize );

                self->SyncThreads();

                while( mapToWrite < mapToWriteEnd )
                    writer.Write( *mapToWrite++, bitSize );
            }

            // Write to disk
            self->SyncThreads();
            if( self->IsControlThread() )
            {
                bitWriter.Submit();
                _ioQueue->SignalFence( *_writeFence, bucket );
                _ioQueue->CommitCommands();
            }
        });

        for( int32 b = 0; b <= (int32)_numBuckets; b++ )
            outMapBucketCounts[b] += totalCounts[b];
    }

    //-----------------------------------------------------------
    void SubmitFinalBits()
    {
        _bucketWriter.SubmitLeftOvers();
        _ioQueue->SignalFence( *_writeFence, _numBuckets );
        _ioQueue->CommitCommands();
    }

private:
    inline byte* GetWriteBuffer( const uint32 bucket )
    {
        return _writebuffers[bucket & 1];
    }

private:
    DiskBufferQueue*               _ioQueue         = nullptr;
    BitBucketWriter<_numBuckets+1> _bucketWriter;
    byte*                          _writebuffers[2] = { nullptr };
    Fence*                         _writeFence      = nullptr;
    Duration*                      _writeWaitTime   = nullptr;
};

template<TableId rTable, uint32 _numBuckets>
class P3StepOne
{
public:
    // #TODO: Move these to a shared object
    static constexpr uint32 _k             = _K;
    static constexpr uint32 _bucketBits    = bblog2( _numBuckets );
    static constexpr uint32 _lpBits        = _k * 2 - ( _bucketBits + 1 );  // LPs have at most 2*k-1. Because we ommit the bucket bits, we substract those too.
    static constexpr uint32 _idxBits       = _k + 1;
    static constexpr uint32 _entrySizeBits = _lpBits + _idxBits; // LP, origin index

#if _DEBUG
uint64 _bpOffset = 0;
#endif
public:
    //-----------------------------------------------------------
    P3StepOne( DiskPlotContext& context, const FileId mapReadId, Fence& readFence, Fence& writeFence )
        : _context    ( context )
        , _ioQueue    ( *context.ioQueue )
        , _threadCount( context.p3ThreadCount )
        , _mapReadId  ( mapReadId  )
        , _readFence  ( readFence  )
        , _writeFence ( writeFence )
    {
        _readFence .Reset();
        _writeFence.Reset();
    }

    //-----------------------------------------------------------
    inline Duration GetReadWaitTime() const { return _readWaitTime; }
    inline Duration GetWriteWaitTime() const { return _writeWaitTime; }

    //-----------------------------------------------------------
    uint64 Run( const uint64 inLMapBucketCounts[_numBuckets+1], uint64 outLPBucketCounts[_numBuckets+1] )
    {
        DiskPlotContext& context = _context;
        DiskBufferQueue& ioQueue = *context.ioQueue;

        ioQueue.SeekBucket( FileId::LP, 0, SeekOrigin::Begin );
        ioQueue.CommitCommands();

        const TableId lTable           = rTable - 1;
        const uint64  maxBucketEntries = (uint64)DiskPlotInfo<TableId::Table1, _numBuckets>::MaxBucketEntries;
        const size_t  rMarksSize       = RoundUpToNextBoundary( context.entryCounts[(int)rTable] / 8, (int)context.tmp1BlockSize );

        // Allocate buffers
        StackAllocator allocator( context.heapBuffer, context.heapSize );
        
        void* rMarks = allocator.Alloc( rMarksSize, context.tmp1BlockSize );

        DiskPairAndMapReader<_numBuckets> rTableReader( context, context.p3ThreadCount, _readFence, rTable, allocator, false );

        using L1Reader = SingleFileMapReader<_numBuckets, P3_EXTRA_L_ENTRIES_TO_LOAD, uint32>;
        using LNReader = DiskMapReader<uint32, _numBuckets, _k>;


        IP3LMapReader<uint32>* lReader = nullptr;
        
        L1Reader lTable1Reader;
        LNReader lTableNReader;
        uint32*  lTableNEntries = nullptr;
        
        if constexpr ( lTable == TableId::Table1 )
        {
            lTable1Reader = L1Reader( FileId::T1, &ioQueue, allocator, maxBucketEntries, context.tmp1BlockSize, context.bucketCounts[(int)TableId::Table1] ); 
            lReader       = &lTable1Reader;
        }
        else
        {
            lTableNReader  = LNReader( _context, _context.p3ThreadCount, lTable, _mapReadId, allocator, inLMapBucketCounts );
            lTableNEntries = allocator.CAlloc<uint32>( maxBucketEntries + P3_EXTRA_L_ENTRIES_TO_LOAD );
        }

        {
            const size_t perBucketWriteSize   = CDiv( maxBucketEntries * _entrySizeBits / _numBuckets, (int)context.tmp2BlockSize * 8 ) * context.tmp2BlockSize;
            const size_t writeBufferAllocSize = RoundUpToNextBoundary( perBucketWriteSize * _numBuckets, (int)context.tmp2BlockSize );

            _lpWriteBuffer[0] = allocator.AllocT<byte>( writeBufferAllocSize, context.tmp2BlockSize );
            _lpWriteBuffer[1] = allocator.AllocT<byte>( writeBufferAllocSize, context.tmp2BlockSize );

            byte* blockBuffers = (byte*)allocator.CAlloc( _numBuckets, context.tmp2BlockSize, context.tmp2BlockSize );
            _lpWriter = BitBucketWriter<_numBuckets>( ioQueue, FileId::LP, blockBuffers );
        }

        Pair*   pairs = allocator.CAlloc<Pair>  ( maxBucketEntries );
        uint64* map   = allocator.CAlloc<uint64>( maxBucketEntries );

        _rPrunedLinePoints = allocator.CAlloc<uint64>( maxBucketEntries );
        _rPrunedMap        = allocator.CAlloc<uint64>( maxBucketEntries );

        auto GetLLoadCount = [=]( const uint32 bucket ) { 
            
            // Use the original (unpruned) bucket count
            const uint64 lEntryCount = _context.bucketCounts[(int)lTable][bucket];
            const uint64 loadCount   = bucket == 0 ? lEntryCount + P3_EXTRA_L_ENTRIES_TO_LOAD :
                                        bucket == _numBuckets - 1 ? lEntryCount - P3_EXTRA_L_ENTRIES_TO_LOAD :
                                        lEntryCount;

            return loadCount;
        };

        auto LoadBucket = [&]( const uint32 bucket ) {

            if( lTable == TableId::Table1 )
                lReader->LoadNextBucket();
                // lTable1Reader.LoadNextBucket();
            else
                lTableNReader.LoadNextEntries( GetLLoadCount( bucket ) );

            rTableReader.LoadNextBucket();
        };

    #if _DEBUG
        // _linePointRef = bbcvirtallocbounded<uint64>( context.entryCounts[(int)rTable] * sizeof( uint64 ) );
        // _indexRef     = bbcvirtallocbounded<uint64>( context.entryCounts[(int)rTable] * sizeof( uint64 ) );
        // _lpRefWriter  = _linePointRef;
        // _idxRefWriter = _indexRef;
    #endif


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
            const uint64 bucketLength = rTableReader.UnpackBucket( bucket, pairs, map );   // This will wait on the read fence
            
            uint32* lEntries;
            uint64  lEntryCount;

            if constexpr ( lTable == TableId::Table1 )
                // lEntries = lTable1Reader.ReadLoadedBucket();
                lEntries = lReader->ReadLoadedBucket();
            else
            {
                lEntries    = lTableNEntries;
                lEntryCount = GetLLoadCount( bucket );

                const uintptr_t offset = bucket == 0 ? 0 : P3_EXTRA_L_ENTRIES_TO_LOAD;

                lTableNReader.ReadEntries( lEntryCount, lEntries + offset );
                // BBDebugBreak();
            }

            ASSERT( bucketLength <= maxBucketEntries );

            // Convert to line points
            uint64 prunedEntryCount = ConvertToLinePoints( bucket, bucketLength, lEntries, rMarks, pairs, map );    ASSERT( prunedEntryCount <= bucketLength );

            WriteLinePointsToBuckets( bucket, (int64)prunedEntryCount, _rPrunedLinePoints, _rPrunedMap, (uint64*)pairs, map, outLPBucketCounts );

            // #TODO: Remove this after making our T2+ table reader an IP3LMapReader
            if constexpr ( lTable > TableId::Table1 )
            {
                // ASSERT( 0 );
                if( bucket < _numBuckets - 1 )
                    memcpy( lTableNEntries, lTableNEntries + _context.bucketCounts[(int)lTable][bucket], P3_EXTRA_L_ENTRIES_TO_LOAD * sizeof( uint32 ) );
            }

            _prunedEntryCount += prunedEntryCount;

            #if _DEBUG
                _bpOffset += _context.bucketCounts[(int)lTable][bucket];
            #endif
        }

        // Submit trailing bits
        _lpWriter.SubmitLeftOvers();

        // #if _DEBUG
        //     for( uint32 b = 0; b < _numBuckets; b++ )
        //         ASSERT( outLPBucketCounts[b] <= context.bucketCounts[(int)lTable][b] );
        // #endif

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
#if _DEBUG
const uint64 ll = p.left  + _bpOffset;
const uint64 rr = p.right + _bpOffset;
#endif
                    const uint64 x = lTable[p.left ];
                    const uint64 y = lTable[p.right];

                    ASSERT( x || y );
                    outLinePoints[i] = SquareToLinePoint( x, y );

#if _DEBUG
// if( outLinePoints[i] == 41631646582366 ) BBDebugBreak();
// if( outLinePoints[i] == 2664297094 ) BBDebugBreak();
// if( p.left  + _bpOffset == 738199293 && p.right + _bpOffset == 738199423) BBDebugBreak();
// if( p.right + _bpOffset == 738199423 ) BBDebugBreak();
#endif
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
                                   uint64* tmpLPs, uint64* tmpIndices, uint64 outLPBucketCounts[_numBuckets+1] )
    {
        using LPJob = AnonPrefixSumJob<uint32>;

        uint32 __totalCounts[_numBuckets]; uint32* _totalCounts = __totalCounts;
        uint64 _bitCounts   [_numBuckets]; uint64*  bitCounts   = _bitCounts;

        LPJob::Run( *_context.threadPool, _threadCount, [=, this]( LPJob* self ) {

            const uint32 threadCount   = self->JobCount();

            const uint32 entrySizeBits = _entrySizeBits;
            const uint32 bucketShift   = _lpBits;

            const uint64* srcLinePoints = linePoints;
            const uint64* srcIndices    = indices;

            uint32* totalCounts = (uint32*)_totalCounts;
            uint32 counts[_numBuckets];
            uint32 pfxSum[_numBuckets];

            memset( counts, 0, sizeof( counts ) );

            int64 count, offset, end;
            GetThreadOffsets( self, entryCount, count, offset, end );

            /// Count entries per bucket
            for( const uint64* lp = srcLinePoints + offset, *lpEnd = srcLinePoints + end; lp < lpEnd; lp++ )
            {
                const uint64 b = (*lp) >> bucketShift; ASSERT( b < _numBuckets );
                counts[b]++;
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

#if _DEBUG
        // memcpy( _lpRefWriter , lpBuckets, totalCounts[0] * sizeof( uint64 ) );
        // memcpy( _idxRefWriter, idxBuckets, totalCounts[0] * sizeof( uint64 ) );
        // _lpRefWriter  += totalCounts[0];
        // _idxRefWriter += totalCounts[0];
#endif

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
#if _DEBUG
                const uint64 bitReadStart = writer.Position();
#endif

                const uint64* lp  = lpBuckets  + writeOffset;
                const uint64* idx = idxBuckets + writeOffset;

                // If we have too few entries, just have a single thread pack them
                if( totalCounts[i] <= threadCount*3 )
                {
                    if( self->_jobId == 0 && totalCounts[i] > 0 )
                        PackEntries( totalCounts[i], writer, lp, idx, i );

                    self->SyncThreads();
                    continue;
                }

                // Compress a couple of entries first, so that we don't get any simultaneaous writes to the same fields
                const int64 firstWrite  = (int64)std::min( 2u, counts[i] );
                const int64 secondWrite = (int64)counts[i] >= 2 ? counts[i] - 2 : 0;
                ASSERT( counts[i] == 0 || counts[i] > 2 );

                PackEntries( firstWrite, writer, lp, idx, i );
                self->SyncThreads();
                PackEntries( secondWrite, writer, lp+2, idx+2, i );
                
#if _DEBUG
                if( 0 )
                {
                    BitReader reader( (uint64*)writer.Fields(), (uint64)counts[i] * entrySizeBits, bitReadStart );
                    const uint64 mask = (uint64)i << _lpBits;

                    for( int64 j = 0; j < counts[i]; j++ )
                    {
                        const uint64 rlp  = reader.ReadBits64( _lpBits  ) | mask;
                        const uint64 ridx = reader.ReadBits64( _idxBits );

                        ASSERT( rlp  == lp [j] );
                        ASSERT( ridx == idx[j] );
                    }
                }
#endif
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
            outLPBucketCounts[b] += _totalCounts[b];
    }

    //-----------------------------------------------------------
    inline void PackEntries( const int64 count, BitWriter& writer, const uint64* lps, const uint64* indices, const uint32 bucket )
    {
        // TEST:
        BitReader reader( (uint64*)writer.Fields(), writer.Capacity(), writer.Position() );
        const uint64 mask = (uint64)bucket  << _lpBits;
        
        for( int64 i = 0; i < count; i++ )
        {
            writer.Write( lps    [i], _lpBits  );
            writer.Write( indices[i], _idxBits );

            ASSERT( indices[i] < (1ull << _K) + ((1ull << _K) / _numBuckets) );

            // #TEST
            const uint64 rlp  = reader.ReadBits64( _lpBits  ) | mask;
            const uint64 ridx = reader.ReadBits64( _idxBits );
            ASSERT( rlp  == lps    [i] );
            ASSERT( ridx == indices[i] );
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
    FileId           _mapReadId;
    Fence&           _readFence;
    Fence&           _writeFence;
    Duration         _writeWaitTime = Duration::zero();
    Duration         _readWaitTime  = Duration::zero();

    // Temporary buffer for storing the pruned pairs/linePoints and map
    uint64*          _rPrunedLinePoints = nullptr;
    uint64*          _rPrunedMap        = nullptr;

    BitBucketWriter<_numBuckets> _lpWriter;
    byte*            _lpWriteBuffer[2] = { nullptr };

    uint64           _prunedEntryCount = 0;
};

template<TableId rTable, uint32 _numBuckets>
class P3StepTwo
{
public:
    // #TODO: Move these to a shared object
    static constexpr uint32 _k             = _K;
    static constexpr uint32 _bucketBits    = bblog2( _numBuckets );
    static constexpr uint32 _lpBits        = _k * 2 - ( _bucketBits + 1 );  // LPs have at most 2*k-1. Because we ommit the bucket bits, we substract those too.
    static constexpr uint32 _idxBits       = _k + 1;
    static constexpr uint32 _entrySizeBits = _lpBits + _idxBits; // LP, origin index

public:
    //-----------------------------------------------------------
    P3StepTwo( DiskPlotContext& context, Fence& readFence, Fence& writeFence, const FileId readId, const FileId writeId )
        : _context    ( context )
        , _ioQueue    ( *context.ioQueue )
        , _threadCount( context.p3ThreadCount )
        , _readFence  ( readFence  )
        , _writeFence ( writeFence )
        , _readId     ( readId )
        , _writeId    ( writeId )
    {
        _readFence .Reset();
        _writeFence.Reset();
    }

    //-----------------------------------------------------------
    inline Duration GetReadWaitTime() const { return _readWaitTime; }
    inline Duration GetWriteWaitTime() const { return _writeWaitTime; }

    //-----------------------------------------------------------
    void Run( const uint64 inLPBucketCounts[_numBuckets+1], uint64 outLMapBucketCounts[_numBuckets+1] )
    {
        _ioQueue.SeekBucket( FileId::LP, 0, SeekOrigin::Begin );
        _ioQueue.CommitCommands();

        const TableId lTable           = rTable - 1;
        const uint64  maxBucketEntries = (uint64)DiskPlotInfo<TableId::Table1, _numBuckets>::MaxBucketEntries;

        // Allocate buffers and needed structures
        StackAllocator allocator( _context.heapBuffer, _context.heapSize );

        const size_t readBufferSize = RoundUpToNextBoundary( CDiv( maxBucketEntries * _entrySizeBits, 8 ), (int)_context.tmp2BlockSize );
        byte* readBuffers[2] = {
            allocator.AllocT<byte>( readBufferSize, _context.tmp2BlockSize ),
            allocator.AllocT<byte>( readBufferSize, _context.tmp2BlockSize ),
        };

        uint64* linePoints    = allocator.CAlloc<uint64>( maxBucketEntries );
        uint64* indices       = allocator.CAlloc<uint64>( maxBucketEntries );
        uint64* tmpLinePoints = allocator.CAlloc<uint64>( maxBucketEntries );
        uint64* tmpIndices    = allocator.CAlloc<uint64>( maxBucketEntries );

        MapWriter<_numBuckets> mapWriter( _ioQueue, _writeId, allocator, maxBucketEntries, _context.tmp2BlockSize, _writeFence, _writeWaitTime );

        Log::Line( "Step 2 using %.2lf / %.2lf GiB.", (double)allocator.Size() BtoGB, (double)allocator.Capacity() BtoGB );

        // Start processing buckets
        auto LoadBucket = [=]( uint32 bucket ) {
            const size_t readSize = RoundUpToNextBoundary( CDiv( inLPBucketCounts[bucket] * _entrySizeBits, 8 ), (int)_context.tmp2BlockSize );

            _ioQueue.ReadFile( FileId::LP, bucket, readBuffers[bucket & 1], readSize );
            _ioQueue.SignalFence( _readFence, bucket + 1 );
            _ioQueue.CommitCommands();
        };

        LoadBucket( 0 );

        uint64 mapOffset = 0;

        for( uint32 bucket = 0; bucket < _numBuckets; bucket++ )
        {
            const int64 entryCount = (int64)inLPBucketCounts[bucket]; 
            if( entryCount < 1 )
                continue;

            ASSERT( (uint64)entryCount <= maxBucketEntries );
            
            // Load bucket
            if( bucket < _numBuckets - 1 )
                LoadBucket( bucket + 1 );

            _readFence.Wait( bucket + 1, _readWaitTime );

            // Unpack bucket
            const byte* packedEntries = readBuffers[bucket & 1];
            UnpackEntries( bucket, entryCount, packedEntries, linePoints, indices );

#if _DEBUG
            // {
            //     uint64 bucketCounts[_numBuckets] = { 0 };
            //     for( int64 i = 0; i < entryCount; i++ )
            //     {
            //         ASSERT( _linePointRef[i] == linePoints[i] );
            //         ASSERT( _indexRef[i] == indices[i] );
            //         ASSERT( indices[i] < (1ull << _K) + ((1ull << _K) / _numBuckets) );

            //         bucketCounts[linePoints[i] & 0xFF]++;
            //     }
            //     ASSERT( bucketCounts[0] > 0 );
            // }
#endif
            // Sort on LP
            constexpr int32 maxSortIter = (int)CDiv( 64 - _bucketBits, 8 );

            EntrySort::SortEntries<_numBuckets, _lpBits>( *_context.threadPool, _threadCount, entryCount, 
                                                          linePoints, tmpLinePoints, indices, tmpIndices );

            // RadixSort256::SortWithKey<BB_DP_MAX_JOBS>( *_context.threadPool, _threadCount, linePoints, tmpLinePoints, indices, tmpIndices, entryCount );

            uint64* sortedLinePoints  = linePoints;
            uint64* sortedIndices     = indices;
            uint64* scratchLinePoints = tmpLinePoints;
            uint64* scratchIndices    = tmpIndices;

            // If our iteration count is not even, it means the final
            // output of the sort is saved in the tmp buffers.
            if( ( maxSortIter & 1 ) != 0)
            {
                std::swap( sortedLinePoints, scratchLinePoints );
                std::swap( sortedIndices   , scratchIndices    );
            }

#if _DEBUG
            // if( lTable > TableId::Table1 )
            ValidateLinePoints( lTable, _context, bucket, sortedLinePoints, (uint64)entryCount );

            // ValidateIndices( lTable, _context, bucket, sortedIndices, (uint64)entryCount );
#endif

            // Write reverse map to disk
            if( rTable < TableId::Table7 )
                mapWriter.Write( *_context.threadPool, _threadCount, bucket, entryCount, mapOffset, sortedIndices, scratchIndices, outLMapBucketCounts );
            
            // Write LP's as parks into the plot file
            WriteLinePointsToPlot( entryCount, sortedLinePoints );  // #TODO: Need a single-bucket writer so that we save unaligned data

            mapOffset += (uint64)entryCount;
        }

        mapWriter.SubmitFinalBits();
        _writeFence.Wait( _numBuckets );
    }

private:

    //-----------------------------------------------------------
    void UnpackEntries( const uint32 bucket, const int64 entryCount, const byte* packedEntries, uint64* outLinePoints, uint64* outIndices )
    {
        AnonMTJob::Run( *_context.threadPool, _threadCount, [=]( AnonMTJob* self ) {

            int64 count, offset, end;
            GetThreadOffsets( self, entryCount, count, offset, end );

            BitReader reader( (uint64*)packedEntries, _entrySizeBits * (uint64)entryCount, (uint64)offset * _entrySizeBits );

            uint64* linePoints = outLinePoints;
            uint64* indices    = outIndices;
            ASSERT( indices + entryCount )

            const uint64 bucketMask = ((uint64)bucket) << _lpBits;
            for( int64 i = offset; i < end; i++ )
            {
                const uint64 lp  = reader.ReadBits64( _lpBits  ) | bucketMask;
                const uint64 idx = reader.ReadBits64( _idxBits );

                // if( bucket == 0 )
                // {
                //     ASSERT( lp  == _linePointRef[i] );
                //     ASSERT( idx == _indexRef[i] );
                // }
                
                ASSERT( idx < (1ull << _K) + ((1ull << _K) / _numBuckets) );

                linePoints[i] = lp;
                indices   [i] = idx;
            }
        });
    }

    //-----------------------------------------------------------
    void WriteLinePointsToPlot( const int64 entryCount, const uint64* _inLinePoints )
    {
        
    }

private:
    DiskPlotContext& _context;
    DiskBufferQueue& _ioQueue;
    uint32           _threadCount;
    Fence&           _readFence;
    Fence&           _writeFence;
    Duration         _writeWaitTime = Duration::zero();
    Duration         _readWaitTime  = Duration::zero();
    FileId           _readId;
    FileId           _writeId;
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
#if _DEBUG
    if( 0 )
    {
        Log::Line( "Validating Xs" );
        uint32* xRef     = nullptr;
        uint64  refCount = 0;

        FatalIf( !Debug::LoadRefTable( "/mnt/p5510a/reference/t1.x.tmp", xRef, refCount ),
            "Failed to load ref table" );
        ASSERT( refCount == (1ull << _K) );

        // int err;
        // uint32* xs = (uint32*)IOJob::ReadAllBytesDirect( "/mnt/p5510a/disk_tmp/t1_0.tmp", err );
        // FatalIf( !xs, "Failed to rad Xs with error: %d", err );

        uint32* xs = bbcvirtallocbounded<uint32>( 1ull << _K );
        {
            auto& ioQueue = *_context.ioQueue;

            const uint64  maxBucketEntries = (uint64)DiskPlotInfo<TableId::Table1, 256>::MaxBucketEntries;
            // using XReader = SingleFileMapReader<256, P3_EXTRA_L_ENTRIES_TO_LOAD, uint32>;
            StackAllocator allocator( _context.heapBuffer, _context.heapSize );
            // XReader lTable1Reader( FileId::T1, &ioQueue, allocator, maxBucketEntries, _context.tmp1BlockSize, _context.bucketCounts[(int)TableId::Table1] ); 

            BlockReader<uint32> lTableReader( FileId::T1, &ioQueue, maxBucketEntries, allocator, _context.tmp1BlockSize, 0 );
            
            uint32* xWriter = xs;

            Fence readFence;
            lTableReader.LoadEntries( _context.bucketCounts[(int)TableId::Table1][0] );
            ioQueue.SignalFence( readFence, 1 );
            ioQueue.CommitCommands();

            for( uint32 b = 0; b < 256; b++ )
            {
                if( b + 1 < 256 )
                {
                    lTableReader.LoadEntries( _context.bucketCounts[(int)TableId::Table1][b+1] );
                    ioQueue.SignalFence( readFence, b+2 );
                    ioQueue.CommitCommands();
                }

                readFence.Wait( b+1 );

                const uint64  entryCount = _context.bucketCounts[(int)TableId::Table1][b];
                const uint32* loadedXs   = lTableReader.ReadEntries();

                bbmemcpy_t( xWriter, loadedXs, entryCount );
                xWriter += entryCount;
            }
        }


        for( uint64 i = 0; i < refCount; i++ )
        {
            if( xs[i] != xRef[i] )
            {
                if( xs[i+1] == xRef[i] || xs[i] == xRef[i+1] )
                {
                    i++;
                    continue;
                }
                else
                {
                    uint32 xA[3] = { xs[i], xs[i+1], xs[i+2] };
                    uint32 xB[3] = { xRef[i], xRef[i+1], xRef[i+2] };

                    std::sort( xA, &xA[3] );
                    std::sort( xB, &xB[3] );

                    if( memcmp( xA, xB, sizeof( xA ) ) == 0 )
                    {
                        i+=2;
                        continue;
                    }
                }
                ASSERT( 0 );
            }
        }

        bbvirtfree( xRef );
        // bbvirtfree( xs );
        bbvirtfreebounded( xs );
        Log::Line( "All good!" );
        }

        if( 0 )
        {
            const uint32 numBuckets = 256;

            // Test Table 2->3 to ensure all map indices are valid
            StackAllocator allocator( _context.heapBuffer, _context.heapSize );
            DiskMapReader<uint32, numBuckets, 32> lReader( _context, _context.p3ThreadCount, TableId::Table2, FileId::LP_MAP_1, allocator );

            uint32* lEntries = bbcvirtallocbounded<uint32>( 1ull << _K );

            for( uint32 b = 0; b < numBuckets; b++ )
            {
                
            }

        }
#endif
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
    size_t lpCacheSize, mapCacheSize;
    GetCacheSizes( lpCacheSize, mapCacheSize );

    ASSERT( lpCacheSize / _context.tmp2BlockSize * _context.tmp2BlockSize == lpCacheSize );
    ASSERT( mapCacheSize / _context.tmp2BlockSize * _context.tmp2BlockSize == mapCacheSize );

    byte* cache = _context.cache;

    FileSetOptions opts = FileSetOptions::DirectIO;

    if( _context.cache )
        opts |= FileSetOptions::Cachable;

    FileSetInitData fdata = {
        .cache     = cache,
        .cacheSize = lpCacheSize
    };

    ioQueue.InitFileSet( FileId::LP, "lp", _context.numBuckets, opts, &fdata );   // LP+origin idx buckets
    
    fdata.cache     = (cache += lpCacheSize);
    fdata.cacheSize = mapCacheSize;

    ioQueue.InitFileSet( FileId::LP_MAP_0, "lp_map_0", _context.numBuckets+1, opts, &fdata );   // Reverse map write/read
    fdata.cache = (cache += mapCacheSize);
    ioQueue.InitFileSet( FileId::LP_MAP_1, "lp_map_1", _context.numBuckets+1, opts, &fdata );   // Reverse map read/write

    // Copy the initial bucket count for the first L table
    // #TODO: Erase this, not needed for the first table.
    // for( int32 b = 0; b <= _context.numBuckets; b++ )
    //     _lMapPrunedBucketCounts[b] = _context.bucketCounts[(int)TableId::Table1][b];

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
void DiskPlotPhase3::GetCacheSizes( size_t& outCacheSizeLP, size_t& outCacheSizeMap )
{
    switch( _context.numBuckets )
    {
        case 128 : GetCacheSizesForBuckets<128 >( outCacheSizeLP, outCacheSizeMap ); break;
        case 256 : GetCacheSizesForBuckets<256 >( outCacheSizeLP, outCacheSizeMap ); break;
        case 512 : GetCacheSizesForBuckets<512 >( outCacheSizeLP, outCacheSizeMap ); break;
        case 1024: GetCacheSizesForBuckets<1024>( outCacheSizeLP, outCacheSizeMap ); break;
    
    default:
        Fatal( "Invalid bucket count %u.", _context.numBuckets );
        break;
    }
}

//-----------------------------------------------------------
template<uint32 _numBuckets>
void DiskPlotPhase3::GetCacheSizesForBuckets( size_t& outCacheSizeLP, size_t& outCacheSizeMap )
{
    const size_t lpEntrySize    = P3StepOne<TableId::Table2, _numBuckets>::_entrySizeBits;
    const size_t mapEntrySizeX2 = MapWriter<_numBuckets>::EntryBitSize * 2;
    
    static_assert( mapEntrySizeX2 >= lpEntrySize );

    const size_t fullCache = _context.cacheSize;
    const size_t blockSize = _context.tmp2BlockSize;

    double mapRatio = (double)mapEntrySizeX2 / ( mapEntrySizeX2 + lpEntrySize );

    const uint32 mapBuckets = _numBuckets + 1;
    
    outCacheSizeMap = ((size_t)(fullCache * mapRatio) / 2 ) / mapBuckets / blockSize * mapBuckets * blockSize;
    outCacheSizeLP  = ( fullCache - outCacheSizeMap * 2 ) / _numBuckets / blockSize * _numBuckets * blockSize;

    ASSERT( outCacheSizeMap + outCacheSizeLP <= fullCache );
}

//-----------------------------------------------------------
template<uint32 _numBuckets>
void DiskPlotPhase3::RunBuckets()
{
    TableId startTable = TableId::Table2;

#if _DEBUG
    // startTable = TableId::Table3;
    if( ((int)startTable & 1) == 0 )
        std::swap( _mapReadId, _mapWriteId );
#endif

    for( TableId rTable = startTable; rTable < TableId::Table7; rTable++ )
    {
        Log::Line( "Compressing tables %u and %u.", rTable, rTable+1 );
        const auto timer = TimerBegin();
        switch( rTable )
        {
            case TableId::Table2: ProcessTable<TableId::Table2, _numBuckets>(); break;
            case TableId::Table3: ProcessTable<TableId::Table3, _numBuckets>(); break;
            // case TableId::Table4: ProcessTable<TableId::Table4, _numBuckets>(); break;
            // case TableId::Table5: ProcessTable<TableId::Table5, _numBuckets>(); break;
            // case TableId::Table6: ProcessTable<TableId::Table6, _numBuckets>(); break;
            // case TableId::Table7: ProcessTable<TableId::Table7, _numBuckets>(); break;
            default:
                ASSERT( 0 );
                break;
        }

        const double elapsed = TimerEnd( timer );
        Log::Line( "Finished compressing tables %u and %u in %.2lf seconds.", rTable, rTable+1, elapsed );

        std::swap( _mapReadId, _mapWriteId );

        _ioQueue.SeekBucket( _mapReadId , 0, SeekOrigin::Begin );
        _ioQueue.SeekBucket( _mapWriteId, 0, SeekOrigin::Begin );
        _ioQueue.CommitCommands();
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
        memset( _lpPrunedBucketCounts, 0, sizeof( uint64 ) * (_numBuckets+1) );

        P3StepOne<rTable, _numBuckets> stepOne( _context, _mapReadId, _readFence, _writeFence );
        prunedEntryCount = stepOne.Run( _lMapPrunedBucketCounts, _lpPrunedBucketCounts );

        _context.readWaitTime  += stepOne.GetReadWaitTime();
        _context.writeWaitTime += stepOne.GetWriteWaitTime();

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
        memset( _lMapPrunedBucketCounts, 0, sizeof( uint64 ) * (_numBuckets+1) );

        P3StepTwo<rTable, _numBuckets> stepTwo( _context, _readFence, _writeFence, _mapReadId, _mapWriteId );
        stepTwo.Run( _lpPrunedBucketCounts, _lMapPrunedBucketCounts );

        _context.readWaitTime  += stepTwo.GetReadWaitTime();
        _context.writeWaitTime += stepTwo.GetWriteWaitTime();
    }

    _tablePrunedEntryCount[(int)rTable-1] = prunedEntryCount;
}


#if _DEBUG
//-----------------------------------------------------------
void ValidateLinePoints( const TableId table, const DiskPlotContext& context, const uint32 bucket, const uint64* linePoints, const uint64 length )
{
    static TableId _loadedTable   = TableId::_Count;
    static uint64* _refLPs        = nullptr;
    static uint64  _refEntryCount = 0;
    static uint64  _refLPOffset   = 0;

    if( _refLPs == nullptr )
    {
        _refLPs = bbvirtallocbounded<uint64>( sizeof( uint64 ) * context.entryCounts[(int)TableId::Table7] );   // T7 is the only one that won't be pruned.
                                                                                                                // So all other tables should fit here.
    }

    if( table != _loadedTable )
    {
        // Time to load a new table and reset the reader to the beginning
        _loadedTable = table;

        Debug::LoadRefLinePointTable( table, _refLPs, _refEntryCount );
        ASSERT( _refEntryCount <= context.entryCounts[(int)TableId::Table7] );

        _refLPOffset = 0;
    }

    ASSERT( _refLPOffset + length <= _refEntryCount );

    AnonMTJob::Run( *context.threadPool, 1, [=]( AnonMTJob* self ) {

        uint64 count, offset, end;
        GetThreadOffsets( self, length, count, offset, end );

        const uint64* refLPReader = _refLPs + _refLPOffset + offset;
        const uint64* lpReader    = linePoints + offset;

        for( uint64 i = 0; i < count; i++ )
        {
            ASSERT( lpReader[i] == refLPReader[i] );
        }

        _refLPOffset += length;
    });

    if( bucket == context.numBuckets - 1 )
        Log::Line( "LinePoints Validated Successfully!" );
}

//-----------------------------------------------------------
void ValidateIndices( const TableId table, const DiskPlotContext& context, const uint32 bucket, const uint64* indices, const uint64 length )
{
    static TableId _loadedTable    = TableId::_Count;
    static uint32* _refIndicess    = nullptr;
    static uint64  _refEntryCount  = 0;
    static uint64  _refIndexOffset = 0;

    if( _refIndicess == nullptr )
        _refIndicess = bbvirtallocbounded<uint32>( sizeof( uint32 ) * context.entryCounts[(int)TableId::Table7] );
    
     if( table != _loadedTable )
    {
        // Time to load a new table and reset the reader to the beginning
        _loadedTable = table;

        Debug::LoadRefLPIndexTable( table, _refIndicess, _refEntryCount );
        ASSERT( _refEntryCount <= context.entryCounts[(int)TableId::Table7] );

        _refIndexOffset = 0;
    }

    ASSERT( _refIndexOffset + length <= _refEntryCount );

    AnonMTJob::Run( *context.threadPool, 1, [=]( AnonMTJob* self ) {

        uint64 count, offset, end;
        GetThreadOffsets( self, length, count, offset, end );

        const uint32* refIdxReader = _refIndicess + _refIndexOffset + offset;
        const uint64* idxReader    = indices + offset;

        for( uint64 i = 0; i < count; i++ )
        {
            ASSERT( idxReader[i] == refIdxReader[i] );
        }

        _refIndexOffset += length;
    });

    if( bucket == context.numBuckets - 1 )
        Log::Line( "LinePoints Validated Successfully!" );

}
#endif