#include "GreenReaper.h"
#include "threading/ThreadPool.h"
#include "threading/GenJob.h"
#include "plotting/Tables.h"
#include "plotting/PlotTools.h"
#include "plotmem/LPGen.h"
#include "ChiaConsts.h"
#include "pos/chacha8.h"
#include "b3/blake3.h"
#include "algorithm/RadixSort.h"
#include "util/jobs/SortKeyJob.h"

// Determined from the average match count at compression level 11, which was 0.288% of the bucket count * 2.
// We round up to a reasonable percentage of 0.5%.
// #NOTE: This must be modified for lower compression levels.
static constexpr double GR_MAX_MATCHES_MULTIPLIER = 0.005;
static constexpr uint32 GR_MAX_BUCKETS            = 32;
static constexpr uint64 GR_MIN_TABLE_PAIRS        = 1024;

inline static uint64 GetEntriesPerBucketForCompressionLevel( const uint32 k, const uint32 cLevel )
{
    const uint32 entryBits        = 17u - cLevel;
    const uint32 bucketBits       = k - entryBits;
    const uint64 bucketEntryCount = 1ull << bucketBits;

    return bucketEntryCount;
}

inline static uint64 GetMaxTablePairsForCompressionLevel( const uint32 k, const uint32 cLevel )
{
    return (uint64)(GetEntriesPerBucketForCompressionLevel( k, cLevel ) * GR_MAX_MATCHES_MULTIPLIER) * (uint64)GR_MAX_BUCKETS;
}

// #TODO: Add GPU decompression.
//          Perhaps try the sort directly to kBC group method again, but with GPU?
//          might prove worth it in that case (did not work that well with CPU).

// Internal types
struct ProofContext
{
    uint32    leftLength;
    uint32    rightLength;

    uint64*   yLeft;        // L table Y
    uint64*   yRight;       // R table Y
    K32Meta4* metaLeft;     // L table metadata
    K32Meta4* metaRight;    // R table metadata

    uint64*   proof;
};

struct ProofTable
{
    Pair*  _pairs    = nullptr;
    uint64 _capacity = 0;
    uint64 _length   = 0;

    struct {
        uint32 count;
        uint32 offset;
    } _groups[16] = {};

    inline Span<Pair> GetGroupPairs( uint32 groupIdx ) const
    {
        ASSERT( groupIdx < 16 );
        
        const uint32 offset = _groups[groupIdx].offset;
        const uint32 count  = _groups[groupIdx].count;
        ASSERT( offset + count <= _length );

        return MakeSpan( _pairs + offset,  count );
    }

    inline Span<Pair> GetUsedTablePairs() const
    {
        return MakeSpan( _pairs, _length );
    }

    inline Span<Pair> GetFreeTablePairs() const
    {
        return MakeSpan( _pairs+_length, _capacity - _length );
    }

    inline Pair& PushGroupPair( uint32 groupIdx )
    {
        ASSERT( _length < _capacity );

        _groups[groupIdx].count++;

        return _pairs[_length++];
    }

    inline void BeginGroup( uint32 groupIdx )
    {
        ASSERT( groupIdx < 16 );
        _groups[groupIdx].count = 0;

        if( groupIdx > 0 )
            _groups[groupIdx].offset = _groups[groupIdx-1].offset + _groups[groupIdx-1].count;
        else
            _length = 0;
    }

    inline void AddGroupPairs( uint32 groupIdx, uint32 pairCount )
    {
        ASSERT( groupIdx < 16 );

        _groups[groupIdx].count += pairCount;
        _length += pairCount;
    }

    template<TableId lTable>
    void GetLTableGroupEntries( GreenReaperContext& cx, const uint32 group, Span<uint64>& outY, Span<typename K32MetaType<lTable>::Out>& outMeta );

    // inline void FinishGroup( uint32 groupIdx )
    // {
    //     ASSERT( groupIdx < 16 );

    //     if( groupIdx < 15 )
    //         _groups[groupIdx+1].offset = _groups[groupIdx].offset + _groups[groupIdx].count;
    // }
};

struct GreenReaperContext
{
    GreenReaperConfig config;

    ThreadPool*    pool = nullptr;

    uint64         maxEntriesPerBucket;

    Span<uint64>   yBufferF1;   // F1 tmp
    Span<uint64>   yBuffer;     // F1/In/Out
    Span<uint64>   yBufferTmp;  // In/Out

    Span<uint32>   xBuffer;
    Span<uint32>   xBufferTmp;
    
    Span<uint32>   sortKey; // #TODO: Remove this, can use x for it
    
    Span<K32Meta4> metaBuffer;
    Span<K32Meta4> metaBufferTmp;
    Span<Pair>     pairs;
    Span<Pair>     pairsTmp;
    Span<uint32>   groupsBoundaries;    // BC group boundaries
    
    ProofTable     tables[7]  = {};

    ProofContext   proofContext;
};

enum class ForwardPropResult
{
    Failed   = 0,
    Success  = 1,
    Continue = 2
};


template<TableId lTable>
inline void ProofTable::GetLTableGroupEntries( GreenReaperContext& cx, const uint32 group, Span<uint64>& outY, Span<typename K32MetaType<lTable>::Out>& outMeta )
{
    using TMeta = typename K32MetaType<lTable>::Out;

    ASSERT( group < 16 );
    outY    = MakeSpan( cx.proofContext.yLeft + _groups[group].offset, _groups[group].count );
    outMeta = MakeSpan( (TMeta*)cx.proofContext.metaLeft + _groups[group].offset, _groups[group].count );
}

/// Internal functions
static void LookupProof();
static void FreeBucketBuffers( GreenReaperContext& cx );
static bool ReserveBucketBuffers( GreenReaperContext& cx, uint32 k, uint32 compressionLevel );

static void GenerateF1( GreenReaperContext& cx, const byte plotId[32], const uint64 bucketEntryCount, const uint32 x0, const uint32 x1 );
static Span<Pair> Match( GreenReaperContext& cx, const Span<uint64> yEntries, Span<Pair> outPairs, const uint32 pairOffset  );

template<TableId rTable>
static void SortTableAndFlipBuffers( GreenReaperContext& cx );

template<TableId rTable, typename TMetaIn, typename TMetaOut>
static void GenerateFxForPairs( GreenReaperContext& cx, const Span<Pair> pairs, const Span<uint64> yIn, const Span<TMetaIn> metaIn, Span<uint64> yOut, Span<TMetaOut> outMeta );

template<TableId rTable, typename TMetaIn, typename TMetaOut>
static void GenerateFx( const Span<Pair> pairs, const Span<uint64> yIn, const Span<TMetaIn> metaIn, Span<uint64> yOut, Span<TMetaOut> outMeta );

static bool ForwardPropTables( GreenReaperContext& cx );
// static BackPtr UnpackLinePointCompressedProof( GreenReaperContext& cx, uint32 bitsPerEntry, uint32 proofLinePoint );
static void BacktraceProof( GreenReaperContext& cx, const TableId tableStart, uint64 proof[GR_POST_PROOF_X_COUNT] );

///
/// Public API
///

//-----------------------------------------------------------
GreenReaperContext* grCreateContext( GreenReaperConfig* config )
{
    auto* context = new GreenReaperContext();
    if( context == nullptr )
        return nullptr;

    if( config )
        context->config = *config;
    else
    {
        context->config.threadCount = std::min( 2u, SysHost::GetLogicalCPUCount() );
    }

    context->pool = new ThreadPool( context->config.threadCount );
    if( !context->pool )
    {
        grDestroyContext( context );
        return nullptr;
    }

    return context;
}

//-----------------------------------------------------------
void grDestroyContext( GreenReaperContext* context )
{
    if( context == nullptr )
        return;

    FreeBucketBuffers( *context );

    if( context->pool )
        delete context->pool;

    delete context;
}

//-----------------------------------------------------------
// int32_t bbHarvesterFetchProofForChallenge( GreenReaperContext* context, const uint8_t* challenge, uint8_t* outFullProof[64], uint32_t maxProofs )
GRProofResult grFetchProofForChallenge( GreenReaperContext* cx, GRCompressedProofRequest* req )
{
    if( !cx || !req || !req->plotId )
        return GRProofResult_Failed;

    // #TODO: Validate entryBitCount

    // Always make sure this has been done
    LoadLTargets();

    const uint32 k                = 32;
    const uint64 entriesPerBucket = GetEntriesPerBucketForCompressionLevel( k, req->compressionLevel );
    
    // Ensure our buffers have enough for the specified entry bit count
    if( !ReserveBucketBuffers( *cx, k, req->compressionLevel ) )
        return GRProofResult_OutOfMemory;

    const uint32 numGroups = GR_POST_PROOF_CMP_X_COUNT;
    
    Span<uint64>   yOut     = cx->yBufferTmp;
    Span<K32Meta2> metaOut  = cx->metaBufferTmp.template As<K32Meta2>();
    Span<Pair>     outPairs = cx->pairs;
    
    for( uint32 i = 0; i < numGroups; i++ )
    {
        const uint32  xLinePoint = (uint32)req->compressedProof[i];
        const BackPtr xs         = LinePointToSquare64( xLinePoint );

        auto yEntries = cx->yBuffer;
        auto xEntries = cx->xBuffer;

        if( xs.x == 0 || xs.y == 0 )
        {
            ASSERT( 0 );
            // ...
        }

        GenerateF1( *cx, req->plotId, entriesPerBucket, (uint32)xs.x, (uint32)xs.y );

        // Perform matches
        auto& table = cx->tables[1];
        const uint32 groupIndex = i / 2;
        if( (i & 1) == 0 )
            table.BeginGroup( groupIndex );

        const Span<Pair> pairs = Match( *cx, yEntries, outPairs, 0 );

        // Expect at least one match
        if( pairs.Length() < 1 )
            return GRProofResult_Failed;

        table.AddGroupPairs( groupIndex, (uint32)pairs.Length() );

        // Perform fx for table2 pairs
        GenerateFxForPairs<TableId::Table2>( *cx, pairs, yEntries, xEntries, yOut, metaOut );

        // Inline x's into the pairs
        {
            const uint32 threadCount = std::min( cx->config.threadCount, (uint32)pairs.Length() );

            AnonMTJob::Run( *cx->pool, threadCount, [=]( AnonMTJob* self ){
                
                const Span<uint32> xs          = xEntries;
                      Span<Pair>   threadPairs = GetThreadOffsets( self, pairs );

                for( uint64 i = 0; i < threadPairs.Length(); i++ )
                {
                    Pair p = threadPairs[i];
                    p.left  = xs[p.left ];
                    p.right = xs[p.right];

                    threadPairs[i] = p;
                }
            });
        }
        
        outPairs = outPairs.Slice( pairs.Length() );
        yOut     = yOut    .Slice( pairs.Length() );
        metaOut  = metaOut .Slice( pairs.Length() );
    }


    // If all groups have a single match, then we have our full proof
    bool hasProof = true;
    
    for( uint32 i = 0; i < numGroups / 2; i++ )
    {
        if( cx->tables[1]._length != 2 )
        {
            hasProof = false;
            break;
        }
    }

    if( hasProof )
    {
        // #TODO: Return the proof
        Fatal( "Unimplemented" );
    }


    /// No proof found yet, continue on to the rest of forward propagation

       
    // Continue forward propagation to the next table
    const uint32 table2Length = (uint32)cx->tables[1]._length;

    cx->proofContext.leftLength  = table2Length;
    // cx->proofContext.rightLength = cx->tables[2]._capacity;

    cx->proofContext.yLeft     = cx->yBuffer.Ptr();
    cx->proofContext.metaLeft  = cx->metaBuffer.Ptr();
    cx->proofContext.yRight    = cx->yBufferTmp.Ptr();
    cx->proofContext.metaRight = cx->metaBufferTmp.Ptr();
    cx->proofContext.proof     = req->fullProof;

    SortTableAndFlipBuffers<TableId::Table2>( *cx );

    return ForwardPropTables( *cx ) ? GRProofResult_OK : GRProofResult_Failed;
}

// -----------------------------------------------------------
void BacktraceProof( GreenReaperContext& cx, const TableId tableStart, uint64 proof[GR_POST_PROOF_X_COUNT] )
{
    Pair _backtrace[2][64] = {};

    Pair* backTraceIn  = _backtrace[0];
    Pair* backTraceOut = _backtrace[1];

    // #TODO: If in table 2, get the x's directly

    // Fill initial back-trace
    {
        ProofTable& table = cx.tables[(int)tableStart];
        bbmemcpy_t( backTraceIn, table._pairs, table._length );

        // const uint32 groupCount = 32 >> (int)tableStart;
            

        // for( uint32 i = 0; i < groupCount; i++ )
        // {
        //     ASSERT( table._length == 2 );

        //     Pair p0 = table._pairs[0];
        //     Pair p1 = table._pairs[1];

        //     const uint32 idx = i*4;

        //     backTraceIn[idx+0] = e0.parentL;
        //     backTraceIn[idx+1] = e0.parentR;
        //     backTraceIn[idx+2] = e1.parentL;
        //     backTraceIn[idx+3] = e1.parentR;
        // }
    }

    for( TableId table = tableStart; table > TableId::Table2; table-- )
    {
        const uint32 entryCount = (32 >> (int)table)*2;

        ProofTable& lTable = cx.tables[(int)table-1];

        for( uint32 i = 0; i < entryCount; i++ )
        {
            const Pair p = backTraceIn[i];
            
            const uint32 idx = i * 2;
            backTraceOut[idx+0] = lTable._pairs[p.left ];
            backTraceOut[idx+1] = lTable._pairs[p.right];
        }

        std::swap( backTraceIn, backTraceOut );
    }

    for( uint32 i = 0; i < GR_POST_PROOF_CMP_X_COUNT; i++ )
    {
        const uint32 idx = i * 2;
        proof[idx+0] = backTraceIn[i].left;
        proof[idx+1] = backTraceIn[i].right;
    }

#if _DEBUG
    Log::Line( "" );
    Log::Line( "Recovered Proof:" );
    for( uint32 i = 0; i < GR_POST_PROOF_X_COUNT; i++ )
        Log::WriteLine( "[%2u]: %-10u ( 0x%08x )", i, (uint32)proof[i], (uint32)proof[i] );
#endif
}

//-----------------------------------------------------------
bool ReserveBucketBuffers( GreenReaperContext& cx, uint32 k, uint32 compressionLevel )
{
    // #TODO: Support other K values
    ASSERT( k == 32 );

    const uint64 entriesPerBucket = GetEntriesPerBucketForCompressionLevel( k, compressionLevel );

    if( cx.maxEntriesPerBucket < entriesPerBucket )
    {
        cx.maxEntriesPerBucket = 0;
        FreeBucketBuffers( cx );

        const size_t allocCount = (size_t)entriesPerBucket * 2;

        // The pair requirements ought to be much less as the number of matches we get per group is not as high.
        uint64 maxPairsPerTable = std::max( (uint64)GR_MIN_TABLE_PAIRS, GetMaxTablePairsForCompressionLevel( k, compressionLevel ) );

        cx.yBufferF1        = bb_try_virt_calloc_bounded_span<uint64>  ( allocCount );
        cx.yBuffer          = bb_try_virt_calloc_bounded_span<uint64>  ( allocCount );
        cx.yBufferTmp       = bb_try_virt_calloc_bounded_span<uint64>  ( allocCount );
        cx.xBuffer          = bb_try_virt_calloc_bounded_span<uint32>  ( allocCount );
        cx.xBufferTmp       = bb_try_virt_calloc_bounded_span<uint32>  ( allocCount );
        cx.sortKey          = bb_try_virt_calloc_bounded_span<uint32>  ( maxPairsPerTable );
        cx.metaBuffer       = bb_try_virt_calloc_bounded_span<K32Meta4>( maxPairsPerTable );
        cx.metaBufferTmp    = bb_try_virt_calloc_bounded_span<K32Meta4>( maxPairsPerTable );
        cx.pairs            = bb_try_virt_calloc_bounded_span<Pair>    ( maxPairsPerTable );
        cx.pairsTmp         = bb_try_virt_calloc_bounded_span<Pair>    ( maxPairsPerTable );
        cx.groupsBoundaries = bb_try_virt_calloc_bounded_span<uint32>  ( allocCount );

        // Allocate proof tables
        // Table 1 needs no groups, as we write to the R table's merged group, always
        for( uint32 i = 1; i < 7; i++ )
        {
            cx.tables[i] = {};
            cx.tables[i]._pairs    = bb_try_virt_calloc_bounded<Pair>( maxPairsPerTable );
            cx.tables[i]._capacity = maxPairsPerTable;

            if( !cx.tables[i]._pairs )
            {
                FreeBucketBuffers( cx );
                return false;
            }

            // Reduce the match count for each subsequent table by nearly half
            maxPairsPerTable = std::max<uint64>( (uint64)(maxPairsPerTable * 0.6), GR_MIN_TABLE_PAIRS );
        }

        if( !cx.yBufferF1.Ptr()     ||
            !cx.yBuffer.Ptr()       || 
            !cx.yBufferTmp.Ptr()    || 
            !cx.xBuffer.Ptr()       || 
            !cx.xBufferTmp.Ptr()    || 
            !cx.sortKey.Ptr()       || 
            !cx.metaBuffer.Ptr()    || 
            !cx.metaBufferTmp.Ptr() || 
            !cx.pairs.Ptr()         ||
            !cx.pairsTmp.Ptr()      ||
            !cx.groupsBoundaries.Ptr() )
        {
            FreeBucketBuffers( cx );
            return false;
        }

        cx.maxEntriesPerBucket = entriesPerBucket;
    }

    return true;
}

//-----------------------------------------------------------
void FreeBucketBuffers( GreenReaperContext& cx )
{
    bbvirtfreebounded_span( cx.yBufferF1 );
    bbvirtfreebounded_span( cx.yBuffer );
    bbvirtfreebounded_span( cx.yBufferTmp );
    bbvirtfreebounded_span( cx.xBuffer );
    bbvirtfreebounded_span( cx.xBufferTmp );
    bbvirtfreebounded_span( cx.sortKey );
    bbvirtfreebounded_span( cx.metaBuffer );
    bbvirtfreebounded_span( cx.metaBufferTmp );
    bbvirtfreebounded_span( cx.pairs );
    bbvirtfreebounded_span( cx.pairsTmp );
    bbvirtfreebounded_span( cx.groupsBoundaries );

    for( uint32 i = 0; i < 7; i++ )
    {
        if( cx.tables[i]._pairs )
            bbvirtfreebounded( cx.tables[i]._pairs );
    }
}

//-----------------------------------------------------------
template<TableId rTable>
inline void SortTableAndFlipBuffers( GreenReaperContext& cx )
{
    using TMeta = typename K32MetaType<rTable>::Out;

    ProofTable& table = cx.tables[(int)rTable];

    const uint32 tableLength = (uint32)table._length;
    const uint32 threadCount = cx.config.threadCount;

    ASSERT( tableLength <= cx.pairs.Length() );
    ASSERT( tableLength <= table._capacity );

    // At this point the yRight/metaRight hold the unsorted fx output 
    // from the left table pairs/y/meta and sort onto the right buffers
    Span<uint64> tableYUnsorted     = MakeSpan( cx.proofContext.yRight, tableLength );
    Span<uint64> tableYSorted       = MakeSpan( cx.proofContext.yLeft , tableLength );
    Span<TMeta>  tableMetaUnsorted  = MakeSpan( (TMeta*)cx.proofContext.metaRight, tableLength );
    Span<TMeta>  tableMetaSorted    = MakeSpan( (TMeta*)cx.proofContext.metaLeft , tableLength );
    Span<Pair>   tablePairsUnsorted = cx.pairs.SliceSize( tableLength );
    Span<Pair>   tablePairsSorted   = MakeSpan( table._pairs, tableLength );

    Span<uint32> keyUnsorted        = cx.xBufferTmp.SliceSize( tableLength );
    Span<uint32> keySorted          = cx.xBuffer   .SliceSize( tableLength );

    const uint32 groupCount = 32 >> (int)rTable;

    for( uint32 i = 0; i < groupCount; i++ )
    {
        const uint32 groupLength  = table._groups[i].count;
        const uint32 groupThreads = std::min( threadCount, groupLength );

        auto yUnsorted     = tableYUnsorted    .SliceSize( groupLength );
        auto ySorted       = tableYSorted      .SliceSize( groupLength );
        auto metaUnsorted  = tableMetaUnsorted .SliceSize( groupLength );
        auto metaSorted    = tableMetaSorted   .SliceSize( groupLength );
        auto pairsUnsorted = tablePairsUnsorted.SliceSize( groupLength );
        auto pairsSorted   = tablePairsSorted  .SliceSize( groupLength );

        tableYUnsorted     = tableYUnsorted    .Slice( groupLength );
        tableYSorted       = tableYSorted      .Slice( groupLength );
        tableMetaUnsorted  = tableMetaUnsorted .Slice( groupLength );
        tableMetaSorted    = tableMetaSorted   .Slice( groupLength );
        tablePairsUnsorted = tablePairsUnsorted.Slice( groupLength ); 
        tablePairsSorted   = tablePairsSorted  .Slice( groupLength );

        
        auto kUnsorted = keyUnsorted.SliceSize( groupLength );
        auto kSorted   = keySorted  .SliceSize( groupLength );

        // #TODO: Perhaps do all groups and this in a single job (more code repetition, though)?
        SortKeyJob::GenerateKey( *cx.pool, groupThreads, keyUnsorted );

        RadixSort256::SortYWithKey<BB_MAX_JOBS>( *cx.pool, groupThreads, yUnsorted.Ptr(), ySorted.Ptr(),
                                                 kUnsorted.Ptr(), kSorted.Ptr(), groupLength );

        SortKeyJob::SortOnKey( *cx.pool, groupThreads, kSorted, metaUnsorted , metaSorted );
        SortKeyJob::SortOnKey( *cx.pool, groupThreads, kSorted, pairsUnsorted, pairsSorted );
    }

    cx.proofContext.leftLength  = (uint32)tableLength;
    cx.proofContext.rightLength = (uint32)cx.tables[(int)rTable+1]._capacity;
}

///
/// F1
///
//-----------------------------------------------------------
void GenerateF1( GreenReaperContext& cx, const byte plotId[32], const uint64 bucketEntryCount, const uint32 x0, const uint32 x1 )
{
    const uint32 k = 32;
    // Log::Line( " F1..." );
    // auto timer = TimerBegin();

    const uint32 f1BlocksPerBucket = (uint32)(bucketEntryCount * sizeof( uint32 ) / kF1BlockSize);

    uint32 threadCount = cx.pool->ThreadCount();

    while( f1BlocksPerBucket < threadCount )
        threadCount--;
    
    // Out buffers are continuous, so that we can merge both buckets into one
    uint32* xBuffer = cx.xBufferTmp.Ptr();
    uint64* yBuffer = cx.yBufferF1 .Ptr();

    auto blocks = Span<uint32>( (uint32*)cx.yBuffer.Ptr(), bucketEntryCount );

    const uint32 xSources[2] = { x0, x1 };

    Span<uint32> xEntries[2] = {
        Span<uint32>( xBuffer, bucketEntryCount ),
        Span<uint32>( xBuffer + bucketEntryCount, bucketEntryCount )
    };

    Span<uint64> yEntries[2] = {
        Span<uint64>( yBuffer, bucketEntryCount ),
        Span<uint64>( yBuffer + bucketEntryCount, bucketEntryCount )
    };

    byte key[32] = { 1 };
    memcpy( key+1, plotId, 31 );

    AnonMTJob::Run( *cx.pool, threadCount, [=]( AnonMTJob* self ) {

        const uint32 xShift            = k - kExtraBits;
        const uint32 f1EntriesPerBlock = kF1BlockSize / sizeof( uint32 );

        uint32 numBlocks, blockOffset, _;
        GetThreadOffsets( self, f1BlocksPerBucket, numBlocks, blockOffset, _ );

        const uint32 entriesPerThread = numBlocks * f1EntriesPerBlock;
        const uint32 entriesOffset    = blockOffset * f1EntriesPerBlock;

        Span<uint32> f1Blocks = blocks.Slice( blockOffset * f1EntriesPerBlock, numBlocks * f1EntriesPerBlock );

        chacha8_ctx chacha;
        chacha8_keysetup( &chacha, key, 256, NULL );

        for( uint32 i = 0; i < 2; i++ )
        {
            Span<uint32> xSlice = xEntries[i].Slice( entriesOffset, entriesPerThread );
            Span<uint64> ySlice = yEntries[i].Slice( entriesOffset, entriesPerThread );

            const uint32 xStart     = (uint32)(( xSources[i] * bucketEntryCount ) + entriesOffset);
            const uint32 blockIndex = xStart / f1EntriesPerBlock;

            chacha8_get_keystream( &chacha, blockIndex, numBlocks, (byte*)f1Blocks.Ptr() );

            for( uint32 j = 0; j < entriesPerThread; j++ )
            {
                // Get the starting and end locations of y in bits relative to our block
                const uint32 x = xStart + j;
                
                uint64 y = Swap32( f1Blocks[j] );
                y = ( y << kExtraBits ) | ( x >> xShift );

                xSlice[j] = x;
                ySlice[j] = y;
            }

// #if _DEBUG
//             {
//                 for( uint32 j = 0; j < entriesPerBucket; j++ )
//                     Log::Line( "[%2u] %-10u | 0x%08x", j,  xEntries[j], xEntries[j] );
//             }
// #endif
        }
    });

    // const auto elapsed = TimerEnd( timer );

    // Log::Line( "Completed F1 in %.2lf seconds.", TimerEnd( timer ) );


// #if _DEBUG
//     for( uint64 i = 0; i < bucketEntryCount*2; i++ )
//     {
//         const uint32 x = xBuffer[i];
//         if( x == 0x02a49264 || x == 0xb5b1cbdc )
//         {
//             Log::WriteLine( "Found {0x%08x} @ %llu", x, i );
//             // BBDebugBreak();
//         }
//     }
// #endif

    // Sort f1 on y
    const uint64 mergedEntryCount = bucketEntryCount * 2;
    RadixSort256::SortYWithKey<BB_MAX_JOBS>( *cx.pool, yBuffer, cx.yBuffer.Ptr(), xBuffer, cx.xBuffer.Ptr(), mergedEntryCount );

#if _DEBUG
    for( uint64 i = 0; i < bucketEntryCount*2; i++ )
    {
        const uint64 y = cx.yBuffer[i];
        const uint32 x = cx.xBuffer[i];

        // Log::Line( "")
        // if( x == 0x02a49264 || x == 0xb5b1cbdc )
        // {
        //     Log::WriteLine( "Found {0x%08x} @ %llu", x, i );
        //     // BBDebugBreak();
        // }
    }
#endif
}


///
/// Forward propagation
///
//-----------------------------------------------------------
template<TableId rTableId>
inline uint64 ForwardPropTableGroup( GreenReaperContext& cx, const uint32 lGroup, Span<Pair> outPairs, Span<uint64> yRight, Span<typename K32MetaType<rTableId>::Out> metaRight  )
{
    using TMetaIn  = typename K32MetaType<rTableId>::In;
    using TMetaOut = typename K32MetaType<rTableId>::Out;

    ASSERT( yRight.Length() == metaRight.Length() );

    ProofTable& lTable = cx.tables[(int)rTableId-1];
    ProofTable& rTable = cx.tables[(int)rTableId];

    Span<uint64>  yLeft;    
    Span<TMetaIn> metaLeft;
    lTable.GetLTableGroupEntries<rTableId-1>( cx, lGroup, yLeft, metaLeft );
    
    // Match
    const uint32 rGroup = lGroup / 2;
    if( (lGroup & 1) == 0)
        rTable.BeginGroup( rGroup );

    Span<Pair> pairs = Match( cx, yLeft, outPairs, lTable._groups[lGroup].offset );

    if( pairs.Length() > yRight.Length() )
        return 0;
    
    rTable.AddGroupPairs( rGroup, (uint32)pairs.Length() );

    // Fx
    if( pairs.Length() > 0 )
    {
        // Since pairs have the global L table offset applied to them,
        // we need to turn the left values back to global table y and meta, instead
        // of group-local y and meta
        yLeft    = MakeSpan( cx.proofContext.yLeft, cx.proofContext.leftLength );
        metaLeft = MakeSpan( (TMetaIn*)cx.proofContext.metaLeft, cx.proofContext.leftLength );

        GenerateFxForPairs<rTableId, TMetaIn, TMetaOut>( cx, pairs, yLeft, metaLeft, yRight, metaRight );
    }

    return pairs.Length();
}

//-----------------------------------------------------------
template<TableId rTable>
ForwardPropResult ForwardPropTable( GreenReaperContext& cx )
{
    using TMetaOut = typename K32MetaType<rTable>::Out;

    auto& table = cx.tables[(int)rTable];
    // ASSERT( table._length == 0 );

    auto yRight    = MakeSpan( cx.proofContext.yRight, table._capacity );
    auto metaRight = MakeSpan( (TMetaOut*)cx.proofContext.metaRight, table._capacity );
    auto outPairs  = cx.pairs.SliceSize( table._capacity );

    // Fp all groups in the table
    const uint32 groupCount = 32 >> ((int)rTable-1);
    uint64 matchCount = 0;

    for( uint32 i = 0; i < groupCount; i++ )
    {
        matchCount = ForwardPropTableGroup<rTable>( cx, i, outPairs, yRight, metaRight );

        if( matchCount == 0 )
            return ForwardPropResult::Failed;

        outPairs  = outPairs .Slice( matchCount ); 
        yRight    = yRight   .Slice( matchCount );
        metaRight = metaRight.Slice( matchCount );
    }

    SortTableAndFlipBuffers<rTable>( cx );

    // Table 6 makes no group entries, it should simply have a single match
    if( rTable == TableId::Table6 )
        return matchCount == 1 ? ForwardPropResult::Success : ForwardPropResult::Failed;

    // Check if we found a match already
    bool hasProof = true;
    
    for( uint32 i = 0; i < groupCount / 2; i++ )
    {
        if( table._groups[i].count != 2 )
        {
            hasProof = false;
            break;
        }
    }        

    return hasProof ? ForwardPropResult::Success : ForwardPropResult::Continue;
}

//-----------------------------------------------------------
bool ForwardPropTables( GreenReaperContext& cx )
{
    for( TableId rTable = TableId::Table3; rTable < TableId::Table7; rTable++ )
    {
        ForwardPropResult r;

        switch( rTable )
        {
            // case TableId::Table2: r = ForwardPropTable<TableId::Table2>( cx ); break;
            case TableId::Table3: r = ForwardPropTable<TableId::Table3>( cx ); break;
            case TableId::Table4: r = ForwardPropTable<TableId::Table4>( cx ); break;
            case TableId::Table5: r = ForwardPropTable<TableId::Table5>( cx ); break;
            case TableId::Table6: r = ForwardPropTable<TableId::Table6>( cx ); break;

            default:
                Fatal( "Unexpected table." );
                break;
        }

        switch( r )
        {
            case ForwardPropResult::Success:
                BacktraceProof( cx, std::min( rTable, TableId::Table6 ), cx.proofContext.proof );
                return true;

            case ForwardPropResult::Continue:
                break;
            
            default:
                ASSERT( 0 );
                return false;
        }   
    }

    return false;
}


///
/// Matching
///
//-----------------------------------------------------------
static void ScanBCGroups( GreenReaperContext& cx, const Span<uint64> yEntries, 
                          Span<uint32> groupBoundariesBuffer,
                          Span<uint32> groupBoundaries[BB_MAX_JOBS], uint32& outThreadCount );

static uint32 MatchJob( const uint32 id, const uint32 startIndex, const Span<uint32> groupBoundaries, const Span<uint64> yEntries, Span<Pair> pairs, const uint32 pairOffset  );

Span<Pair> Match( GreenReaperContext& cx, const Span<uint64> yEntries, Span<Pair> outputPairs, const uint32 pairOffset )
{
    struct Input
    {
        Span<uint32> groupBoundaries[BB_MAX_JOBS];
        Span<Pair>   pairBuffer;
        Span<uint64> yEntries;
        uint32       pairOffset;
    };

    struct Output
    {
        Span<Pair>          pairBuffer;
        uint32              matchCounts[BB_MAX_JOBS];
        std::atomic<uint32> pairCount = 0;
    };

    uint32 threadCount = 0;
    Input  input       = {};
    input.yEntries   = yEntries;
    input.pairBuffer = cx.pairsTmp;
    input.pairOffset = pairOffset;

    Output output = {};
    output.pairBuffer = outputPairs;

    // Re-use sort buffers for group boundaries and pairs
    auto groupsBoundariesBuffer = cx.groupsBoundaries; //Span<uint32>( cx.xBufferTmp, cx.maxEntriesPerBucket * 2 );

    // Get the group boundaries and the adjusted thread count
    ScanBCGroups( cx, yEntries, groupsBoundariesBuffer, input.groupBoundaries, threadCount );
    
    GenJobRun<Input, Output>( *cx.pool, threadCount, &input, &output, []( auto* self ){

        const uint32 id     = self->JobId();
        const auto&  input  = *self->input;
              auto&  output = *self->output;

        const uint32 startIndex = id == 0 ? 0 : input.groupBoundaries[id-1][input.groupBoundaries[id-1].Length()-1];
    
        Span<Pair> tmpPairs = GetThreadOffsets( self, input.pairBuffer );

        const uint32 matchCount = MatchJob( id, startIndex, input.groupBoundaries[id], input.yEntries, tmpPairs, input.pairOffset );

        output.matchCounts[id] =  matchCount;
        output.pairCount       += matchCount;
        
        // Ensure we have enough space in the output buffer
        Span<Pair> outPairs = output.pairBuffer;

        if( self->BeginLockBlock() )
        {
            // #TODO: Set an error, there should not be more pairs than bucket size
            if( outPairs.Length() < output.pairCount )
                output.pairCount = 0;
        }
        self->EndLockBlock();

        if( matchCount )
        {
            uint32 copyOffset = 0;
            for( uint32 i = 0; i < id; i++ )
                copyOffset += output.matchCounts[i];

            tmpPairs.CopyTo( outPairs.Slice( copyOffset, matchCount ), matchCount );
        }
    });

    return output.pairBuffer.SliceSize( output.pairCount );
}


//-----------------------------------------------------------
void ScanBCGroups( GreenReaperContext& cx, const Span<uint64> yEntries, 
                   Span<uint32> groupBoundariesBuffer, Span<uint32> groupBoundaries[BB_MAX_JOBS], uint32& outThreadCount )
{
    uint32 startPositions[BB_MAX_JOBS] = {};
    uint32 entryCounts   [BB_MAX_JOBS] = {};

    // Find thread start positions
          ThreadPool& pool        = *cx.pool;
          uint32      threadCount = pool.ThreadCount();
    const uint32      entryCount  = (uint32)yEntries.Length();

    // Lower thread count depending on how many entries we've got
    while( threadCount > entryCount )
        threadCount--;

    if( entryCount < 4096 )
        threadCount = 1;

    outThreadCount = threadCount;

    const uint32 groupBufferCapacity = (uint32)groupBoundariesBuffer.Length();
    const uint32 groupsPerThread     = (uint32)(groupBufferCapacity / threadCount);
    
    // Re-use the sorting buffer for group boundaries

    // Find the scan starting position
    groupBoundaries[0]    = groupBoundariesBuffer.SliceSize( groupsPerThread );
    groupBoundariesBuffer = groupBoundariesBuffer.Slice( groupsPerThread );

    for( uint32 i = 1; i < threadCount; i++ )
    {
        uint32 count, offset, _;
        GetThreadOffsets( i, threadCount, entryCount, count, offset, _ );

        const uint64* start   = yEntries.Ptr();
        const uint64* entries = start + offset;

        uint64 curGroup = *entries / kBC;
        while( entries > start )
        {
            if( entries[-1] / kBC != curGroup )
                break;
            --entries;
        }

        startPositions [i]    = (uint32)(uintptr_t)(entries - start);
        entryCounts    [i-1]  = startPositions[i] - startPositions[i-1];
        groupBoundaries[i]    = groupBoundariesBuffer.SliceSize( groupsPerThread );
        groupBoundariesBuffer = groupBoundariesBuffer.Slice( groupsPerThread );
    }

    entryCounts[threadCount-1] = entryCount - startPositions[threadCount-1];

    AnonMTJob::Run( *cx.pool, threadCount, [&]( AnonMTJob* self ) {

        const uint32 id = self->JobId();

        Span<uint32>  groups      = groupBoundaries[id];
        const uint32  maxGroups   = (uint32)groups.Length();
        uint32        groupCount  = 0;
        const uint32  entryOffset = startPositions[id];
        auto          entries     = yEntries.Slice( entryOffset, entryCounts[id] );

        uint64 curGroup = entries[0] / kBC;
        
        uint32 i;
        for( i = 1; i < entries.Length(); i++ )
        {
            const uint64 g = entries[i] / kBC;

            if( curGroup != g )
            {
                FatalIf( groupCount >= maxGroups, "Group count exceeded." );
                groups[groupCount++] = entryOffset + i;
                curGroup = g;
            }
        }

        // Add the end location of the last R group
        if( !self->IsLastThread() )
        {
            FatalIf( groupCount >= maxGroups-1, "Group count exceeded." );  // Need 2 places for the last ghost group
            groups[groupCount] = startPositions[id+1];                      // Where our last usable group ends
            groupCount += 2;
        }
        else
        {
            FatalIf( groupCount >= maxGroups, "Group count exceeded." );
            groups[groupCount++] = (uint32)yEntries.Length();
        }

        groups.length       = groupCount;
        groupBoundaries[id] = groups;
    });

    // Add the ghost groups, which tells us where the last R group
    // of a group ends (only for non-last threads)
    for( uint32 i = 0; i < threadCount-1; i++ )
    {
        auto& group = groupBoundaries[i];
        group[group.Length()-1] = (uint32)groupBoundaries[i+1][0];
    }
}

// #TODO: Add SIMD optimization
//-----------------------------------------------------------
uint32 MatchJob( const uint32 id, const uint32 startIndex, const Span<uint32> groupBoundaries, const Span<uint64> yEntries, Span<Pair> pairs, const uint32 pairOffset )
{
    // const uint32 id         = self->JobId();
    const uint32 groupCount = (uint32)groupBoundaries.Length() - 1; // Ignore the extra ghost group
    const uint32 maxPairs   = (uint32)pairs.Length();

    uint32 pairCount = 0;

    uint8  rMapCounts [kBC];
    uint16 rMapIndices[kBC];

    uint64 groupLStart = startIndex;
    uint64 groupL      = yEntries[groupLStart] / kBC;

    for( uint32 i = 0; i < groupCount; i++ )
    {
        const uint64 groupRStart = groupBoundaries[i];
        const uint64 groupR      = yEntries[groupRStart] / kBC;

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
                uint64 localRY = yEntries[iR] - groupRRangeStart;
                ASSERT( yEntries[iR] / kBC == groupR );

                if( rMapCounts[localRY] == 0 )
                    rMapIndices[localRY] = (uint16)( iR - groupRStart );

                rMapCounts[localRY] ++;
            }

            // For each group L entry
            for( uint64 iL = groupLStart; iL < groupRStart; iL++ )
            {
                const uint64 yL     = yEntries[iL];
                const uint64 localL = yL - groupLRangeStart;

                for( int iK = 0; iK < kExtraBitsPow; iK++ )
                {
                    const uint64 targetR = L_targets[parity][localL][iK];

                    // for( uint j = 0; j < rMapCounts[targetR]; j++ )
                    if( rMapCounts[targetR] > 0 )
                    {
                        for( uint j = 0; j < rMapCounts[targetR]; j++ )
                        {
                            const uint64 iR = groupRStart + rMapIndices[targetR] + j;
                            ASSERT( iL < iR );

                            // Add a new pair
                            FatalIf( pairCount >= maxPairs, "Too many pairs found." );

                            Pair& pair = pairs[pairCount++];
                            pair.left  = (uint32)iL + pairOffset;
                            pair.right = (uint32)iR + pairOffset;
                            // Log::Line( "M: %-8u, %8u : %-12llu, %-12llu", pair.left, pair.right, yL, yEntries[iR] );
                        }
                    }
                }

                //NEXT_L:;
            }
        }
        // Else: Not an adjacent group, skip to next one.

        // Go to next group
        groupL      = groupR;
        groupLStart = groupRStart;
    }

    return pairCount;
}


///
/// Fx
///

//-----------------------------------------------------------
template<TableId rTable, typename TMetaIn, typename TMetaOut>
void GenerateFxForPairs( GreenReaperContext& cx, const Span<Pair> pairs, const Span<uint64> yIn, const Span<TMetaIn> metaIn, Span<uint64> yOut, Span<TMetaOut> metaOut )
{
    ASSERT( yOut.Length() >= pairs.Length() );
    ASSERT( metaOut.Length() >= pairs.Length() );

    const uint32 threadCount = std::min( cx.config.threadCount, (uint32)pairs.Length() );

    if( threadCount == 1 )
    {
        GenerateFx<rTable, TMetaIn, TMetaOut>( pairs, yIn, metaIn, yOut, metaOut );
        return;
    }

    AnonMTJob::Run( *cx.pool, cx.config.threadCount, [&]( AnonMTJob* self ){
        
        uint64 count, offset, _;
        GetThreadOffsets( self, (uint64)pairs.Length(), count, offset, _ );
        
        GenerateFx<rTable, TMetaIn, TMetaOut>( pairs.Slice( offset, count ), yIn, metaIn, yOut.Slice( offset, count ), metaOut.Slice( offset, count ) );
    });
}

//-----------------------------------------------------------
template<TableId rTable, typename TMetaIn, typename TMetaOut>
inline void GenerateFx( const Span<Pair> pairs, const Span<uint64> yIn, const Span<TMetaIn> metaIn, Span<uint64> yOut, Span<TMetaOut> outMeta )
{
    constexpr size_t MetaInMulti  = TableMetaIn <rTable>::Multiplier;
    constexpr size_t MetaOutMulti = TableMetaOut<rTable>::Multiplier;
    static_assert( MetaInMulti != 0, "Invalid metaKMultiplier" );

    const uint32 k           = 32;
    const uint32 shiftBits   = MetaOutMulti == 0 ? 0 : kExtraBits;
    const uint32 ySize       = k + kExtraBits;         // = 38
    const uint32 yShift      = 64 - (k + shiftBits);   // = 26 or 32
    
    const size_t metaSize    = k * MetaInMulti;
    const size_t metaSizeLR  = metaSize * 2;
    const size_t bufferSize  = CDiv( ySize + metaSizeLR, 8 );

    // Hashing
    uint64 input [5]; // y + L + R
    uint64 output[4]; // blake3 hashed output

    blake3_hasher hasher;
    static_assert( bufferSize <= sizeof( input ), "Invalid fx input buffer size." );

    for( uint64 i = 0; i < pairs.Length(); i++ )
    {
        const Pair   pair = pairs[i];
        const uint64 y    = yIn[pair.left];

        const TMetaIn metaL = metaIn[pair.left ];
        const TMetaIn metaR = metaIn[pair.right];

        TMetaOut& mOut = outMeta[i];

        if constexpr( MetaInMulti == 1 )
        {
            const uint64 l = metaL;
            const uint64 r = metaR;

            input[0] = Swap64( y << 26 | l >> 6  );
            input[1] = Swap64( l << 58 | r << 26 );

            // Metadata is just L + R of 8 bytes
            if constexpr( MetaOutMulti == 2 )
                mOut = l << 32 | r;
        }
        else if constexpr( MetaInMulti == 2 )
        {
            const uint64 l = metaL;
            const uint64 r = metaR;

            input[0] = Swap64( y << 26 | l >> 38 );
            input[1] = Swap64( l << 26 | r >> 38 );
            input[2] = Swap64( r << 26 );

            // Metadata is just L + R again of 16 bytes
            if constexpr( MetaOutMulti == 4 )
            {
                mOut.m0 = l;
                mOut.m1 = r;
            }
        }
        else if constexpr( MetaInMulti == 3 )
        {
            const uint64 l0 = metaL.m0;
            const uint64 l1 = metaL.m1 & 0xFFFFFFFF;
            const uint64 r0 = metaR.m0;
            const uint64 r1 = metaR.m1 & 0xFFFFFFFF;
        
            input[0] = Swap64( y  << 26 | l0 >> 38 );
            input[1] = Swap64( l0 << 26 | l1 >> 6  );
            input[2] = Swap64( l1 << 58 | r0 >> 6  );
            input[3] = Swap64( r0 << 58 | r1 << 26 );
        }
        else if constexpr( MetaInMulti == 4 )
        {
            const auto l = metaL;
            const auto r = metaR;

            input[0] = Swap64( y    << 26 | l.m0 >> 38 );
            input[1] = Swap64( l.m0 << 26 | l.m1 >> 38 );
            input[2] = Swap64( l.m1 << 26 | r.m0 >> 38 );
            input[3] = Swap64( r.m0 << 26 | r.m1 >> 38 );
            input[4] = Swap64( r.m1 << 26 );
        }

        // Hash the input
        blake3_hasher_init( &hasher );
        blake3_hasher_update( &hasher, input, bufferSize );
        blake3_hasher_finalize( &hasher, (uint8_t*)output, sizeof( output ) );

        const uint64 f = Swap64( *output ) >> yShift;
        yOut[i] = f;

        if constexpr ( MetaOutMulti == 2 && MetaInMulti == 3 )
        {
            const uint64 h0 = Swap64( output[0] );
            const uint64 h1 = Swap64( output[1] );

            mOut = h0 << ySize | h1 >> 26;
        }
        else if constexpr ( MetaOutMulti == 3 )
        {
            const uint64 h0 = Swap64( output[0] );
            const uint64 h1 = Swap64( output[1] );
            const uint64 h2 = Swap64( output[2] );

            mOut.m0 = h0 << ySize | h1 >> 26;
            mOut.m1 = ((h1 << 6) & 0xFFFFFFC0) | h2 >> 58;
        }
        else if constexpr ( MetaOutMulti == 4 && MetaInMulti != 2 ) // In = 2 is calculated above with L + R
        {
            const uint64 h0 = Swap64( output[0] );
            const uint64 h1 = Swap64( output[1] );
            const uint64 h2 = Swap64( output[2] );

            mOut.m0 = h0 << ySize | h1 >> 26;
            mOut.m1 = h1 << 38    | h2 >> 26;
        }
    }
}

