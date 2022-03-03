#include "catch2/catch_test_macros.hpp"
#include "util/jobs/MemJobs.h"
#include "util/BitView.h"
#include "util/Log.h"
#include "threading/ThreadPool.h"
#include "plotting/PlotTypes.h"
#include "plotting/PlotTools.h"
#include "plotting/PackEntries.h"
#include "plotting/Tables.h"
#include "plotting/GenSortKey.h"
#include "pos/chacha8.h"
#include "plotdisk/DiskPlotConfig.h"
#include "algorithm/RadixSort.h"

#define ENSURE( x ) \
    if( !(x) ) { ASSERT( x ); REQUIRE( x ); }

template<TableId table>
struct FpEntry {};

template<> struct FpEntry<TableId::Table1> { byte bytes[8 ]; };
template<> struct FpEntry<TableId::Table2> { byte bytes[16]; };
template<> struct FpEntry<TableId::Table3> { byte bytes[24]; };
template<> struct FpEntry<TableId::Table4> { byte bytes[24]; };
template<> struct FpEntry<TableId::Table5> { byte bytes[24]; }; // It should be 20, but we use an extra 4 to round up to 64 bits.
template<> struct FpEntry<TableId::Table6> { byte bytes[16]; };
template<> struct FpEntry<TableId::Table7> { byte bytes[8 ]; };

typedef FpEntry<TableId::Table1> T1Entry;
typedef FpEntry<TableId::Table2> T2Entry;
typedef FpEntry<TableId::Table3> T3Entry;
typedef FpEntry<TableId::Table4> T4Entry;
typedef FpEntry<TableId::Table5> T5Entry;
typedef FpEntry<TableId::Table6> T6Entry;
typedef FpEntry<TableId::Table7> T7Entry;

static_assert( sizeof( T1Entry ) == 8  );
static_assert( sizeof( T2Entry ) == 16 );
static_assert( sizeof( T3Entry ) == 24 );
static_assert( sizeof( T4Entry ) == 24 );
static_assert( sizeof( T5Entry ) == 24 );
static_assert( sizeof( T6Entry ) == 16 );
static_assert( sizeof( T7Entry ) == 8  );


static void TestBucket( uint32 bucketCount, uint32 k );
static void FakeFxGen( const int64 entryCount, byte seed[BB_PLOT_ID_LEN],  uint64* y, uint64* map, uint64* meta, const size_t metaSize );

template<typename T, typename TJob, typename TBuckets = uint64>
static void FxSort( PrefixSumJob<TJob,uint64>* self, const int64 entryCount, const int64 offset, T* entries, T* tmpEntries, const uint32 entryBitSize );

struct FxSortJob : PrefixSumJob<FxSortJob, uint64>
{
    int64   offset;
    int64   entryCount;
    void*   entries;
    void*   tmp;
    TableId table;
    uint32  remainderBits;

    template<TableId table>
    static void Sort( ThreadPool& pool, const uint32 threadCount, void* input, void* tmp, const int64 entryCount, const uint32 remainderBits );
    
    void Run() override;

    template<TableId table>
    void RunForTable();
};

uint64* _ySrc       = nullptr;
uint64* _mapSrc     = nullptr;
uint64* _metaSrc    = nullptr;
uint64* _yDst       = nullptr;
uint64* _mapDst     = nullptr;
uint64* _metaDst    = nullptr;
uint64* _entries    = nullptr;
uint64* _entriesTmp = nullptr;


inline uint32 GetYBitsForBuckets( const uint32 numBuckets )
{
    return _K - ( (uint32)log2( numBuckets ) - 6 );
}

template<TableId table>
static void TestBucketForTable( ThreadPool& pool, byte seed[BB_PLOT_ID_LEN], const uint32 threadCount, const int64 entryCount, const uint32 numBuckets );

template<TableId table>
static void FxExpandEntries( const uint32 numBuckets, ThreadPool& pool, const uint32 threadCount, void* input, uint64* output, const int64 entryCount );


//-----------------------------------------------------------
TEST_CASE( "FxSort", "[fx]" )
{
    const uint32 k                = _K;
    const uint64 maxEntries       = 1ull << k;
    const uint64 maxBucketEntries = maxEntries / 16;//maxEntries / 64;
    const uint32 threadCount      = SysHost::GetLogicalCPUCount();
    const char   seedHex[]        = "7a709594087cca18cffa37be61bdecf9b6b465de91acb06ecb6dbe0f4a536f73";

    byte seed[BB_PLOT_ID_LEN];
    HexStrToBytes( seedHex, sizeof( seedHex )-1, seed, sizeof( seed ) );

    ThreadPool pool( threadCount );

    const size_t compressedEntriesAlloc = RoundUpToNextBoundary( maxBucketEntries * 24, 24 );
    const size_t totalMemory            = ( (maxBucketEntries + 128) * 6 * sizeof( uint64 ) ) + ( compressedEntriesAlloc * 2 );
    Log::Line( "Allocating %.2lf GiB of memory.", (double)totalMemory BtoGB );

    _ySrc       = bbcvirtallocboundednuma<uint64>( maxBucketEntries + 128 );
    _mapSrc     = bbcvirtallocboundednuma<uint64>( maxBucketEntries + 128 );
    _metaSrc    = bbcvirtallocboundednuma<uint64>( maxBucketEntries + 128 );
    _yDst       = bbcvirtallocboundednuma<uint64>( maxBucketEntries + 128 );
    _mapDst     = bbcvirtallocboundednuma<uint64>( maxBucketEntries + 128 );
    _metaDst    = bbcvirtallocboundednuma<uint64>( maxBucketEntries + 128 );
    
    _entries    = bbvirtallocboundednuma<uint64>( compressedEntriesAlloc );
    _entriesTmp = bbvirtallocboundednuma<uint64>( compressedEntriesAlloc );

    FaultMemoryPages::RunJob( pool, threadCount, _ySrc      , maxBucketEntries * sizeof( uint64 ) );
    FaultMemoryPages::RunJob( pool, threadCount, _mapSrc    , maxBucketEntries * sizeof( uint64 ) );
    FaultMemoryPages::RunJob( pool, threadCount, _metaSrc   , maxBucketEntries * sizeof( uint64 ) );
    FaultMemoryPages::RunJob( pool, threadCount, _yDst      , maxBucketEntries * sizeof( uint64 ) );
    FaultMemoryPages::RunJob( pool, threadCount, _mapDst    , maxBucketEntries * sizeof( uint64 ) );
    FaultMemoryPages::RunJob( pool, threadCount, _metaDst   , maxBucketEntries * sizeof( uint64 ) );
    FaultMemoryPages::RunJob( pool, threadCount, _entries   , compressedEntriesAlloc );
    FaultMemoryPages::RunJob( pool, threadCount, _entriesTmp, compressedEntriesAlloc );

    uint32 buckets[] = { 128, 256, 512, 1024 };
    
    // TestBucketForTable<TableId::Table2>( pool, seed, 5, maxBucketEntries, 256 );

    // for( uint32 t = 32; t <= threadCount; t++ )
    const uint32 t = SysHost::GetLogicalCPUCount();
    {
        Log::Line( "[Threads: %u]", t );
        Log::Line( "----------------------------------------" );
        for( uint32 i = 0; i < 4; i++ )
        {
            // const int64 entryCount = maxEntries;
            // const int64 entryCount = maxEntries / (int64)buckets[i];;
            const int64 entryCount = maxBucketEntries;
            TestBucketForTable<TableId::Table2>( pool, seed, t, entryCount , buckets[i] );
            Log::Line( "" );
        }
        Log::Line( "" );
    }
}


//-----------------------------------------------------------
template<TableId table>
void TestBucketForTable( ThreadPool& pool, byte seed[BB_PLOT_ID_LEN], const uint32 threadCount, const int64 entryCount, const uint32 numBuckets )
{
    using Entry = FpEntry<table>;

    const uint32 k                  = _K;
    const uint32 yBits              = GetYBitsForBuckets( numBuckets );
    const uint32 mapBits            = k + 1;
    const uint32 metaBits           = TableMetaOut<table>::Multiplier * k;
    const size_t entrySizeBits      = yBits + mapBits + metaBits;
    const size_t unpackedEntrySize  = sizeof( Entry ) * 8;
    
    ASSERT( unpackedEntrySize >= entrySizeBits );

    // const uint64 maxEntries           = 1ull << k;
    // const int64  entryCount           = (int64)( maxEntries / numBuckets );
    const size_t entriesTotalSizeBits = (uint64)entryCount * entrySizeBits;
    const uint64 yMask                = 0xFFFFFFFFFFFFFFFFull >> (64-yBits);
    const uint32 yBump                = entrySizeBits - yBits;

    const int64 entriesPerThread  = entryCount / (int64)threadCount;
    const int64 lastThreadEntries = entriesPerThread +  entryCount - entriesPerThread *(int64)threadCount;

    int64 offsets    [BB_MAX_JOBS] = { 0 };
    int64 entryCounts[BB_MAX_JOBS] = { 0 };
    int64 ends       [BB_MAX_JOBS] = { 0 };

    for( int64 i = 0; i < (int64)threadCount; i++ )
    {
        offsets[i]     = i * entriesPerThread;
        entryCounts[i] = entriesPerThread;
        ends[i]        = offsets[i] + entryCounts[i];
    }
    entryCounts[threadCount-1] = lastThreadEntries;
    ends[threadCount-1] = entryCount;


    Log::Line( "[Buckets: %u]", numBuckets );
    Log::Line( " Thread Count  : %u", threadCount );
    Log::Line( " Entry Count   : %lld", entryCount );
    Log::Line( " Entries/Thread: %lld", entryCount / (int64)threadCount );
    Log::Line( " Entry Size    : %u bits", entrySizeBits );
    Log::Line( " Unpacked Size : %u bits (%u) bytes", unpackedEntrySize, unpackedEntrySize / 8 );

    uint64* ySrc       = _ySrc;
    uint64* mapSrc     = _mapSrc;
    uint64* metaSrc    = _metaSrc;
    uint64* yDst       = _yDst;
    uint64* mapDst     = _mapDst;
    uint64* metaDst    = _metaDst;
    uint64* entries    = _entries;
    uint64* entriesTmp = _entriesTmp;

    Log::Line( "FxGen" );
    auto timer = TimerBegin();
    AnonMTJob::Run( pool, threadCount, [=]( AnonMTJob* self ) {
        const auto id = self->JobId();
        FakeFxGen( entryCounts[id], seed, ySrc+offsets[id], mapSrc+offsets[id], metaSrc+offsets[id], sizeof( uint64 ) );

        for( int64 i = offsets[id]; i < ends[id]; i++ )
            mapSrc[i] = (uint32)i;
    });
    Log::Line( " Elasped: %.2lfs", TimerEnd( timer ) );

    Log::Line( "Packing entries" );
    timer = TimerBegin();
    AnonMTJob::Run( pool, threadCount, [=]( AnonMTJob* self ) {
        
        const uint32 id = self->JobId();

        // Writte the last entry first
        {
            const uint64 lastEntry = ends[id]-1;
            
            BitWriter writer( entries, entriesTotalSizeBits, lastEntry * entrySizeBits );

            writer.Write( ySrc   [lastEntry], yBits    );
            writer.Write( mapSrc [lastEntry], mapBits  );
            writer.Write( metaSrc[lastEntry], metaBits );
        }

        self->SyncThreads();

         {
            BitWriter writer( entries, entriesTotalSizeBits, offsets[id] * entrySizeBits );

            for( int64 i = offsets[id]; i < ends[id]-1; i++ )
            {
                writer.Write( ySrc   [i], yBits    );
                writer.Write( mapSrc [i], mapBits  );
                writer.Write( metaSrc[i], metaBits );
            }
         }
    });
    Log::Line( " Elasped: %.2lfs", TimerEnd( timer ) );

    Log::Line( "Masking source entries" );
    timer = TimerBegin();
    AnonMTJob::Run( pool, threadCount, [=]( AnonMTJob* self ) {

        const uint32 id = self->JobId();
        for( int64 i = offsets[id]; i < ends[id]; i++ )
            ySrc[i] = ySrc[i] & yMask;
    });
    Log::Line( " Elapsed: %.2lfs", TimerEnd( timer ) );

    Log::Line( "Testing unpacked" );
    timer = TimerBegin();
    AnonMTJob::Run( pool, threadCount, [=]( AnonMTJob* self ) {

        const uint32 id = self->JobId();
        BitReader reader( entries, entriesTotalSizeBits, offsets[id] * entrySizeBits );

        for( int64 i = offsets[id]; i < ends[id]; i++ )
        {
            const uint64 yRef = ySrc[i];
            const uint64 y    = reader.ReadBits64( yBits );
            ENSURE( yRef == y );

            reader.Bump( yBump );
        }
    });
    Log::Line( " Elapsed: %.2lfs", TimerEnd( timer ) );

    Log::Line( "Expanding entries" );
    timer = TimerBegin();

    FxExpandEntries<table>( numBuckets, pool, threadCount, entries, entriesTmp, entryCount );

    std::swap( entries, entriesTmp );
    Log::Line( " Elapsed: %.2lfs", TimerEnd( timer ) );

    Log::Line( "Testing expanded unpacked" );
    timer = TimerBegin();
    AnonMTJob::Run( pool, threadCount, [=]( AnonMTJob* self ) {

        const uint32 id          = self->JobId();
        const uint32 entrySize64 = unpackedEntrySize / 8 / sizeof( uint64 );

        for( int64 i = offsets[id], j = offsets[id]*entrySize64; i < ends[id]; i++, j += entrySize64 )
        {
            const uint64 yRef = ySrc[i];
            const uint64 y    = entries[j] & yMask;
            ENSURE( yRef == y );

            const uint64 mapRef = mapSrc[i];
            const uint64 map    = entries[j] >> yBits;
            ENSURE( mapRef == map );
        }
    });
    Log::Line( " Elapsed: %.2lfs", TimerEnd( timer ) );
    
    /// 
    /// Sort with separate components
    ///
    Log::Line( "Sorting components with key" );
    timer = TimerBegin();
    {
        auto stimer = timer;
        Log::Line( "Creating a sort key" );
        AnonMTJob::Run( pool, threadCount, [=]( AnonMTJob* self ) {
            
            const auto id = self->JobId();
            for( int64 i = offsets[id]; i < ends[id]; i++ )
                ySrc[i] = ( ySrc[i] & 0xFFFFFFFFull ) | ( (uint64)i << 32 );
        });
        Log::Line( " Elapsed: %.2lfs", TimerEnd( stimer ) );
        
        Log::Line( "Sorting source entries" );
        stimer = TimerBegin();
        RadixSort256::Sort<BB_MAX_JOBS, uint64, 4>( pool, threadCount, ySrc, yDst, (uint64)entryCount );
        Log::Line( " Elapsed: %.2lfs", TimerEnd( stimer ) );


        Log::Line( "Unpacking sort key." );
        stimer = TimerBegin();

        uint32* sortKey = (uint32*)yDst;
        AnonMTJob::Run( pool, threadCount, [=]( AnonMTJob* self ) {
            
            const auto id = self->JobId();
            for( int64 i = offsets[id]; i < ends[id]; i++ )
                sortKey[i] = (uint32)( ySrc[i] >> 32 );
        });
        Log::Line( " Elapsed: %.2lfs", TimerEnd( stimer ) );

        Log::Line( "Masking y entries" );
        stimer = TimerBegin();
        AnonMTJob::Run( pool, threadCount, [=]( AnonMTJob* self ) {

            const uint32 id = self->JobId();
            for( int64 i = offsets[id]; i < ends[id]; i++ )
                ySrc[i] = ySrc[i] & yMask;
        });
        Log::Line( " Elapsed: %.2lfs", TimerEnd( stimer ) );

        Log::Line( "Sorting map on Y" );
        stimer = TimerBegin();
        // SortKeyGen::Sort<BB_MAX_JOBS>( pool, threadCount, entryCount, sortKey, mapSrc, mapDst );
        AnonMTJob::Run( pool, threadCount, [=]( AnonMTJob* self ) {

            const uint32 id = self->JobId();
            for( int64 i = offsets[id]; i < ends[id]; i++ )
                mapDst[i] = mapSrc[sortKey[i]];
        });
        Log::Line( " Elapsed: %.2lfs", TimerEnd( stimer ) );

        Log::Line( "Sorting meta on Y" );
        stimer = TimerBegin();
        SortKeyGen::Sort<BB_MAX_JOBS>( pool, threadCount, entryCount, sortKey, metaSrc, metaDst );
        Log::Line( " Elapsed: %.2lfs", TimerEnd( stimer ) );
    }
    Log::Line( "  Component Sort Elapsed: %.2lfs", TimerEnd( timer ) );


    ///
    /// Sort with packed components
    ///
    Log::Line( "Sorting packed entries" );
    timer = TimerBegin();
    {
        auto stimer = timer;
        FxSortJob::Sort<TableId::Table2>( pool, threadCount, entries, entriesTmp, entryCount, k - yBits );
        Log::Line( " Elapsed: %.2lfs", TimerEnd( stimer ) );

        Log::Line( "Unpacking Y" );
        stimer = TimerBegin();
        AnonMTJob::Run( pool, threadCount, [=]( AnonMTJob* self ) {

            const uint32 id          = self->JobId();
            const uint32 entrySize64 = unpackedEntrySize / 8 / sizeof( uint64 );

            for( int64 i = offsets[id], j = offsets[id]*entrySize64; i < ends[id]; i++, j += entrySize64 )
                yDst[i] = entries[j] & yMask;
        });
        Log::Line( " Elapsed: %.2lfs", TimerEnd( stimer ) );

        Log::Line( "Unpacking Map" );
        stimer = TimerBegin();
        AnonMTJob::Run( pool, threadCount, [=]( AnonMTJob* self ) {

            const uint32 id          = self->JobId();
            const uint32 entrySize64 = unpackedEntrySize / 8 / sizeof( uint64 );

            for( int64 i = offsets[id], j = offsets[id]*entrySize64; i < ends[id]; i++, j += entrySize64 )
                mapSrc[i] = entries[j] >> yBits;
        });
        Log::Line( " Elapsed: %.2lfs", TimerEnd( stimer ) );

        Log::Line( "Unpacking Meta" );
        stimer = TimerBegin();
        AnonMTJob::Run( pool, threadCount, [=]( AnonMTJob* self ) {

            const uint32 id          = self->JobId();
            const uint32 entrySize64 = unpackedEntrySize / 8 / sizeof( uint64 );

            for( int64 i = offsets[id], j = offsets[id]*entrySize64+1; i < ends[id]; i++, j += entrySize64 )
                metaSrc[i] = entries[j];
        });
        Log::Line( " Elapsed: %.2lfs", TimerEnd( stimer ) );
    }
    Log::Line( " Packed Entry Sort Elapsed: %.2lfs", TimerEnd( timer ) );

    // Ensure we are sorted
    Log::Line( "Validating sorted entries" );
    timer = TimerBegin();
    AnonMTJob::Run( pool, threadCount, [=]( AnonMTJob* self ) {
        
        const auto id = self->JobId();

        for( int64 i = offsets[id]; i < ends[id]; i++ )
        {
            const uint64 yRef = ySrc[i];
            const uint64 y    = yDst[i];
            ENSURE( yRef == y );

            const uint64 mapRef = mapDst[i];
            const uint64 map    = mapSrc[i];
            ENSURE( mapRef == map );

            const uint64 metaRef = metaDst[i];
            const uint64 meta    = metaSrc[i];
            ENSURE( metaRef == meta );
        }
    });
    
    Log::Line( " Elapsed: %.2lfs", TimerEnd( timer ) );
}

//-----------------------------------------------------------
void FakeFxGen( const int64 entryCount, byte seed[BB_PLOT_ID_LEN], uint64* y, uint64* map, uint64* meta, const size_t metaSize )
{
    chacha8_ctx chacha;
    chacha8_keysetup( &chacha, seed, 256, NULL );

    const uint64 blockCount = CDiv( (size_t)entryCount * sizeof( uint64 ), kF1BlockSize );

    chacha8_get_keystream( &chacha, 0, blockCount, (uint8_t*)y    );
    chacha8_get_keystream( &chacha, 0, blockCount, (uint8_t*)meta );
    // chacha8_get_keystream( &chacha, 0, blockCount, (uint8_t*)map  );
}

//-----------------------------------------------------------
template<TableId table>
void FxSortJob::Sort( ThreadPool& pool, const uint32 threadCount, void* input, void* tmp, const int64 entryCount, const uint32 remainderBits  )
{
    ASSERT( input );
    ASSERT( tmp );
    ASSERT( entryCount > 0 );

    const int64 entriesPerThread = entryCount / (int64)threadCount;

    MTJobRunner<FxSortJob> jobs( pool );

    for( int64 i = 0; i < (int64)threadCount; i++ )
    {
        auto& job = jobs[i];
        job.offset        = entriesPerThread * i;
        job.entryCount    = entriesPerThread;
        job.entries       = input;
        job.tmp           = tmp;
        job.table         = table;
        job.remainderBits = remainderBits;
    }
    jobs[threadCount-1].entryCount += entryCount - entriesPerThread * (int64)threadCount;
    jobs.Run( threadCount );
}

//-----------------------------------------------------------
void FxSortJob::Run()
{
    switch( table )
    {
        case TableId::Table1: RunForTable<TableId::Table1>(); break;
        case TableId::Table2: RunForTable<TableId::Table2>(); break;
        case TableId::Table3: RunForTable<TableId::Table3>(); break;
        case TableId::Table4: RunForTable<TableId::Table4>(); break;
        case TableId::Table5: RunForTable<TableId::Table5>(); break;
        case TableId::Table6: RunForTable<TableId::Table6>(); break;
        case TableId::Table7: RunForTable<TableId::Table7>(); break;
    
        default:
            ASSERT( 0 );
            break;
    }
}

//-----------------------------------------------------------
template<TableId table>
void FxSortJob::RunForTable()
{
    using Entry = FpEntry<table>;
    
    FxSort( this, entryCount, offset, (Entry*)entries, (Entry*)tmp, remainderBits );
}

//-----------------------------------------------------------
template<typename T, typename TJob, typename TBuckets>
void FxSort( PrefixSumJob<TJob, uint64>* self, const int64 entryCount, const int64 offset, T* entries, T* tmpEntries, const uint32 remainderBits )
{
    ASSERT( self );
    ASSERT( entryCount > 0);
    ASSERT( entries );
    ASSERT( tmpEntries );

    using BucketT = uint64;

    constexpr uint Radix = 256;

    constexpr int32 iterations = 4; //MaxIter > 0 ? MaxIter : sizeof( T1 );
    constexpr uint32 shiftBase = 8;
    uint32 shift = 0;

    BucketT counts     [Radix];
    BucketT pfxSum     [Radix];
    BucketT totalCounts[Radix];

    T* input  = entries;
    T* output = tmpEntries;

    // #TODO: Use real value here
    // const uint32 remainderBits = 2;////CDiv( entryBitSize, 8 ) * 8 - entryBitSize;
    const uint32 lastByteMask  = 0xFF >> remainderBits;
    uint32 masks[iterations] = { 0xFF, 0xFF, 0xFF, lastByteMask };

    for( int32 iter = 0; iter < iterations ; iter++, shift += shiftBase )
    {
        const uint32 mask = masks[iter];

        // Zero-out the counts
        memset( counts, 0, sizeof( BucketT ) * Radix );

        T*       src   = input + offset;
        const T* start = src;
        const T* end   = start + entryCount;

        do {
            counts[(*((uint32*)src) >> shift) & mask]++;
        } while( ++src < end );
        
        self->CalculatePrefixSum( Radix, counts, pfxSum, totalCounts );

        while( --src >= start )
        {
            const T       value  = *src;
            const uint32  bucket = ( *(uint32*)&value >> shift ) & mask;
            const BucketT dstIdx = --pfxSum[bucket];
            
            output[dstIdx] = value;
        }

        std::swap( input, output );
        self->SyncThreads();
    }
}

//-----------------------------------------------------------
// Expand entries from bit-compressed form into 8-byte aligned entries
// The entry components remain together, but each entry
// will start at an 8-byte boundary
//-----------------------------------------------------------
template<TableId table>
void FxExpandEntries( const uint32 numBuckets, ThreadPool& pool, const uint32 threadCount, void* input, uint64* output, const int64 entryCount )
{
    AnonMTJob::Run( pool, threadCount, [=]( AnonMTJob* self ){
        
        const uint32 jobId         = self->JobId();
        const uint32 threadCount   = self->JobCount();

        const uint32 ySize            = GetYBitsForBuckets( numBuckets );
        const uint32 mapSize          = _K + 1;
        const uint32 metaSize         = TableMetaOut<table>::Multiplier * _K;

        const size_t entrySizeBits    = ySize + mapSize + metaSize;
        const size_t unpackedSize64   = CDiv( entrySizeBits, 64 );
        
        int64 entriesPerThread        = entryCount / threadCount;
        const uint64 inputOffsetBits  = entrySizeBits * entriesPerThread * jobId;

        uint64* out = output + entriesPerThread * unpackedSize64 * jobId;

        BitReader reader( (uint64*)input, entrySizeBits * (uint64)entryCount, inputOffsetBits );

        if( self->IsLastThread() )
            entriesPerThread += entryCount - entriesPerThread * threadCount;
        
        for( int64 i = 0; i < entriesPerThread; i++ )
        {
            const uint64 y     = reader.ReadBits64( ySize    );
            const uint64 map   = reader.ReadBits64( mapSize  );
            const uint64 metaA = reader.ReadBits64( metaSize );

            out[0] = y | ( map << ySize );
            out[1] = metaA;

            out += unpackedSize64;
        }
    });
}


