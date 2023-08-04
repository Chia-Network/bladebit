#include "TestUtil.h"
#include "plotmem/LPGen.h"
#include "plotdisk/jobs/IOJob.h"
#include "plotdisk/DiskPlotConfig.h"
#include "plotting/PlotTools.h"
#include "plotting/FSETableGenerator.h"
#include "plotmem/ParkWriter.h"

struct LpData
{
    uint64 deltaMax;
    uint64 deltaMin;
    uint64 deltaMaxPark;
    uint64 deltaMinPark;
    uint64 deltaBitsMax;
    uint64 deltaBitsMin;
    uint64 deltaTotalBits;
    uint64 overflowCount;
};

struct ParkData
{
    size_t totalParkSize;
    size_t parkSizeMax;
    size_t parkSizeMin;
};

static Span<uint64> LoadLpTableForCompressionLevel( uint compressionLevel );
static void DumpLpData( Span<uint64> linePoints, uint32 compressionLevel, uint32 stubBitSize, LpData& outLpData );
static uint32 CountBits( uint64 sm );
static void CalculateParkSizes( const Span<uint64> linePoints, const uint32 stubBitSize, const FSE_CTable* cTable, const size_t parkBufferSize, ParkData& parkData );

ThreadPool* _pool = nullptr;


//-----------------------------------------------------------
TEST_CASE( "line-point-deltas", "[sandbox]" )
{
    _pool = new ThreadPool( SysHost::GetLogicalCPUCount() );

    const size_t defaultParkSize   = CalculateParkSize( TableId::Table1 );
    const uint32 defaultSutbBits   = 32 - kStubMinusBits;
    const size_t defaultStubBytes  = CDiv( (kEntriesPerPark-1) * defaultSutbBits, 8 );
    const size_t defaultDeltasSize = defaultParkSize - defaultStubBytes;

    // byte* parkBuffer = (byte*)malloc( defaultParkSize * 4 );

          uint32 cLevel    = GetEnvU32( "bb_clevel", 1 );
    const uint32 endCLevel = GetEnvU32( "bb_end_clevel", 9 );
    const uint32 k         = 32;

    for( ; cLevel <= endCLevel; cLevel++ )
    {
        Log::Line( "[Level %u]", cLevel );
        Log::Line( " Loading line points" );
        
        Span<uint64> linePoints = LoadLpTableForCompressionLevel( cLevel );

        // Convert to park-specific deltas

        LpData lpData = {};

        const uint32 xBitSize     = (17 - cLevel);
        const uint32 entryBitSize = xBitSize * 2;
              uint32 stubBitSize  = k-1;    // Delta average should be k size, so remove 1 bit for small delta

        double rValue = 0;

        Log::Line( " Calculating for entries of %u bits", entryBitSize/2 );
        for( ;; )
        {
            DumpLpData( linePoints, cLevel, stubBitSize, lpData );

            const double averageBits = lpData.deltaTotalBits / double(linePoints.Length()-1);
            Log::Line( " [Stub bit size: %u]", stubBitSize );
            Log::Line( "  Average delta bit count : %.2lf", averageBits          );
            Log::Line( "  Max delta bits          : %llu" , lpData.deltaBitsMax  );
            Log::Line( "  Min delta bits          : %llu" , lpData.deltaBitsMin  );
            Log::Line( "  Max delta value         : %llu" , lpData.deltaMax      );
            Log::Line( "  Min delta value         : %llu" , lpData.deltaMin      );
            Log::Line( "  Max delta park          : %llu" , lpData.deltaMaxPark  );
            Log::Line( "  Min delta park          : %llu" , lpData.deltaMinPark  );
            Log::Line( "  N overflowed 256-bits   : %llu" , lpData.overflowCount );
            Log::WriteLine( "" );

            if( lpData.overflowCount > 0 )
            {
                stubBitSize++;
                break;
            }

            stubBitSize--;
            rValue = lpData.deltaTotalBits / double(linePoints.Length()-1); 
        }


        // Now try to determine a more reasonable ANS encoding value
        Log::NewLine();
        Log::Line( " Selected sub bit size: %u", stubBitSize );
        Log::NewLine();
        Log::Line( " [Calculating ANS value]" );

        size_t deltasSize = defaultDeltasSize;


        const size_t stubBytes = CDiv( (kEntriesPerPark-1) * stubBitSize, 8 );
        size_t currentParkSize  = stubBytes + defaultDeltasSize;
        size_t smallestParkSize = currentParkSize;

        bool wentLower  = false;
        bool wentHigher = false;

        // uint64 parkEntries[kEntriesPerPark];

        const uint64 totalParkCount = CDiv( (1ull << 32), kEntriesPerPark );
        ParkData parkData;

        uint64 end = 0xFFFFFFFFFFFFFFFFull;
        for( uint64 i = 0 ; i < end; i++ )
        {
            rValue += 0.1;

            FSE_CTable* cTable = FSETableGenerator::GenFSECompressionTable( rValue );
            ASSERT( cTable );

            Log::Line( "  [R value: %.2lf]", rValue );
            CalculateParkSizes( linePoints, stubBitSize, cTable, defaultParkSize, parkData );

            FSE_freeCTable( cTable );

            const size_t averageParkSize = (size_t)std::ceil( parkData.totalParkSize / (double)totalParkCount );

            const uint64 parkDeltaMin = parkData.parkSizeMin - stubBytes;
            const uint64 parkDeltaMax = parkData.parkSizeMax - stubBytes;

            Log::Line( "   Average park size in bytes: %llu", averageParkSize );
            Log::Line( "   Minimum park size         : %llu", parkData.parkSizeMin );
            Log::Line( "   Maximum park size         : %llu", parkData.parkSizeMax );
            Log::Line( "   Min deltas size           : %llu", parkDeltaMin );
            Log::Line( "   Max deltas size           : %llu", parkDeltaMax );
            Log::Line( "" );

            // const size_t parkSize = WritePark( defaultParkSize*4, kEntriesPerPark, parkEntries, parkBuffer, stubBitSize, cTable );

            if( parkData.parkSizeMax < smallestParkSize && !wentLower )
            {
                wentLower = true;
                continue;
            }
            
            // If we already gone below the default threshhold, and 
            // we got a park size greater than our smallest one, then we can break
            if( !wentHigher && wentLower && parkData.parkSizeMax > smallestParkSize )
            {
                Log::Line( "*** Lowest reached ***" );
                // Do 6 more and exit
                end = i + 7;
                wentHigher = true;
            }

            smallestParkSize = std::min( smallestParkSize, parkData.parkSizeMax );
            // deltasSize      = parkData.parkSizeMax - stubBytes;
            // break;
        }

        bbvirtfree( linePoints.Ptr() );
        Log::Line( "" );
    }
}

//-----------------------------------------------------------
void CalculateParkSizes( const Span<uint64> linePoints, const uint32 stubBitSize, const FSE_CTable* cTable, const size_t parkBufferSize, ParkData& parkData )
{
    const uint64 parkCount = (uint64)CDiv( linePoints.Length(), kEntriesPerPark );

    ParkData parkDatas[BB_MAX_JOBS] = {};

    AnonMTJob::Run( *_pool, [&]( AnonMTJob* self ){
        
        uint64 count, offset, end;
        GetThreadOffsets( self, parkCount, count, offset, end );

        const uint64 sliceEnd = self->IsLastThread() ? linePoints.length : end * kEntriesPerPark;

        Span<uint64> entries = linePoints.Slice( offset * kEntriesPerPark, sliceEnd - offset * kEntriesPerPark );

        uint64 parkEntries[kEntriesPerPark];
        byte*  parkBuffer = (byte*)malloc( parkBufferSize*4 );

        ParkData& data = parkDatas[self->_jobId];
        data.parkSizeMax   = 0;
        data.parkSizeMin   = 0xFFFFFFFFFFFFFFFFull;
        data.totalParkSize = 0;
        
        while( entries.Length() > 0 )
        {
            const uint64 entryCount = std::min( entries.Length(), (size_t)kEntriesPerPark );
            entries.SliceSize( entryCount ).CopyTo( Span<uint64>( parkEntries, entryCount ) );
            
            const size_t parkSize = WritePark( parkBufferSize*4, entryCount, parkEntries, parkBuffer, stubBitSize, cTable );

            data.parkSizeMax   = std::max( data.parkSizeMax, parkSize );
            data.parkSizeMin   = std::min( data.parkSizeMin, parkSize );
            data.totalParkSize += parkSize;

            entries = entries.Slice( entryCount );
        }
        
        free( parkBuffer );
    });

    ZeroMem( &parkData );
    parkData.parkSizeMin = 0xFFFFFFFFFFFFFFFFull;

    for( uint32 i = 0; i < _pool->ThreadCount(); i++ )
    {
        parkData.parkSizeMax   = std::max( parkData.parkSizeMax, parkDatas[i].parkSizeMax );
        parkData.parkSizeMin   = std::min( parkData.parkSizeMin, parkDatas[i].parkSizeMin );
        parkData.totalParkSize += parkDatas[i].totalParkSize;
    }
}

//-----------------------------------------------------------
void DumpLpData( Span<uint64> linePoints, const uint32 compressionLevel, const uint32 stubBitSize, LpData& outLpData )
{
    struct Job : MTJob<Job>
    {
        Span<uint64> linePoints;
        uint32       stubBitSize;

        uint64 deltaMax;
        uint64 deltaMin;
        uint64 overflowCount;
        uint64 deltaMaxPark;
        uint64 deltaMinPark;
        uint64 deltaBitsMax;
        uint64 deltaBitsMin;
        uint64 deltaTotalBits;
        
        inline void Run() override
        {
            deltaMax       = 0;
            deltaMin       = 0xFFFFFFFFFFFFFFFFull;
            deltaMaxPark   = 0;
            deltaMinPark   = 0xFFFFFFFFFFFFFFFFull;;
            deltaBitsMax   = 0;
            deltaBitsMin   = 0xFFFFFFFFFFFFFFFFull;
            deltaTotalBits = 0;
            overflowCount  = 0;

            const uint32 id         = this->JobId();
            Span<uint64> linePoints = this->linePoints;

            const uint64 totalParkCount = (uint64)CDiv( linePoints.Length(), kEntriesPerPark );

            uint64 parkCount, parkOffset, parkEnd;
            GetThreadOffsets( this, totalParkCount, parkCount, parkOffset, parkEnd );

            const uint64 entryStart = parkOffset * kEntriesPerPark;
            const uint64 entryEnd   = std::min( entryStart + parkCount * kEntriesPerPark, (uint64)linePoints.Length() );

            linePoints = linePoints.Slice( entryStart, entryEnd - entryStart );

            // Deltafy
            for( uint64 park = parkOffset; park < parkEnd; park++ )
            {
                const uint64 parkEntryCount = std::min( linePoints.Length(), (size_t)kEntriesPerPark );
                
                uint64 prevLp = linePoints[0];

                for( uint64 i = 1; i < parkEntryCount; i++ )
                {
                    const uint64 lp    = linePoints[i]; ASSERT( lp >= prevLp );
                    const uint64 delta = lp - prevLp;

                    // Remove stub (convert to small deltas)
                    const uint64 smallDelta = delta >> stubBitSize;

                    if( smallDelta >= 256 )
                        overflowCount++;

                    const uint64 deltaBits = CountBits( smallDelta );
                    deltaTotalBits += deltaBits;

                    if( smallDelta > deltaMax )
                        deltaMaxPark = park;
                    if( smallDelta < deltaMin )
                        deltaMinPark = park;

                    deltaMax     = std::max( deltaMax    , smallDelta );
                    deltaMin     = std::min( deltaMin    , smallDelta );
                    deltaBitsMax = std::max( deltaBitsMax, deltaBits  );
                    deltaBitsMin = std::min( deltaBitsMin, deltaBits  );

                    prevLp = lp;
                }

                linePoints = linePoints.Slice( parkEntryCount );
            }
        }
    };

    const uint32 threadCount = _pool->ThreadCount();
    MTJobRunner<Job> jobs( *_pool );
    
    for( uint32 i = 0; i < threadCount; i++ )
    {
        jobs[i].linePoints  = linePoints;
        jobs[i].stubBitSize = stubBitSize;
    }
    jobs.Run( threadCount );

    {
        uint64 deltaMax       = 0;
        uint64 deltaMin       = 0xFFFFFFFFFFFFFFFFull;
        uint64 deltaMaxPark   = 0;
        uint64 deltaMinPark   = 0xFFFFFFFFFFFFFFFFull;;
        uint64 deltaBitsMax   = 0;
        uint64 deltaBitsMin   = 0xFFFFFFFFFFFFFFFFull;
        uint64 deltaTotalBits = 0;
        uint64 overflowCount  = 0;

        for( uint32 i = 0; i < threadCount; i++ )
        {
            deltaMax       = std::max( deltaMax    , jobs[i].deltaMax     );
            deltaMin       = std::min( deltaMin    , jobs[i].deltaMin     ); 
            deltaMaxPark   = std::max( deltaMaxPark, jobs[i].deltaMaxPark );
            deltaMinPark   = std::min( deltaMinPark, jobs[i].deltaMinPark );
            deltaBitsMax   = std::max( deltaBitsMax, jobs[i].deltaBitsMax );
            deltaBitsMin   = std::min( deltaBitsMin, jobs[i].deltaBitsMin );
            deltaTotalBits += jobs[i].deltaTotalBits;
            overflowCount  += jobs[i].overflowCount;
        }
        
        outLpData.deltaMax       = deltaMax;
        outLpData.deltaMin       = deltaMin;
        outLpData.deltaMaxPark   = deltaMaxPark;
        outLpData.deltaMinPark   = deltaMinPark;
        outLpData.deltaBitsMax   = deltaBitsMax;
        outLpData.deltaBitsMin   = deltaBitsMin;
        outLpData.deltaTotalBits = deltaTotalBits;
        outLpData.overflowCount  = overflowCount;
    }
}

//-----------------------------------------------------------
Span<uint64> LoadLpTableForCompressionLevel( const uint compressionLevel )
{
    char filePath[1024] = {};

    // if( compressionLevel < 9 )
    //     sprintf( filePath, "%st2.lp.c%u.ref", BB_DP_DBG_REF_DIR "compressed-lps/", compressionLevel );
    // else
        sprintf( filePath, "%slp.c%u.ref", BB_DP_DBG_REF_DIR "compressed-lps/", compressionLevel );

    size_t byteCount = 0;
    int err = 0;
    uint64* linePoints = (uint64*)IOJob::ReadAllBytesDirect( filePath, err, byteCount );
    
    FatalIf( !linePoints, "Failed to load line points from '%s' with error %d", filePath, err );

    // Get the line point count
    int64 count = (int64)(byteCount / sizeof( uint64 ));
    for( int64 i = count-2; i >= 0; i-- )
    {
        if( linePoints[i] > linePoints[i+1] )
        {
            count = i+1;
            break;
        }
    }

    return Span<uint64>( linePoints, (size_t)count );
}

//-----------------------------------------------------------
inline uint32 CountBits( uint64 sm )
{
    for( int32 i = 63; i >= 0; i-- )
    {
        if( sm >> i )
            return (uint32)i+1;
    }

    return 0;
}