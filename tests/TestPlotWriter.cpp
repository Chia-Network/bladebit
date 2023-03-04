#include "TestUtil.h"
#include "plotdisk/jobs/IOJob.h"
#include "plotting/PlotWriter.h"
#include "tools/PlotReader.h"

#define PLOT_FILE_DIR "/home/harold/plot/tmp/"
#define PLOT_FILE_NAME "tester.plot.tmp"
#define PLOT_FILE_PATH PLOT_FILE_DIR PLOT_FILE_NAME

template<typename T>
inline T RandRange( T min, T max )
{
    const T      range = max - min;
    const double r     = std::rand() / (double)RAND_MAX;
    return min + (T)(range * r);
}

//-----------------------------------------------------------
TEST_CASE( "plot-writer", "[sandbox]" )
{
    PlotWriter writer;

    byte plotId[32];

    const size_t blockSize  = 4096;
    const size_t totalCount = ( 6ull GiB ) / sizeof( uint32 );
    // const size_t nBlocks    = CDiv( totalCount, blockSize / sizeof( uint32 ) );

    uint32* data = bbcvirtallocbounded<uint32>( totalCount );//  new uint32[totalCount/sizeof( uint32 )];

    const uint32 TEST_ROUNDS = 64;


    for( uint32 round = 0; round < TEST_ROUNDS; round++ )
    {
        const size_t c1Min  = 1024*27 + 115, c1Max = 1024*30 + 357;
        const size_t c2Min  = 21,      c2Max = 57;
        const size_t c3Base = RandRange( 100, 500 );

        // const size_t c1Length = 30504;
        // const size_t c2Length = 36;
        const size_t c1Length = RandRange( c1Min, c1Max );
        const size_t c2Length = RandRange( c2Min, c2Max );
        const size_t c3Length = totalCount - ( c1Length + c2Length );

        Log::Line( "Testing round %u:", round    );
        Log::Line( " C1 Length: %llu" , c1Length );
        Log::Line( " C2 Length: %llu" , c2Length );
        Log::Line( " C3 Length: %llu" , c3Length );
        Log::Line( " C3 Base  : %llu" , c3Base );

        for( size_t i = 0; i < c1Length+c2Length; i++ )
            data[i] = (uint32)(i+1);

        for( size_t i = c1Length+c2Length, j = 0; i < totalCount; i++, j++ )
            data[i] = (uint32)(c3Base + j);

        ASSERT( writer.BeginPlot( PlotVersion::v1_0, PLOT_FILE_DIR, PLOT_FILE_NAME, plotId, plotId, sizeof( plotId ), 0  ) );
        writer.ReserveTableSize( PlotTable::C1, c1Length * sizeof( uint32 ) );
        writer.ReserveTableSize( PlotTable::C2, c2Length * sizeof( uint32 ) );
        writer.BeginTable( PlotTable::C3 );
        writer.WriteTableData( data+c1Length+c2Length, c3Length * sizeof( uint32 ) );
        writer.EndTable();
        writer.WriteReservedTable( PlotTable::C1, data );
        writer.WriteReservedTable( PlotTable::C2, data + c1Length );
        writer.EndPlot( false );

        int err;
        const uint32* plotData = (uint32*)IOJob::ReadAllBytesDirect( PLOT_FILE_PATH, err ) + 1024;
        ASSERT( plotData );

        Log::Line( " Validating..." );
        FilePlot plot;
        plot.Open( PLOT_FILE_PATH );
        ASSERT( plot.IsOpen() );
        const size_t c1Address = plot.TableAddress( PlotTable::C1 );
        const size_t c2Address = plot.TableAddress( PlotTable::C2 );
        const size_t c3Address = plot.TableAddress( PlotTable::C3 );

        Log::Line( " C1: %lu ( 0x%016lx ) )", c1Address, c1Address );
        Log::Line( " C2: %lu ( 0x%016lx ) )", c2Address, c2Address );
        Log::Line( " C3: %lu ( 0x%016lx ) )", c3Address, c3Address );

        ASSERT( c1Address == blockSize );
        ASSERT( c2Address == c1Address + c1Length * 4 );
        ASSERT( c3Address == c1Address + (c1Length + c2Length) * 4 );

        for( size_t i = c1Length+c2Length, j = 0; i < totalCount; i++, j++ )
            ASSERT( plotData[i] == c3Base + j );

        Log::Line( " OK!" );
        Log::Line( "" );
    }

}
