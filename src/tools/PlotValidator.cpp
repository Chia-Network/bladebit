#include "memplot/CTables.h"
#include "ChiaConsts.h"
#include "util/Log.h"
#include "util/BitView.h"
#include "io/FileStream.h"
#include "PlotReader.h"
#include "plotshared/PlotTools.h"

std::string plotPath = "/mnt/p5510a/plots/plot-k32-2022-02-07-03-50-c6b84729c23dc6d60c92f22c17083f47845c1179227c5509f07a5d2804a7b835.plot";

//-----------------------------------------------------------
void TestPlotValidate()
{
    MemoryPlot plotFile;
    FatalIf( !plotFile.Open( plotPath.c_str() ), 
        "Failed to open plot at path '%s'.", plotPath.c_str() );

    PlotReader plot( plotFile );

    uint64 numLinePoint = 0;
    uint128* linePoints = bbcalloc<uint128>( kEntriesPerPark );
    plot.ReadLPPark( PlotTable::Table6, 0, linePoints, numLinePoint );

    // Test read C3 park
    uint64* f7Entries = bbcalloc<uint64>( kCheckpoint1Interval );
    memset( f7Entries, 0, kCheckpoint1Interval * sizeof( uint64 ) );

    // Check how many C3 parks we have
    const uint64 c3ParkCount = plotFile.TableSize( PlotTable::C1 ) / sizeof( uint32 ) - 1;

    // Read the C3 parks
    FatalIf( !plot.ReadC3Park( 0, f7Entries ),
        "Could not read C3 park." );

    // Test read p7
    uint64* p7Entries = bbcalloc<uint64>( kEntriesPerPark );
    memset( p7Entries, 0, sizeof( kEntriesPerPark ) * sizeof( uint64 ) );

    FatalIf( !plot.ReadP7Entries( 0, p7Entries ),
        "Failed to read park 7 entries." );

    ASSERT( p7Entries[0] == 3208650999 );
}
