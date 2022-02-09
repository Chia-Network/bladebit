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

    uint64* f7Entries = bbcalloc<uint64>( kCheckpoint1Interval );
    memset( f7Entries, 0, kCheckpoint1Interval * sizeof( uint64 ) );

    // Read the C3 parks
    FatalIf( !plot.ReadC3Park( 0, f7Entries ),
        "Could not read C3 park." );
    
}
