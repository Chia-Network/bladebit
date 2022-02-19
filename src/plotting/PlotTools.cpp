#include "PlotTools.h"
#include "util/Util.h"


#define PLOT_FILE_PREFIX_LEN (sizeof("plot-k32-2021-08-05-18-55-")-1)

//-----------------------------------------------------------
void PlotTools::GenPlotFileName( const byte plotId[BB_PLOT_ID_LEN], char outPlotFileName[BB_PLOT_FILE_LEN] )
{
    ASSERT( plotId );
    ASSERT( outPlotFileName );

    time_t     now = time( nullptr );
    struct tm* t   = localtime( &now ); ASSERT( t );
    
    const size_t r = strftime( outPlotFileName, BB_PLOT_FILE_LEN, "plot-k32-%Y-%m-%d-%H-%M-", t );
    if( r != PLOT_FILE_PREFIX_LEN )
        Fatal( "Failed to generate plot file." );

    PlotIdToString( plotId, outPlotFileName + r );
    memcpy( outPlotFileName + r + BB_PLOT_ID_HEX_LEN, ".plot.tmp", sizeof( ".plot.tmp" ) );
}

//-----------------------------------------------------------
void PlotTools::PlotIdToString( const byte plotId[BB_PLOT_ID_LEN], char plotIdString[BB_PLOT_ID_HEX_LEN+1] )
{
    ASSERT( plotId );
    ASSERT( plotIdString );

    size_t numEncoded = 0;
    BytesToHexStr( plotId, BB_PLOT_ID_LEN, plotIdString, BB_PLOT_ID_HEX_LEN, numEncoded );
    ASSERT( numEncoded == BB_PLOT_ID_LEN );

    plotIdString[BB_PLOT_ID_HEX_LEN] = '\0';
}

