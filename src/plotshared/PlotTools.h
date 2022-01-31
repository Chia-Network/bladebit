#pragma once

#define BB_PLOT_ID_LEN 32
#define BB_PLOT_ID_HEX_LEN (BB_PLOT_ID_LEN * 2)

#define BB_PLOT_MEMO_MAX_SIZE (48+48+32)

#define BB_PLOT_FILE_LEN_TMP (sizeof( "plot-k32-2021-08-05-18-55-77a011fc20f0003c3adcc739b615041ae56351a22b690fd854ccb6726e5f43b7.plot.tmp" ) - 1)
#define BB_PLOT_FILE_LEN (BB_PLOT_FILE_LEN_TMP - 4)

struct PlotTools
{
    static void GenPlotFileName( const byte plotId[BB_PLOT_ID_LEN], char outPlotFileName[BB_PLOT_FILE_LEN] );
    static void PlotIdToString   ( const byte plotId[BB_PLOT_ID_LEN], char plotIdString[BB_PLOT_ID_HEX_LEN+1] );
    // static void PlotIdToStringTmp( const byte* plotId, const byte plotIdString[BB_PLOT_FILE_LEN_TMP] );
};

