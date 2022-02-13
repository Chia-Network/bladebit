#include "PlotTools.h"
#include "util/Log.h"
#include <string.h>


bool ValidatePlot( const ValidatePlotOptions& options );

int ValidatePlotCmd( int argc, const char* argv[] );

inline bool match( const char* ref, const char* arg )
{
    return strcmp( ref, arg ) == 0;
}

inline bool match( const char* ref0, const char* ref1, const char* arg )
{
    return strcmp( ref0, arg ) == 0 ||
           strcmp( ref1, arg ) == 0;
}


//-----------------------------------------------------------
int main( int argc, const char* argv[] )
{
    argc--;
    argv++;

    for( int i = 0; i < argc; i++ )
    {
        const char* arg = argv[i];

        if( match( "validate", arg ) )
            return ValidatePlotCmd( argc-1, argv+i+1 );
    }

    Log::Line( "No command specified." );
    return 1;
}

//-----------------------------------------------------------
int ValidatePlotCmd( int argc, const char* argv[] )
{
    ValidatePlotOptions opts;

    for( int i = 0; i < argc; i++ )
    {
        const char* arg = argv[i];

        if( match( "-m", "--in-ram", arg ) )
            opts.inRAM = true;
        else if( i == argc - 1 )
            opts.plotPath = arg;
        else
        {
            Log::Error( "Unknown argument '%s'", arg );
            return 1;
        }
    }

    // #TODO: Allow many plots to be validated
    return ValidatePlot( opts ) ? 0 : 1;
}


