#include "PlotTools.h"
#include "util/Log.h"
#include "util/Util.h"
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

    int i;
    const char* arg;
    auto value = [&](){

        FatalIf( ++i >= argc, "Expected a value for parameter '%s'", arg );
        return argv[i];
    };

    auto ivalue = [&]() {

        const char* val = value();
        int64 v = 0;
        
        int r = sscanf( val, "%lld", &v );
        FatalIf( r != 1, "Invalid int64 value for argument '%s'.", arg );

        return v;
    };

    auto uvalue = [&]() {
        
        const char* val = value();
        uint64 v = 0;

        int r = sscanf( val, "%llu", &v );
        FatalIf( r != 1, "Invalid uint64 value for argument '%s'.", arg );

        return v;
    };

    auto fvalue = [&]() {
        
        const char* val = value();
        float v = 0.f;

        int r = sscanf( val, "%f", &v );
        FatalIf( r != 1, "Invalid float32 value for argument '%s'.", arg );

        return v;
    };

    for( i = 0; i < argc; i++ )
    {
        arg = argv[i];

        if( match( "-m", "--in-ram", arg ) )
            opts.inRAM = true;
        else if( match( "-t", "--threads", arg ) )
            opts.threadCount = uvalue();
        else if( match( "-o", "--offset", arg ) )
            opts.startOffset = std::max( std::min( fvalue() / 100.f, 100.f ), 0.f );
        else if( i == argc - 1 )
            opts.plotPath = arg;
        else
        {
            Log::Error( "Unknown argument '%s'", arg );
            return 1;
        }
    }

    if( opts.threadCount == 0 )
        opts.threadCount = SysHost::GetLogicalCPUCount();
    else
        opts.threadCount = std::min( opts.threadCount, SysHost::GetLogicalCPUCount() );

    // #TODO: Allow many plots to be validated
    return ValidatePlot( opts ) ? 0 : 1;
}


