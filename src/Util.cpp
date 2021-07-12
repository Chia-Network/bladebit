#include "Util.h"
#include "util/Log.h"

//-----------------------------------------------------------
void Fatal( const char* message, ... )
{
    va_list args;
    va_start( args, message );

    Log::Error( "Fatal Error:" );
    Log::WriteError( "  " );
    Log::Error( message, args );
    Log::FlushError();
    va_end( args );

    ASSERT( 0 );
    exit( 1 );
}