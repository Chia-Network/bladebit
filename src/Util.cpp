#include "Util.h"
#include "util/Log.h"

//-----------------------------------------------------------
void VFatal( const char* message, va_list args )
{
    Log::Error( "Fatal Error:" );
    Log::WriteError( "  " );
    Log::Error( message, args );
    Log::FlushError();

    ASSERT( 0 );
    exit( 1 );
}

//-----------------------------------------------------------
void Fatal( const char* message, ... )
{
    va_list args;
    va_start( args, message );
    VFatal( message, args );
    va_end( args );
}

//-----------------------------------------------------------
void FatalIf( bool condition, const char* message, ... )
{
    if( condition )
    {
        va_list args;
        va_start( args, message );
        VFatal( message, args );
        va_end( args );
    }
}