#include "util/Util.h"
#include "util/Log.h"

//-----------------------------------------------------------
void FatalExit()
{
    exit( 1 );
}

//-----------------------------------------------------------
void VFatalErrorMsg( const char* message, va_list args  )
{
    Log::Flush();
    Log::FlushError();

    Log::Error( "\nFatal Error:  " );
    Log::Error( message, args );
    Log::FlushError();
}

//-----------------------------------------------------------
void FatalErrorMsg( const char* message, ... )
{
    va_list args;
    va_start( args, message );
    VFatalErrorMsg( message, args );
    va_end( args );
}

//-----------------------------------------------------------
void VFatal( const char* message, va_list args )
{
    VFatalErrorMsg( message, args );
    FatalExit();
}

//-----------------------------------------------------------
void _Fatal( const char* message, ... )
{
    va_list args;
    va_start( args, message );
    VFatal( message, args );
    va_end( args );
}

//-----------------------------------------------------------
void _FatalIf( bool condition, const char* message, ... )
{
    if( condition )
    {
        va_list args;
        va_start( args, message );
        VFatal( message, args );
        va_end( args );
    }
}

//-----------------------------------------------------------
bool AssertLog( int line, const char* file, const char* func )
{
    Log::Error( "Assertion Failed @ %s:%d %s().", file, line, func );
    Log::FlushError();
    return true;
}