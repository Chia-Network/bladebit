#include "Log.h"


FILE* Log::_outStream = nullptr;

bool Log::_verbose = false;

//-----------------------------------------------------------
inline FILE* Log::GetOutStream()
{
    if( _outStream == nullptr )
    {
        _outStream = stdout;
        setvbuf(stdout, NULL, _IONBF, 0);
    }

    return _outStream;
}

//-----------------------------------------------------------
void Log::Write( const char* msg, ... )
{
    va_list args;
    va_start( args, msg );
    
    Write( msg, args );

    va_end( args );
}

//-----------------------------------------------------------
void Log::WriteLine( const char* msg, ... )
{
    va_list args;
    va_start( args, msg );
    
    WriteLine( msg, args );

    va_end( args );
}

//-----------------------------------------------------------
void Log::Line( const char* msg, ... )
{
    va_list args;
    va_start( args, msg );
    
    WriteLine( msg, args );

    va_end( args );
}

//-----------------------------------------------------------
void Log::Write( const char* msg, va_list args )
{
    vfprintf( GetOutStream(), msg, args );
}

//-----------------------------------------------------------
void Log::WriteLine( const char* msg, va_list args )
{
    FILE* stream = GetOutStream();
    vfprintf( stream, msg, args );
    fputc( '\n', stream );
}

//-----------------------------------------------------------
void Log::Error( const char* msg, ... )
{
    va_list args;
    va_start( args, msg );
    
    Error( msg, args );

    va_end( args );
}

//-----------------------------------------------------------
void Log::WriteError( const char* msg, ... )
{
    va_list args;
    va_start( args, msg );
    
    WriteError( msg, args );

    va_end( args );
}

//-----------------------------------------------------------
void Log::Error( const char* msg, va_list args )
{
    WriteError( msg, args );
    fputc( '\n', stderr );
}

//-----------------------------------------------------------
void Log::WriteError( const char* msg, va_list args )
{
    vfprintf( stderr, msg, args );
}

//-----------------------------------------------------------
void Log::Verbose( const char* msg, ...  )
{
    if( !_verbose )
        return;
    
    va_list args;
    va_start( args, msg );
    
    vfprintf( stderr, msg, args );
    fputc( '\n', stderr );

    va_end( args );
}

//-----------------------------------------------------------
void Log::VerboseWrite( const char* msg, ...  )
{
    if( !_verbose )
        return;

    va_list args;
    va_start( args, msg );
    
    vfprintf( stderr, msg, args );

    va_end( args );
}


//-----------------------------------------------------------
void Log::Flush()
{
    fflush( GetOutStream() );
}


//-----------------------------------------------------------
void Log::FlushError()
{
    fflush( stderr );
}
