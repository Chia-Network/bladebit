#include "Log.h"
#include <iostream>
#include <chrono>
#include <ctime>

#if _DEBUG && defined( _WIN32 )
    #include <Windows.h>
    #include <debugapi.h>
#endif

FILE* Log::_outStream = nullptr;
FILE* Log::_errStream = nullptr;

bool Log::_verbose = false;

// #if DBG_LOG_ENABLE
    std::atomic<int> _dbglock = 0;
// #endif

//-----------------------------------------------------------
inline FILE* Log::GetOutStream()
{
    if( _outStream == nullptr )
    {
        _outStream = stdout;
        setvbuf( stdout, NULL, _IONBF, 0 );
    }

    return _outStream;
}

//-----------------------------------------------------------
inline FILE* Log::GetErrStream()
{
    if( _errStream == nullptr )
    {
        _errStream = stderr;
        setvbuf( stderr, NULL, _IONBF, 0 );
    }

    return _errStream;
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

#if _DEBUG && defined( _WIN32 )
    //::OutputDebugStringA( (LPCSTR)msg );
#endif
}


//-----------------------------------------------------------
void Log::WriteLine( const char* msg, va_list args )
{
    FILE* stream = GetOutStream();
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
    char timestamp[80];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now_time_t));
    std::string current_time_str(timestamp);

    fprintf(stream, "[%s] ", timestamp);
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
    fputc( '\n', GetErrStream() );
}

//-----------------------------------------------------------
void Log::WriteError( const char* msg, va_list args )
{
    vfprintf( GetErrStream(), msg, args );
    
#if _DEBUG && defined( _WIN32 )
    //::OutputDebugStringA( (LPCSTR)msg );
#endif
}

//-----------------------------------------------------------
void Log::Verbose( const char* msg, ...  )
{
    if( !_verbose )
        return;
    
    va_list args;
    va_start( args, msg );
    
    FILE* stream = GetErrStream();
    vfprintf( stream, msg, args );
    fputc( '\n', stream );

    va_end( args );
}

//-----------------------------------------------------------
void Log::VerboseWrite( const char* msg, ...  )
{
    if( !_verbose )
        return;

    va_list args;
    va_start( args, msg );
    
    vfprintf( GetErrStream(), msg, args );

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
    fflush( GetErrStream() );
}

//-----------------------------------------------------------
void Log::NewLine()
{
    #if _WIN32
        Log::Write( "\r\n" );
    #else
        Log::Write( "\n" );
    #endif
}


#if DBG_LOG_ENABLE

//-----------------------------------------------------------
void Log::Debug( const char* msg, ... )
{
    va_list args;
    va_start( args, msg );

    DebugV( msg, args );

    va_end( args );
}


//-----------------------------------------------------------
void Log::DebugV( const char* msg, va_list args )
{
    const size_t BUF_SIZE = 1024;
    char buffer[BUF_SIZE];

    int count = vsnprintf( buffer, BUF_SIZE, msg, args );

    ASSERT( count >= 0 );
    count = std::min( count, (int)BUF_SIZE-1 );

    buffer[count] = '\n'; 

    DebugWrite( buffer, (size_t)count + 1 );
}

//-----------------------------------------------------------
void Log::DebugWrite( const char* msg, size_t size )
{
    SafeWrite( msg, size );
}

#endif

//-----------------------------------------------------------
void Log::SafeWrite( const char* msg, size_t size )
{
    // #TODO: Just use a futex
    // Lock
    int lock = 0;
    while( !_dbglock.compare_exchange_weak( lock, 1 ) )
        lock = 0;
    
    fwrite( msg, 1, size, stderr );
    fflush( stderr );

    // Unlock
    _dbglock.store( 0, std::memory_order_release );
}
