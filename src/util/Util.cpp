#include "util/Util.h"
#include "util/Log.h"

//-----------------------------------------------------------
void Exit( int code )
{
    exit( code );
}

//-----------------------------------------------------------
void FatalExit()
{
    Exit( 1 );
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
void PanicExit()
{
    SysHost::DumpStackTrace();
    Exit( 1 );
}

//-----------------------------------------------------------
void VPanicErrorMsg( const char* message, va_list args  )
{
    Log::Flush();
    Log::FlushError();

    Log::Error( "\n*** Panic!!! *** Fatal Error:  " );
    Log::Error( message, args );
    Log::FlushError();
}

//-----------------------------------------------------------
void PanicErrorMsg( const char* message, ... )
{
    va_list args;
    va_start( args, message );
    VPanicErrorMsg( message, args );
    va_end( args );
}

//-----------------------------------------------------------
void VFatal( const char* message, va_list args )
{
    VFatalErrorMsg( message, args );
    FatalExit();
}

//-----------------------------------------------------------
bool AssertLog( int line, const char* file, const char* func )
{
    Log::Error( "Assertion Failed @ %s:%d %s().", file, line, func );
    Log::FlushError();
    return true;
}

//-----------------------------------------------------------
std::string HexToString( const byte* bytes, size_t length )
{
    ASSERT( length );

    const size_t slen = length * 2 + 1;
    char* buffer      = (char*)malloc( slen );
    memset( buffer, 0, slen );

    size_t numEncoded;
    BytesToHexStr( bytes, length, buffer, slen, numEncoded );

    std::string str( buffer );
    free( buffer );

    return str;
}

//-----------------------------------------------------------
std::vector<uint8_t> HexStringToBytes( const char* hexStr )
{
    const size_t len  = strlen( hexStr );

    byte* buffer = (byte*)malloc( len / 2 );

    HexStrToBytes( hexStr, len, buffer, len / 2 );
    std::vector<uint8_t> ret( buffer, buffer + len / 2 );

    free( buffer );
    return ret;
}

//-----------------------------------------------------------
std::vector<uint8_t> HexStringToBytes( const std::string& hexStr )
{
    return HexStringToBytes( hexStr.c_str() );
}



