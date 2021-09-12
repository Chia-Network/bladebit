#pragma once

class Log
{
    static bool _verbose;

public:
    static void Write( const char* msg, ... );
    static void WriteLine( const char* msg, ... );
    static void Line( const char* msg, ... );

    static void Write( const char* msg, va_list args );
    static void WriteLine( const char* msg, va_list args );

    static void Error( const char* msg, ... );
    static void WriteError( const char* msg, ... );
    static void Error( const char* msg, va_list args );
    static void WriteError( const char* msg, va_list args );

    inline static void SetVerbose( bool enabled ) { _verbose = enabled; }
    
    static void Verbose( const char* msg, ...  );
    static void VerboseWrite( const char* msg, ...  );

    static void Flush();
    static void FlushError();
private:

    static FILE* GetOutStream();
    static FILE* GetErrStream();

private:
    static FILE* _outStream;
    static FILE* _errStream;
};