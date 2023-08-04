#pragma once
#include <cstring>
#include "util/Util.h"

namespace bls
{
    class G1Element;
}

class CliParser
{
public:
    //-----------------------------------------------------------
    inline CliParser( int argc, const char* argv[] )
            : _i   ( 0 )
            , _argc( argc )
            , _argv( argv )
    {}

    //-----------------------------------------------------------
    [[nodiscard]]
    inline bool HasArgs() const
    {
        return _i < _argc;
    }

    //-----------------------------------------------------------
    [[nodiscard]]
    inline bool IsLastArg() const
    {
        return _i == _argc - 1;
    }

    //-----------------------------------------------------------
    inline int32 RemainingArgCount() const
    {
        return _argc - _i;
    }

    //-----------------------------------------------------------
    inline const char* Arg()
    {
        FatalIf( !HasArgs(), "Expected an argument." );
        return _argv[_i];
    }

    //-----------------------------------------------------------
    inline const char* Peek()
    {
        if( !HasArgs() )
            return nullptr;
        
        return _argv[_i];
    }

    //-----------------------------------------------------------
    inline void NextArg()
    {
        _i++;
    }

    //-----------------------------------------------------------
    inline bool ArgMatch( const char* paramA, const char* paramB = nullptr )
    {
        const char* arg = Arg();
        return strcmp( paramA, arg ) == 0 ||
               ( paramB && strcmp( paramB, arg ) == 0 );
    }

    //-----------------------------------------------------------
    inline bool ArgConsume( const char* paramA, const char* paramB = nullptr )
    {
        if( ArgMatch( paramA, paramB ) )
        {
            NextArg();
            return true;
        }
        
        return false;
    }

    //-----------------------------------------------------------
    inline const char* ArgConsume()
    {
        const char* arg = Arg();
        NextArg();

        return arg;
    }


    bool ReadSwitch( bool& value, const char* paramA, const char* paramB = nullptr );

    // Same as ReadSwitch but set's the value to false if
    // this parameter is matched.
    //-----------------------------------------------------------
    bool ReadUnswitch( bool& value, const char* paramA, const char* paramB = nullptr );

    bool ReadStr( const char*& value, const char* paramA, const char* paramB = nullptr );
    
    uint64 ReadU64();
    
    bool ReadU64( uint64& value, const char* paramA, const char* paramB = nullptr );
    
    bool ReadI64( int64& value, const char* paramA, const char* paramB = nullptr );
    
    bool ReadU32( uint32& value, const char* paramA, const char* paramB = nullptr );

    bool ReadI32( int32& value, const char* paramA, const char* paramB = nullptr );

    bool ReadF64( float64& value, const char* paramA, const char* paramB = nullptr );

    bool ReadF32( float32& value, const char* paramA, const char* paramB = nullptr );

    bool ReadPKey( bls::G1Element* value, const char* paramA, const char* paramB = nullptr );

    bool ReadPKey( bls::G1Element& value, const char* paramA, const char* paramB = nullptr );

    bool ReadPuzzleHash( struct PuzzleHash* value, const char* paramA, const char* paramB = nullptr );

    bool ReadSize( size_t& value, const char* paramA, const char* paramB = nullptr  );

    bool ReadSize( const char* sizeText, size_t& size, const char* arg = "" );

    bool ReadHexStr( const char*& hexStr, size_t maxStrLength, const char* paramA, const char* paramB = nullptr  );
    
    bool ReadHexStrAsBytes( byte* bytes, size_t maxBytes, const char* paramA, const char* paramB = nullptr  );

    size_t ReadSize( const char* arg );

    size_t ReadSize();


private:
    int          _i;
    int          _argc;
    const char** _argv;
};

