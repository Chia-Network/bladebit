#pragma once
#include <cstring>
#include "util/Util.h"
#include "util/KeyTools.h"

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
    inline const char* Arg()
    {
        FatalIf( !HasArgs(), "Expected an argument." );
        return _argv[_i];
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

    //-----------------------------------------------------------
    inline bool ReadSwitch( bool& value, const char* paramA, const char* paramB = nullptr )
    {
        if( ArgMatch( paramA, paramB ) )
        {
            NextArg();
            value = true;
            return true;
        }

        return false;
    }

    // Same as ReadSwitch but set's the value to false if
    // this parameter is matched.
    //-----------------------------------------------------------
    inline bool ReadUnswitch( bool& value, const char* paramA, const char* paramB = nullptr )
    {
        if( ArgMatch( paramA, paramB ) )
        {
            NextArg();
            value = false;
            return true;
        }

        return false;
    }

    //-----------------------------------------------------------
    inline bool ReadStr( const char*& value, const char* paramA, const char* paramB = nullptr )
    {
        if( !ArgMatch( paramA, paramB ) )
            return false;

        NextArg();
        FatalIf( !HasArgs(), "Expected a value for argument '%s'.", _argv[_i-1] );

        value = _argv[_i];
        NextArg();

        return true;
    }

    //-----------------------------------------------------------
    inline uint64 ReadU64()
    {
        const char* strValue = Arg();
        NextArg();

        uint64 value;
        int r = sscanf( strValue, "%llu",(llu*)&value );
        FatalIf( r != 1, "Expected an uint64 value at parameter %d.", _i );
        
        return value;
    }

    //-----------------------------------------------------------
    inline bool ReadU64( uint64& value, const char* paramA, const char* paramB = nullptr )
    {
        const char* strValue = nullptr;
        if( !ReadStr( strValue, paramA, paramB ) )
            return false;

        const char* arg = _argv[_i-2];
        int r = sscanf( strValue, "%llu", (llu*)&value );
        FatalIf( r != 1, "Invalid uint64 value for argument '%s'.", arg );

        return true;
    }

    //-----------------------------------------------------------
    inline bool ReadI64( int64& value, const char* paramA, const char* paramB = nullptr )
    {
        const char* strValue = nullptr;
        if( !ReadStr( strValue, paramA, paramB ) )
            return false;

        const char* arg = _argv[_i-2];
        int r = sscanf( strValue, "%lld", &value );
        FatalIf( r != 1, "Invalid int64 value for argument '%s'.", arg );

        return true;
    }

    //-----------------------------------------------------------
    inline bool ReadU32( uint32& value, const char* paramA, const char* paramB = nullptr )
    {
        uint64 u64Value = 0;
        if( !ReadU64( u64Value, paramA, paramB ) )
            return false;

        value = (uint32)u64Value;
        const char* arg = _argv[_i-2];
        FatalIf( value != u64Value, "Invalid uint32 value for argument '%s'.", arg );

        return true;
    }

    //-----------------------------------------------------------
    inline bool ReadI32( int32& value, const char* paramA, const char* paramB = nullptr )
    {
        int64 i64Value = 0;
        if( !ReadI64( i64Value, paramA, paramB ) )
            return false;

        value = (int32)i64Value;
        const char* arg = _argv[_i-2];
        FatalIf( value != i64Value, "Invalid int32 value for argument '%s'.", arg );

        return true;
    }

    //-----------------------------------------------------------
    inline bool ReadF64( float64& value, const char* paramA, const char* paramB = nullptr )
    {
        const char* strValue = nullptr;
        if( !ReadStr( strValue, paramA, paramB ) )
            return false;

        const char* arg = _argv[_i-2];
        int r = sscanf( strValue, "%lf", &value );
        FatalIf( r != 1, "Invalid float64 value for argument '%s'.", arg );

        return true;
    }

    //-----------------------------------------------------------
    inline bool ReadF32( float32& value, const char* paramA, const char* paramB = nullptr )
    {
        const char* strValue = nullptr;
        if( !ReadStr( strValue, paramA, paramB ) )
            return false;

        const char* arg = _argv[_i-2];
        int r = sscanf( strValue, "%f", &value );
        FatalIf( r != 1, "Invalid float32 value for argument '%s'.", arg );

        return true;
    }

     //-----------------------------------------------------------
    inline bool ReadPKey( bls::G1Element* value, const char* paramA, const char* paramB = nullptr )
    {
        const char* strValue = nullptr;
        if( !ReadStr( strValue, paramA, paramB ) )
            return false;

        value = new bls::G1Element();
        if( !KeyTools::HexPKeyToG1Element( strValue, *value ) )
        {
            const char* arg = _argv[_i-2];
            Fatal( "Invalid public key value for argument '%s'.", arg );
        }

        return true;
    }

    //-----------------------------------------------------------
    inline bool ReadPKey( bls::G1Element& value, const char* paramA, const char* paramB = nullptr )
    {
        const char* strValue = nullptr;
        if( !ReadStr( strValue, paramA, paramB ) )
            return false;

        if( !KeyTools::HexPKeyToG1Element( strValue, value ) )
        {
            const char* arg = _argv[_i-2];
            Fatal( "Invalid public key value for argument '%s'.", arg );
        }

        return true;
    }
    
    //-----------------------------------------------------------
    inline bool ReadPuzzleHash( PuzzleHash* value, const char* paramA, const char* paramB = nullptr )
    {
        const char* strValue = nullptr;
        if( !ReadStr( strValue, paramA, paramB ) )
            return false;

        auto* ph = new PuzzleHash();
        if( !PuzzleHash::FromAddress( *ph, strValue ) )
        {
            const char* arg = _argv[_i-2];
            Fatal( "Invalid puzzle hash value '%s' for argument '%s'.", strValue, arg );
        }

        return true;
    }

    //-----------------------------------------------------------
    inline bool ReadSize( size_t& value, const char* paramA, const char* paramB = nullptr  )
    {
        const char* sizeText = nullptr;
        if( !ReadStr( sizeText, paramA, paramB ) )
            return false;

        const char* arg = _argv[_i-2];
        return ReadSize( sizeText, value, arg );
    }

    //-----------------------------------------------------------
    inline bool ReadSize( const char* sizeText, size_t& size, const char* arg = "" )
    {
        ASSERT( sizeText );
        size = 0;

        const size_t len = strlen( sizeText );
        const char*  end = sizeText + len;

        const char* suffix = sizeText;

        #ifdef _WIN32
            #define StriCmp _stricmp
        #else
            #define StriCmp strcasecmp
        #endif

        // Try to find a suffix:
        //  Find the first character that's not a digit
        do
        {
            const char c = *suffix;
            if( c < '0' || c > '9' )
                break;
        }
        while( ++suffix < end );

        // Apply multiplier depending on the suffix
        size_t multiplier = 1;

        const size_t suffixLength = end - suffix;
        if( suffixLength > 0 )
        {
            if( StriCmp( "GB", suffix ) == 0 || StriCmp( "G", suffix ) == 0 )
                multiplier = 1ull GB;
            else if( StriCmp( "MB", suffix ) == 0 || StriCmp( "M", suffix ) == 0 )
                multiplier = 1ull MB;
            else if( StriCmp( "KB", suffix ) == 0 || StriCmp( "K", suffix ) == 0 )
                multiplier = 1ull KB;
            else
            {
                Fatal( "Invalid suffix '%s' for argument '%s'", suffix, arg );
            }
        }

        size_t parsedSize = 0;

        const size_t MAX_DIGITS = 19;
        char digits[MAX_DIGITS + 1];

        const size_t digitsLength = suffix - sizeText;
        FatalIf( digitsLength < 1 || digitsLength > MAX_DIGITS, "Invalid parameters value for argument '%s'.", arg );

        // Read digits
        memcpy( digits, sizeText, digitsLength );
        digits[digitsLength] = 0;

        FatalIf( sscanf( digits, "%llu", (llu*)&parsedSize ) != 1,
                 "Invalid parameters value for argument '%s'.", arg );

        size = parsedSize * multiplier;

        // Check for overflow
        FatalIf( size < size, "Size overflowed for argument '%s'.", arg );

        return true;
        #undef StriCmp
    }

    //-----------------------------------------------------------
    inline size_t ReadSize( const char* arg )
    {
        size_t size;
        FatalIf( !ReadSize( Arg(), size, arg ),
            "Expected a size argument for paramter '%s'", arg );

        NextArg();
        return size;
    }

    //-----------------------------------------------------------
    inline size_t ReadSize()
    {
        size_t size;
        FatalIf( !ReadSize( Arg(), size, "" ),
            "Expected a size argument at index %d", _i );

        NextArg();
        return size;
    }

    //-----------------------------------------------------------
    inline void NextArg()
    {
        _i++;
    }

private:
    int          _i;
    int          _argc;
    const char** _argv;
};

