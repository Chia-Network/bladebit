#include "CliParser.h"
#include "util/KeyTools.h"

//-----------------------------------------------------------
bool CliParser::ReadSwitch( bool& value, const char* paramA, const char* paramB )
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
bool CliParser::ReadUnswitch( bool& value, const char* paramA, const char* paramB )
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
bool CliParser::ReadStr( const char*& value, const char* paramA, const char* paramB )
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
uint64 CliParser::ReadU64()
{
    const char* strValue = Arg();
    NextArg();

    uint64 value;
    int r = sscanf( strValue, "%llu",(llu*)&value );
    FatalIf( r != 1, "Expected an uint64 value at parameter %d.", _i );
    
    return value;
}

//-----------------------------------------------------------
bool CliParser::ReadU64( uint64& value, const char* paramA, const char* paramB )
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
bool CliParser::ReadI64( int64& value, const char* paramA, const char* paramB )
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
bool CliParser::ReadU32( uint32& value, const char* paramA, const char* paramB )
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
bool CliParser::ReadI32( int32& value, const char* paramA, const char* paramB )
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
bool CliParser::ReadF64( float64& value, const char* paramA, const char* paramB )
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
bool CliParser::ReadF32( float32& value, const char* paramA, const char* paramB )
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
bool CliParser::ReadPKey( class bls::G1Element* value, const char* paramA, const char* paramB )
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
bool CliParser::ReadPKey( bls::G1Element& value, const char* paramA, const char* paramB )
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
bool CliParser::ReadPuzzleHash( PuzzleHash* value, const char* paramA, const char* paramB )
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
bool CliParser::ReadSize( size_t& value, const char* paramA, const char* paramB )
{
    const char* sizeText = nullptr;
    if( !ReadStr( sizeText, paramA, paramB ) )
        return false;

    const char* arg = _argv[_i-2];
    return ReadSize( sizeText, value, arg );
}

//-----------------------------------------------------------
bool CliParser::ReadSize( const char* sizeText, size_t& size, const char* arg )
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
        if( StriCmp( "GiB", suffix ) == 0 || StriCmp( "G", suffix ) == 0 )
            multiplier = 1ull GiB;
        else if( StriCmp( "MiB", suffix ) == 0 || StriCmp( "M", suffix ) == 0 )
            multiplier = 1ull MiB;
        else if( StriCmp( "KiB", suffix ) == 0 || StriCmp( "K", suffix ) == 0 )
            multiplier = 1ull KiB;
        else if( StriCmp( "TiB", suffix ) == 0 || StriCmp( "T", suffix ) == 0 )
            multiplier = 1ull TiB;
        else if( StriCmp( "PiB", suffix ) == 0 || StriCmp( "P", suffix ) == 0 )
            multiplier = 1ull PiB;

        // SI Units
        else if( StriCmp( "PB", suffix ) == 0 )
            multiplier = 1ull PBSi;
        else if( StriCmp( "TB", suffix ) == 0 )
            multiplier = 1ull TBSi;
        else if( StriCmp( "GB", suffix ) == 0  )
            multiplier = 1ull GBSi;
        else if( StriCmp( "MB", suffix ) == 0 )
            multiplier = 1ull MBSi;
        else if( StriCmp( "KB", suffix ) == 0 )
            multiplier = 1ull KBSi;

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

///-----------------------------------------------------------
size_t CliParser::ReadSize( const char* arg )
{
    size_t size;
    FatalIf( !ReadSize( Arg(), size, arg ),
        "Expected a size argument for paramter '%s'", arg );

    NextArg();
    return size;
}

//-----------------------------------------------------------
size_t CliParser::ReadSize()
{
    size_t size;
    FatalIf( !ReadSize( Arg(), size, "" ),
        "Expected a size argument at index %d", _i );

    NextArg();
    return size;
}

//-----------------------------------------------------------
bool CliParser::ReadHexStr( const char*& hexStr, const size_t maxStrLength, const char* paramA, const char* paramB )
{
    if( !ReadStr( hexStr, paramA, paramB ) )
        return false;
    
    size_t len = strlen( hexStr );
    if( len >= 2 && hexStr[0] == '0' && hexStr[0] == 'x' )
    {
        hexStr += 2;
        len -= 2;
    }

    FatalIf( len == 0, "Expected a hexadecimal string for parameter '%s'.", _argv[_i-1] );
    FatalIf( len < maxStrLength, "Hexadecimal string '%s' for parameter '%s' is too long.", hexStr, _argv[_i-1] );

    for( size_t i = 0; i < len; i++ )
    {
        FatalIf( !IsHexChar( hexStr[i] ), "Expected a hexadecimal string for parameter '%s'.", _argv[_i-1] );
    }

    return true;
}
    
//-----------------------------------------------------------
bool CliParser::ReadHexStrAsBytes( byte* bytes, size_t maxBytes, const char* paramA, const char* paramB )
{
    const char* hexStr = nullptr;
    if( !ReadHexStr( hexStr, maxBytes*2, paramA, paramB ) )
        return false;

    FatalIf( !HexStrToBytesSafe( hexStr, strlen( hexStr ), bytes, maxBytes ),
        "Failed to parse hexadecimal string for parameter '%s'.",  _argv[_i-1] );

    return true;
}