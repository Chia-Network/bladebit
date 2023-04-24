#pragma once

#include <string>
#include <string.h>
#include <vector>
#include "Platform.h"
#include "SysHost.h"
#include "Log.h"

#ifdef _MSC_VER
    #define Swap16( x ) _byteswap_ushort( x )
    #define Swap32( x ) _byteswap_ulong( x )
    #define Swap64( x ) _byteswap_uint64( x )
#elif defined( __GNUC__ ) || defined( __clang__ )
    #define Swap16( x ) __builtin_bswap16( x )
    #define Swap32( x ) __builtin_bswap32( x )
    #define Swap64( x ) __builtin_bswap64( x )
#else 
    #error Byte swapping intrinsics not configured for this compiler.
#endif

#if defined(__GNUC__)
    #define AlignAs(bytes) __attribute__( (aligned((bytes))) )
#elif defined(_MSC_VER)
    #define AlignAs(bytes) __declspec( align((bytes)) )
#endif


/// Byte size conversions
#define KB *(1llu<<10)
#define MB *(1llu<<20)
#define GB *(1llu<<30)
#define TB *(1llu<<40)
#define PB *(1llu<<50)

#define KiB *(1llu<<10)
#define MiB *(1llu<<20)
#define GiB *(1llu<<30)
#define TiB *(1llu<<40)
#define PiB *(1llu<<50)

#define BtoKB /(1llu<<10)
#define BtoMB /(1llu<<20)
#define BtoGB /(1llu<<30)


/// SI Units
#define KBSi *(1000llu)
#define MBSi *(1000llu KBSi )
#define GBSi *(1000llu MBSi )
#define TBSi *(1000llu GBSi )
#define PBSi *(1000llu TBSi )


#define BtoKBSi( v ) ((v) / 1000llu)
#define BtoMBSi( v ) (BtoKBSi(v) / 1000llu)
#define BtoGBSi( v ) (BtoMBSi(v) / 1000llu)
#define BtoTBSi( v ) (BtoGBSi(v) / 1000llu)
#define BtoPBSi( v ) (BtoTBSi(v) / 1000llu)

#define BtoKBSiF( v ) ((v) / 1000.0)
#define BtoMBSiF( v ) (BtoKBSiF(v) / 1000.0)
#define BtoGBSiF( v ) (BtoMBSiF(v) / 1000.0)
#define BtoTBSiF( v ) (BtoGBSiF(v) / 1000.0)
#define BtoPBSiF( v ) (BtoTBSiF(v) / 1000.0)


///
/// Assorted utility functions
/// 
void Exit( int code );
void FatalExit();
void PanicExit();
void FatalErrorMsg( const char* message, ... );
void PanicErrorMsg( const char* message, ... );

// Fatal: Post a message and exit with error
// Panic: Same as panic, but the error is unexpected,
//         so a stack trace is also printed out.
//-----------------------------------------------------------
#ifdef _WIN32
    #define Fatal( message, ... )  { FatalErrorMsg( message, __VA_ARGS__ ); BBDebugBreak(); FatalExit(); }
    #define Panic( message, ... )  { PanicErrorMsg( message, __VA_ARGS__ ); BBDebugBreak(); PanicExit(); }
#else
    #define Fatal( message, ... )  { FatalErrorMsg( message, ## __VA_ARGS__ ); BBDebugBreak(); FatalExit(); }
    #define Panic( message, ... )  { PanicErrorMsg( message, ## __VA_ARGS__ ); BBDebugBreak(); PanicExit(); }
#endif

#define FatalIf( cond, message, ... ) if( (cond) ) { Fatal( message, ## __VA_ARGS__ ); }
#define PanicIf( cond, message, ... ) if( (cond) ) { Panic( message, ## __VA_ARGS__ ); }


//-----------------------------------------------------------
template <typename T>
constexpr inline T bblog2( T x )
{
    T r = 0;
    while( x >>= 1 )
        r++;
    return r;
}

//-----------------------------------------------------------
constexpr inline int64 bbconst_ceil( const double x )
{
    return (static_cast<double>( static_cast<int64>(x) ) == x) ?
        static_cast<int64>( x ) :
        static_cast<int64>( x ) + (x > 0.0 ? 1 : 0);
}


// Divide a by b and apply ceiling if needed.
//-----------------------------------------------------------
template <typename T>
constexpr inline T CDiv( T a, int b )
{
    return ( a + (T)b - 1 ) / (T)b;
}

//-----------------------------------------------------------
template <typename T>
constexpr inline T CDivT( T a, T b )
{
    return ( a + b - (T)1 ) / b;
}

// Round up a number to the next upper boundary.
// For example, if we want to round up some bytes to the next 8-byte boundary.
//-----------------------------------------------------------
template<typename T>
constexpr inline T RoundUpToNextBoundary( T value, int boundary )
{
    return value + ( boundary - ( value % boundary ) ) % boundary;
}

template<typename T>
constexpr inline T RoundUpToNextBoundaryT( T value, T boundary )
{
    return value + ( boundary - ( value % boundary ) ) % boundary;
}

//-----------------------------------------------------------
inline bool MemCmp( const void* a, const void* b, size_t size )
{
    return memcmp( a, b, size ) == 0;
}

//-----------------------------------------------------------
template<typename T>
inline T bbclamp( const T value, const T min, const T max )
{
    return value < min ? min : value > max ? max : value;
}

//-----------------------------------------------------------
template<typename T>
inline void ZeroMem( T* ptr )
{
    memset( ptr, 0, sizeof( T ) );
}

//-----------------------------------------------------------
template<typename T>
inline void ZeroMem( T* ptr, size_t count )
{
    ASSERT( count > 0 );
    memset( ptr, 0, sizeof( T ) * count );
}

//-----------------------------------------------------------
template<typename T>
inline T* bbmalloc( size_t size )
{
    void* ptr = malloc( size );
    FatalIf( !ptr, "bbmalloc(): Out of memory." );
    
    return reinterpret_cast<T*>( ptr );
}

//-----------------------------------------------------------
template<typename T>
inline T* bbrealloc( T* ptr, size_t newSize )
{
    ptr = reinterpret_cast<T*>( realloc( ptr, newSize ) );
    FatalIf( !ptr, "bbrealloc(): Out of memory." );

    return reinterpret_cast<T*>( ptr );
}

// #NOTE: Unlike calloc, this does not initialize memory to 0
//-----------------------------------------------------------
template<typename T>
inline T* bbcalloc( size_t count )
{
    return bbmalloc<T>( count * sizeof( T ) );
}

//-----------------------------------------------------------
template<typename T>
inline Span<T> bbcalloc_span( size_t count )
{
    T* ptr = bbmalloc<T>( count * sizeof( T ) );
    return Span<T>( ptr, count );
}

//-----------------------------------------------------------
template<typename T>
inline T* bbcrealloc( T* ptr, size_t newCount )
{
    return bbrealloc( ptr, newCount * sizeof( T ) );
}

//-----------------------------------------------------------
template<typename T>
inline void bbmemcpy_t( T* dst, const T* src, size_t count )
{
    memcpy( dst, src, sizeof( T ) * count );
}

//-----------------------------------------------------------
inline void* bballoca( size_t size )
{
#if PLATFORM_IS_WINDOWS
    return _malloca( size );
#elif PLATFORM_IS_MACOS
    return alloca( size );
#elif PLATFORM_IS_LINUX
    return alloca( size );
#else
    #error Unimplemented Platform
#endif
}

//-----------------------------------------------------------
inline void bbvirtfree( void* ptr )
{
    ASSERT( ptr );
    SysHost::VirtualFree( ptr );
}

//-----------------------------------------------------------
template<typename T>
inline void bbvirtfree_span( Span<T>& span )
{
    if( span.values )
        SysHost::VirtualFree( span.values );

    span = {};
}

//-----------------------------------------------------------
template<typename T = void>
inline T* bbvirtalloc( size_t size )
{
    ASSERT( size );
    void* ptr = SysHost::VirtualAlloc( size, false );
    FatalIf( !ptr, "VirtualAlloc failed." );
    return reinterpret_cast<T*>( ptr );
}

//-----------------------------------------------------------
template<typename T = void>
inline T* bbvirtallocnuma( size_t size )
{
    T* ptr = bbvirtalloc<T>( size );

    if( SysHost::GetNUMAInfo() )
    {
        if( !SysHost::NumaSetMemoryInterleavedMode( ptr, size ) )
            Log::Error( "Warning: Failed to bind NUMA memory." );
    }

    return ptr;
}

//-----------------------------------------------------------
template<typename T>
inline T* bbcvirtalloc( size_t count )
{
    return bbvirtalloc<T>( sizeof( T ) * count );
}

//-----------------------------------------------------------
inline void* bb_try_virt_alloc( size_t size )
{
    return SysHost::VirtualAlloc( size, false );
}

// Allocate virtual memory with protected boundary pages
// #NOTE: Only free with bbvirtfreebounded
//-----------------------------------------------------------
inline void* bb_try_virt_alloc_bounded( size_t size )
{
    const size_t pageSize = SysHost::GetPageSize();
    size = RoundUpToNextBoundaryT<size_t>( size, pageSize ) + pageSize * 2;

    auto* ptr = (byte*)bb_try_virt_alloc( size );

    SysHost::VirtualProtect( ptr, pageSize, VProtect::NoAccess );
    SysHost::VirtualProtect( ptr + size - pageSize, pageSize, VProtect::NoAccess );

    return ptr + pageSize;
}

//-----------------------------------------------------------
template<typename T>
inline T* bb_try_virt_calloc_bounded( const size_t count )
{
    return reinterpret_cast<T*>( bb_try_virt_alloc_bounded( count * sizeof( T ) ) );
}

//-----------------------------------------------------------
template<typename T>
inline Span<T> bb_try_virt_calloc_bounded_span( const size_t count )
{
    auto span = Span<T>( bb_try_virt_calloc_bounded<T>( count ), count );
    if( span.Ptr() == nullptr )
        span.length = 0;

    return span;
}

// Allocate virtual memory with protected boundary pages
// #NOTE: Only free with bbvirtfreebounded
//-----------------------------------------------------------
template<typename T = void>
inline T* bbvirtallocbounded( const size_t size )
{
    void* ptr = bb_try_virt_alloc_bounded( size );
    FatalIf( !ptr, "VirtualAlloc failed." );

    return reinterpret_cast<T*>( ptr );
}

//-----------------------------------------------------------
template<typename T = void>
inline T* bbvirtallocboundednuma( size_t size )
{
    T* ptr = bbvirtallocbounded<T>( size );
    if( SysHost::GetNUMAInfo() )
    {
        if( !SysHost::NumaSetMemoryInterleavedMode( ptr, size ) )
            Log::Error( "Warning: Failed to bind NUMA memory." );
    }
   
   return ptr;
}

//-----------------------------------------------------------
template<typename T>
inline T* bbcvirtallocbounded( size_t count )
{
    return bbvirtallocbounded<T>( sizeof( T ) * count );
}

//-----------------------------------------------------------
template<typename T = void>
inline T* bbcvirtallocboundednuma( size_t count )
{
    T* ptr = bbcvirtallocbounded<T>( count );
    if( SysHost::GetNUMAInfo() )
    {
        if( !SysHost::NumaSetMemoryInterleavedMode( ptr, count * sizeof( T ) ) )
            Log::Error( "Warning: Failed to bind NUMA memory." );
    }
   
   return ptr;
}

//-----------------------------------------------------------
template<typename T = void>
inline bool bb_interleave_numa_memory( const size_t count, T* ptr )
{
    return SysHost::GetNUMAInfo() && SysHost::NumaSetMemoryInterleavedMode( ptr, count * sizeof( T ) );
}

//-----------------------------------------------------------
template<typename T = void>
inline Span<T> bbcvirtallocboundednuma_span( size_t count )
{
    return Span<T>( bbcvirtallocboundednuma<T>( count ), count );
}

//-----------------------------------------------------------
inline void bbvirtfreebounded( void* ptr )
{
    ASSERT( ptr );
    SysHost::VirtualFree( ((byte*)ptr) - SysHost::GetPageSize() );
}

//-----------------------------------------------------------
template<typename T>
inline void bbvirtfreebounded_span( Span<T>& span )
{
    if( span.values )
        bbvirtfreebounded( span.values );
    
    span = {};
}


const char HEX_TO_BIN[256] = {
    0,   // 0	00	NUL
    0,   // 1	01	SOH
    0,   // 2	02	STX
    0,   // 3	03	ETX
    0,   // 4	04	EOT
    0,   // 5	05	ENQ
    0,   // 6	06	ACK
    0,   // 7	07	BEL
    0,   // 8	08	BS
    0,   // 9	09	HT
    0,   // 10	0A	LF
    0,   // 11	0B	VT
    0,   // 12	0C	FF
    0,   // 13	0D	CR
    0,   // 14	0E	SO
    0,   // 15	0F	SI
    0,   // 16	10	DLE
    0,   // 17	11	DC1
    0,   // 18	12	DC2
    0,   // 19	13	DC3
    0,   // 20	14	DC4
    0,   // 21	15	NAK
    0,   // 22	16	SYN
    0,   // 23	17	ETB
    0,   // 24	18	CAN
    0,   // 25	19	EM
    0,   // 26	1A	SUB
    0,   // 27	1B	ESC
    0,   // 28	1C	FS
    0,   // 29	1D	GS
    0,   // 30	1E	RS
    0,   // 31	1F	US
    0,   // 32	20	space
    0,   // 33	21	!
    0,   // 34	22	"
    0,   // 35	23	#
    0,   // 36	24	$
    0,   // 37	25	%
    0,   // 38	26	&
    0,   // 39	27	'
    0,   // 40	28	(
    0,   // 41	29	)
    0,   // 42	2A	*
    0,   // 43	2B	+
    0,   // 44	2C	,
    0,   // 45	2D	-
    0,   // 46	2E	.
    0,   // 47	2F	/
    0,   // 48	30	0
    1,   // 49	31	1
    2,   // 50	32	2
    3,   // 51	33	3
    4,   // 52	34	4
    5,   // 53	35	5
    6,   // 54	36	6
    7,   // 55	37	7
    8,   // 56	38	8
    9,   // 57	39	9
    0,   // 58	3A	:
    0,   // 59	3B	;
    0,   // 60	3C	<
    0,   // 61	3D	=
    0,   // 62	3E	>
    0,   // 63	3F	?
    0,   // 64	40	@
    10,   // 65	41	A
    11,   // 66	42	B
    12,   // 67	43	C
    13,   // 68	44	D
    14,   // 69	45	E
    15,   // 70	46	F
    0,   // 71	47	G
    0,   // 72	48	H
    0,   // 73	49	I
    0,   // 74	4A	J
    0,   // 75	4B	K
    0,   // 76	4C	L
    0,   // 77	4D	M
    0,   // 78	4E	N
    0,   // 79	4F	O
    0,   // 80	50	P
    0,   // 81	51	Q
    0,   // 82	52	R
    0,   // 83	53	S
    0,   // 84	54	T
    0,   // 85	55	U
    0,   // 86	56	V
    0,   // 87	57	W
    0,   // 88	58	X
    0,   // 89	59	Y
    0,   // 90	5A	Z
    0,   // 91	5B	[
    0,   // 92	5C	\ //
    0,   // 93	5D	]
    0,   // 94	5E	^
    0,   // 95	5F	_
    0,   // 96	60	`
    10,  // 97	61	a
    11,  // 98	62	b
    12,  // 99	63	c
    13,  // 100	64	d
    14,  // 101	65	e
    15,  // 102	66	f
    0,   // 103	67	g
    0,   // 104	68	h
    0,   // 105	69	i
    0,   // 106	6A	j
    0,   // 107	6B	k
    0,   // 108	6C	l
    0,   // 109	6D	m
    0,   // 110	6E	n
    0,   // 111	6F	o
    0,   // 112	70	p
    0,   // 113	71	q
    0,   // 114	72	r
    0,   // 115	73	s
    0,   // 116	74	t
    0,   // 117	75	u
    0,   // 118	76	v
    0,   // 119	77	w
    0,   // 120	78	x
    0,   // 121	79	y
    0,   // 122	7A	z
    0,   // 123	7B	{
    0,   // 124	7C	|
    0,   // 125	7D	}
    0,   // 126	7E	~
    0,   // 127	7F	DEL
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};


//-----------------------------------------------------------
inline bool IsHexChar( const char c )
{
    if( c < '0' || c > 'f' ) return false;
    return c <= '9' || c >= 'a' || ( c >= 'A' && c <= 'B' );
}

//-----------------------------------------------------------
inline void HexStrToBytes( const char* str, const size_t strSize,
                           byte* dst, size_t dstSize )
{
    ASSERT( str && strSize );
    ASSERT( dst && dstSize );

    const size_t maxSize = (strSize / 2) * 2;
    const char* end = str + maxSize;

    ASSERT( dstSize >= maxSize / 2 );

    int i = 0;
    while( str < end )
    {
        byte msb = (byte)HEX_TO_BIN[(int)str[0]];
        byte lsb = (byte)HEX_TO_BIN[(int)str[1]];

        byte v = lsb + msb * 16u;
        dst[i++] = v;
        str += 2;
    }
}

//-----------------------------------------------------------
inline bool HexStrToBytesSafe( const char* str, const size_t strSize,
                               byte* dst, size_t dstSize )
{
    if( !str || !strSize || !dst || !dstSize )
        return false;

    const size_t maxSize = (strSize / 2) * 2;
    const char* end      = str + maxSize;

    if( dstSize < maxSize / 2 )
        return false;

    int i = 0;
    while( str < end )
    {
        const int char0 = (int)str[0];
        const int char1 = (int)str[1];

        if( (char0 - (int)'0' > 9 && char0 - (int)'A' > 5 && char0 - (int)'a' > 5) ||
            (char1 - (int)'0' > 9 && char1 - (int)'A' > 5 && char1 - (int)'a' > 5) )
            return false;

        byte msb = (byte)HEX_TO_BIN[char0];
        byte lsb = (byte)HEX_TO_BIN[char1];

        byte v = lsb + msb * 16u;
        dst[i++] = v;
        str += 2;
    }

    return true;
}

//-----------------------------------------------------------
// Encode bytes into hex format
// Return:
//  0 if OK
//  -1 if Needed more space in the dst buffer to write
//  -2 if the required dstSize would overflow
//-----------------------------------------------------------
inline int BytesToHexStr( const byte* src, size_t srcSize,
                          char* dst, size_t dstSize,
                          size_t& numEncoded, 
                          bool uppercase = false )
{
    const char HEXUC[] = "0123456789ABCDEF";
    const char HEXLC[] = "0123456789abcdef";

    const char* HEX = uppercase ? HEXUC : HEXLC;

    const size_t MAX_SRC_SIZE = std::numeric_limits<size_t>::max() / 2;

    numEncoded = 0;
    int ret = 0;

    if( dstSize == 0 )
    {
        return -1;
    }

    size_t maxEncode    = srcSize;
    size_t dstRequired;
      
    // Check for overflow
    if( maxEncode > MAX_SRC_SIZE )
    {
        maxEncode   = MAX_SRC_SIZE;
        dstRequired = std::numeric_limits<size_t>::max();
        numEncoded  = MAX_SRC_SIZE;
        ret = -2;
    }
    else
    {
        dstRequired = maxEncode * 2;
        numEncoded  = maxEncode;
    }

    // Cap the encode count to the dst buffer size
    if( dstRequired > dstSize )
    {
        ret = -1;
        numEncoded = dstSize/2;
    }

    const byte* s   = src;
    const byte* end = src + numEncoded;
    char* d = dst;

    while( s < end )
    {
        d[0] = (char)HEX[(*s >> 4) & 0x0F];
        d[1] = (char)HEX[*s & 15];

        s++;
        d += 2;
    }

    return ret;
}

//-----------------------------------------------------------
inline std::vector<uint8_t> BytesConcat( std::vector<uint8_t> a, std::vector<uint8_t> b, std::vector<uint8_t> c )
{
    a.insert( a.end(), b.begin(), b.end() );
    a.insert( a.end(), c.begin(), c.end() );
    return a;
}

std::string BytesToHexStdString( const byte* bytes, size_t length );
std::vector<uint8_t> HexStringToBytes( const char* hexStr );
std::vector<uint8_t> HexStringToBytes( const std::string& hexStr );

//-----------------------------------------------------------
inline bool IsDigit( const char c )
{
    return c >= '0' && c <= '9';
}

//-----------------------------------------------------------
inline bool IsNumber( const char* str )
{
    if( str == nullptr )
        return false;

    while( *str )
    {
        if( !IsDigit( *str++ ) )
            return false;
    }
    
    return true;
}

// Offsets a string pointer to the bytes from immediately
// following the "0x", if the string starts with such a prefix.
// #NOTE: Assumes a null-terminated string.
//-----------------------------------------------------------
inline const char* Offset0xPrefix( const char* str )
{
    ASSERT( str );
    if( str && str[0] == '0' && (str[1] == 'x' || str[1] == 'X') )
        return str+2;

    return str;
}

template<typename T>
inline void PrintBits( const T value, const uint32 bitCount )
{
    const uint32 shift = bitCount - 1;
    for( uint32 i = 0; i < bitCount; i++ )
        (value >> (shift-i)) & T(1) ? Log::Write( "1" ) : Log::Write( "0" );
}

//-----------------------------------------------------------
/// Convertes 8 bytes to uint64 and endian-swaps it.
/// This takes any byte alignment, so that bytes does
/// not have to be aligned to 64-bit boundary.
/// This is for compatibility for how chiapos extracts
/// bytes into integers.
//-----------------------------------------------------------
inline uint64 BytesToUInt64( const byte bytes[8] )
{
    uint64 tmp;

    if( (((uintptr_t)&bytes[0]) & 7) == 0 ) // Is address 8-byte aligned?
        tmp = *(uint64*)&bytes[0];
    else
        memcpy( &tmp, bytes, sizeof( uint64 ) );

    return Swap64( tmp );
}

