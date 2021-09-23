#pragma once

#include <string>
#include <string.h>

#ifdef _MSC_VER
    #define Swap16( x ) _byteswap_ushort( x )
    #define Swap32( x ) _byteswap_ulong( x )
    #define Swap64( x ) _byteswap_uint64( x )
#elif defined( __GNUC__ )
    #define Swap16( x ) __builtin_bswap16( x )
    #define Swap32( x ) __builtin_bswap32( x )
    #define Swap64( x ) __builtin_bswap64( x )
#else 
    #error Byte swapping intrinsics not configured for this compiler.
#endif


/// Byte size conversions
#define KB *(1<<10)
#define MB *(1<<20)
#define GB *(1<<30)

#define BtoKB /(1<<10)
#define BtoMB /(1<<20)
#define BtoGB /(1<<30)


///
/// Assorted utility functions
/// 

// Post a message and exit with error
//-----------------------------------------------------------
void Fatal( const char* message, ... );

void FatalIf( bool condition, const char* message, ... );

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

// Like ceil div, but tells you how man times to multiply
// a in order to fit into b-sized chunks
//-----------------------------------------------------------
template <typename T>
constexpr inline T CDiv( T a, int b )
{
    return ( a + (T)b - 1 ) / (T)b;
}

/// Divides <deividend> by <divisor> and rounds
/// it up to the next factor of <divisor>
//-----------------------------------------------------------
template<typename T>
inline T CeildDiv( T dividend, T divisor )
{
    return dividend + ( divisor - ( dividend % divisor ) ) % divisor;
}

// Round up a number to the next upper boundary.
// For example, if we want to round up some bytes to the
// next 8-byte boundary.
// This is the same as CeilDiv, but with a more intuitive name.
//-----------------------------------------------------------
template<typename T>
inline T RoundUpToNextBoundary( T value, int boundary )
{
    return CeildDiv( value, (T)boundary );
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

