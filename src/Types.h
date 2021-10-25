#pragma once

typedef uint8_t                 byte;
typedef uint8_t                 uint8;
typedef uint16_t                uint16;
typedef uint32_t                uint32;
typedef unsigned long long int  uint64;

#if !uint
    typedef uint32          uint;
#endif

#if _WIN32
    typedef int64_t         ssize_t;
#endif

typedef uint8  u8;
typedef uint16 u16;
typedef uint32 u32;
typedef uint64 u64;

typedef int8_t              sbyte;
typedef int8_t              int8;
typedef int16_t             int16;
typedef int32_t             int32;
typedef long long int       int64;

typedef int8  i8;
typedef int16 i16;
typedef int32 i32;
typedef int64 i64;

typedef float               float32;
typedef double              float64;
typedef float32             f32;
typedef float64             f64;

typedef uint32              size_t32;
typedef uint64              size_t64;

static_assert( sizeof( byte   ) == 1, "byte must be 1"   );
static_assert( sizeof( uint8  ) == 1, "uint8 must be 1"  );
static_assert( sizeof( uint16 ) == 2, "uint16 must be 2" );
static_assert( sizeof( uint32 ) == 4, "uint32 must be 4" );
static_assert( sizeof( uint64 ) == 8, "uint64 must be 8" );

static_assert( sizeof( sbyte ) == 1, "sbyte must be 1" );
static_assert( sizeof( int8  ) == 1, "int8 must be 1"  );
static_assert( sizeof( int16 ) == 2, "int16 must be 2" );
static_assert( sizeof( int32 ) == 4, "int32 must be 4" );
static_assert( sizeof( int64 ) == 8, "int64 must be 8" );

static_assert( sizeof( float32 ) == 4, "float32 must be 4" );
static_assert( sizeof( float64 ) == 8, "float64 must be 8" );
