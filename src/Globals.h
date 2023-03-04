#pragma once

#define DEFER( m, ...)  m( __VA_ARGS__ )
#define XSTR( x ) #x
#define STR( x ) XSTR( x )



#ifdef _MSC_VER
    #define FORCE_INLINE __forceinline
    #define WARNING_PUSH Unimplemented

    #ifdef _M_ARM
        #define PLATFORM_IS_ARM 1
    #endif
#else

    #if  __aarch64__
        #define PLATFORM_IS_ARM 1
    #endif

    #if defined( __clang__ )
        #define FORCE_INLINE  __attribute__((always_inline))
        // #define WARNING_PUSH _Pragma( STR() )
    #elif defined( __GNUC__ )
        #define FORCE_INLINE  __attribute__((always_inline))
    #endif
#endif

#if __linux__ || __APPLE__
    #define PLATFORM_IS_UNIX 1
#endif

#if __linux__
    #define PLATFORM_IS_LINUX 1
#elif __APPLE__
    #define PLATFORM_IS_APPLE 1

    #include <TargetConditionals.h>
    #if TARGET_OS_MAC
        #define PLATFORM_IS_MACOS 1
    #else
        error Unsupported Apple platform.
    #endif

#elif defined( _WIN32 )
    #define PLATFORM_IS_WINDOWS 1
#endif


#define TimerBegin() std::chrono::steady_clock::now()
#define TimerEnd( startTime ) \
    (std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now() - (startTime) ).count() / 1000.0)

#define TimerEndTicks( startTime ) \
    (std::chrono::steady_clock::now() - (startTime))

#define NanoSecondsToSeconds( nano ) ( (nano) * 1e-9 )

#define TicksToSeconds( duration ) \
    NanoSecondsToSeconds( std::chrono::duration_cast<std::chrono::nanoseconds>( (duration) ).count() )

#define TicksToNanoSeconds( duration ) \
    ( std::chrono::duration_cast<std::chrono::nanoseconds>( (duration) ).count() )
    


#define ImplementFlagOps( FlagType ) \
inline FlagType operator | ( FlagType lhs, FlagType rhs )                                               \
{                                                                                                       \
    using FlagT = typename std::underlying_type<FlagType>::type;                                        \
    return static_cast<FlagType>( static_cast<FlagT>( lhs ) | static_cast<FlagT>( rhs ) );              \
}                                                                                                       \
                                                                                                        \
inline FlagType& operator |= ( FlagType& lhs, FlagType rhs )                                            \
{                                                                                                       \
    using FlagT = typename std::underlying_type<FlagType>::type;                                        \
    lhs = static_cast<FlagType>( static_cast<FlagT>( lhs ) | static_cast<FlagT>( rhs ) );               \
    return lhs;                                                                                         \
}                                                                                                       \
                                                                                                        \
inline FlagType operator & ( FlagType lhs, FlagType rhs )                                               \
{                                                                                                       \
    using FlagT = typename std::underlying_type<FlagType>::type;                                        \
    return static_cast<FlagType>( static_cast<FlagT>( lhs ) & static_cast<FlagT>( rhs ) );              \
}                                                                                                       \
                                                                                                        \
inline FlagType& operator &= ( FlagType& lhs, FlagType rhs )                                            \
{                                                                                                       \
    using FlagT = typename std::underlying_type<FlagType>::type;                                        \
    lhs = static_cast<FlagType>( static_cast<FlagT>( lhs ) & static_cast<FlagT>( rhs ) );               \
    return lhs;                                                                                         \
}                                                                                                       \
                                                                                                        \
inline FlagType operator ~ ( FlagType lhs )                                                             \
{                                                                                                       \
    using FlagT = typename std::underlying_type<FlagType>::type;                                        \
    lhs = static_cast<FlagType>( ~static_cast<FlagT>( lhs ) );                                          \
    return lhs;                                                                                         \
}

#define ImplementArithmeticOps( EnumType ) \
constexpr inline EnumType operator+( EnumType lhs, EnumType rhs )   \
{                                                                   \
    using BaseT = typename std::underlying_type<EnumType>::type;    \
    return static_cast<EnumType>( static_cast<BaseT>( lhs ) +       \
                                  static_cast<BaseT>( rhs ) );      \
}                                                                   \
constexpr inline EnumType operator+( EnumType lhs, int rhs )        \
{                                                                   \
    using BaseT = typename std::underlying_type<EnumType>::type;    \
    return static_cast<EnumType>( static_cast<BaseT>( lhs ) + rhs); \
}                                                                   \
                                                                    \
constexpr inline EnumType operator-( EnumType lhs, EnumType rhs )   \
{                                                                   \
    using BaseT = typename std::underlying_type<EnumType>::type;    \
    return static_cast<EnumType>( static_cast<BaseT>( lhs ) -       \
                                  static_cast<BaseT>( rhs ) );      \
}                                                                   \
constexpr inline EnumType operator-( EnumType lhs, int rhs )        \
{                                                                   \
    using BaseT = typename std::underlying_type<EnumType>::type;    \
    return static_cast<EnumType>( static_cast<BaseT>( lhs ) - rhs );\
}                                                                   \
/* Prefix ++*/                                                      \
constexpr inline EnumType& operator++( EnumType& self )             \
{                                                                   \
    self = self + static_cast<EnumType>( 1 );                       \
    return self;                                                    \
}                                                                   \
/* Postfix ++*/                                                     \
constexpr inline EnumType operator++( EnumType& self, int )         \
{                                                                   \
    return ++self;                                                  \
}                                                                   \
/* Prefix --*/                                                      \
constexpr inline EnumType& operator--( EnumType& self )             \
{                                                                   \
    self = self - static_cast<EnumType>( 1 );                       \
    return self;                                                    \
}                                                                   \
/* Postfix --*/                                                     \
constexpr inline EnumType operator--( EnumType& self, int )         \
{                                                                   \
    return --self;                                                  \
}                                                                   \
                                                                    \


template< typename T>
inline bool IsFlagSet( T flags, T flag )
{
    using UT = typename std::underlying_type<T>::type;
    return ( static_cast<UT>( flags ) & static_cast<UT>( flag ) ) != static_cast<UT>( 0 );
}

template< typename T>
inline bool AreAllFlagSet( T flagsToCheck, T flagsRequired )
{
    using UT = typename std::underlying_type<T>::type;
    return ( static_cast<UT>( flagsToCheck ) & static_cast<UT>( flagsRequired ) ) == static_cast<UT>( flagsRequired );
}

template< typename T>
inline T SetFlag( T& flags, T flag )
{
    using UT = typename std::underlying_type<T>::type;
    flags = static_cast<T>( static_cast<UT>( flags ) | static_cast<UT>( flag ) );

    return flags;
}

template< typename T>
inline T UnSetFlag( T& flags, T flag )
{
    using UT = typename std::underlying_type<T>::type;
    flags = static_cast<T>( static_cast<UT>( flags ) & ~( static_cast<UT>( flag ) ) );

    return flags;
}

template< typename T>
inline T ToggleFlag( T& flags, T flag, bool value )
{
    if( value )
        SetFlag<T>( flags, flag );
    else
        UnSetFlag<T>( flags, flag );

    return flags;
}


template<size_t _size>
struct NBytes
{
    uint8_t data[_size];

    inline uint8_t& operator[]( size_t index ) const { return this->values[index]; }
    inline uint8_t& operator[]( uint32_t index ) const { return this->values[index]; }
    inline uint8_t& operator[]( int32_t index ) const { return this->values[index]; }
    inline uint8_t& operator[]( int64_t index ) const { return this->values[index]; }
};


