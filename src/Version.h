#pragma once

#define BLADEBIT_VERSION_MAJ    2
#define BLADEBIT_VERSION_MIN    0
#define BLADEBIT_VERSION_REV    0
#define BLADEBIT_VERSION_SUFFIX ""
#define BLADEBIT_GIT_COMMIT     ""

#ifdef __GNUC__
    #define BLADEBIT_BUILD_COMPILER "gcc " STR(__GNUC__) "." STR(__GNUC_MINOR__) "." STR(__GNUC_PATCHLEVEL__)
#elif defined( _MSC_VER )
    #define MSVC_MAJ (_MSC_FULL_VER/10000000)
    #define MSVC_MIN ((_MSC_FULL_VER/100000)-(MSVC_MAJ*100))
    #define MSVC_REV ((_MSC_FULL_VER/10)-((_MSC_FULL_VER/1000000)*1000000))
    #define MSVC_BUILD (_MSC_FULL_VER-(_MSC_FULL_VER/10*10))

    #define BLADEBIT_BUILD_COMPILER "msvc " STR(MSVC_MAJ) "." STR(MSVC_MIN) "." STR(MSVC_REV) "." STR(MSVC_BUILD)

    #undef MSVC_MAJ
    #undef MSVC_MIN
    #undef MSVC_REV
#elif defined( __clang__ )
    #define BLADEBIT_BUILD_COMPILER "clang " STR(__clang_major__) "." STR(__clang_minor__) "." STR(__clang_patchlevel__)
#else
    #define BLADEBIT_BUILD_COMPILER "Unknown compiler"
#endif

#define BLADEBIT_BUILD_COMPILER     ""


#define BLADEBIT_VERSION \
    ((uint64)BLADEBIT_VERSION_MAJ) << 32 \
    ((uint64)BLADEBIT_VERSION_MIN) << 16 \
    ((uint64)BLADEBIT_VERSION_REV)

#define BLADEBIT_VERSION_STR \
    STR( BLADEBIT_VERSION_MAJ ) "." STR( BLADEBIT_VERSION_MIN ) "." STR( BLADEBIT_VERSION_REV ) \
    BLADEBIT_VERSION_SUFFIX

