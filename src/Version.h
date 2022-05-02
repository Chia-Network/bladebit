#pragma once

#ifndef BLADEBIT_VERSION_MAJ
    #define BLADEBIT_VERSION_MAJ    0
#endif

#ifndef BLADEBIT_VERSION_MIN
    #define BLADEBIT_VERSION_MIN    0
#endif

#ifndef BLADEBIT_VERSION_REV
    #define BLADEBIT_VERSION_REV    0
#endif

#ifndef BLADEBIT_VERSION_SUFFIX
    #define BLADEBIT_VERSION_SUFFIX "-dev"
#endif

#ifndef BLADEBIT_GIT_COMMIT
    #define BLADEBIT_GIT_COMMIT     "unknown"
#endif

// Record compiler version
#if defined( __clang__ )
    struct BBCompilerVer
    {
        static constexpr const char* compiler = "clang";
        static constexpr uint32_t major    = __clang_major__;
        static constexpr uint32_t minor    = __clang_minor__;
        static constexpr uint32_t revision = __clang_patchlevel__;
    };
#elif defined( __GNUC__ )
    struct BBCompilerVer
    {
        static constexpr const char* compiler = "gcc";
        static constexpr uint32_t major    = __GNUC__;
        static constexpr uint32_t minor    = __GNUC_MINOR__;
        static constexpr uint32_t revision = __GNUC_PATCHLEVEL__;
    };
#elif defined( _MSC_VER )
    struct BBCompilerVer
    {
        static constexpr const char* compiler = "msvc";
        static constexpr uint32_t major    = _MSC_VER / 100u;
        static constexpr uint32_t minor    = _MSC_VER - major * 100u;
        static constexpr uint32_t revision = _MSC_FULL_VER - _MSC_VER * 100000u;
    };

#else
    #warning "Unknown compiler"
    struct BBCompilerVer
    {
        static constexpr const char* compiler = "unknown";
        static constexpr uint32_t major    = 0;
        static constexpr uint32_t minor    = 0;
        static constexpr uint32_t revision = 0;
    };
#endif

// #NOTE: Not thread safe
inline const char* BBGetCompilerVersion()
{
    static char c[256] = {};
    sprintf( c, "%s %u.%u.%u", BBCompilerVer::compiler, BBCompilerVer::major,
                               BBCompilerVer::minor, BBCompilerVer::revision );
    return c;
}


#define BLADEBIT_VERSION \
    ((uint64)BLADEBIT_VERSION_MAJ) << 32 \
    ((uint64)BLADEBIT_VERSION_MIN) << 16 \
    ((uint64)BLADEBIT_VERSION_REV)

#define BLADEBIT_VERSION_STR \
    STR( BLADEBIT_VERSION_MAJ ) "." STR( BLADEBIT_VERSION_MIN ) "." STR( BLADEBIT_VERSION_REV ) \
    BLADEBIT_VERSION_SUFFIX

