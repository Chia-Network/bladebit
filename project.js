
const CONFIG_NAMES = [
    'release',
    'debug',
    'debug.arm',
    'release.arm',

    // Test/sandbox mode
    'debug.test',
    'release.test',
    // 'debug.arm.test',
    // 'release.arm.test'
];

const UNIX_SOURCES = [
    'src/platform/**/unix/**/*'
];


// Project Transform Function
const bladebit = project( 'bladebit', CONFIG_NAMES, {
    headerPaths: [
        'include',
        'lib/include',
        'lib/include/relic',
        'src'
    ],

    src: [
        'src/**/*',
        'src/b3/*.S',
    ],

    ignore: [
        'src/platform/**/*'
        
        ,'src/b3/blake3_avx*.c'
        ,'src/b3/blake3_sse*.c'
        ,'src/test/**/*'
    ],

    // precompiledHeader: 'src/pch.cpp', 

    libraryPaths : [
        'lib'
    ],

    def :{
        _HAS_STD_BYTE: 0
    },

    undef:[ 
        
    ],

    libs: [
        'bls'
    ]
});

///
/// Configs
///
bladebit.configs.release = () => config({
    def  : { NDEBUG: 1, _NDEBUG: 1 },
    undef: [ 'DEBUG', '_DEBUG' ]

});

bladebit.configs.debug = () => config({
    def  : { DEBUG: 1, _DEBUG: 1 },
    undef: [ 'NDEBUG', '_NDEBUG' ],

});

/// Architectures
bladebit.configs.arm = () => config({
    def  : {
        // PLATFORM_IS_ARM: 1
        // ,BLAKE3_USE_NEON: 1
    }

    ,libraryPaths : [
        'lib/linux/arm'
    ]

    ,ignore: [
        'src/b3/*.S'
    ]
});

///
/// Platforms
///
bladebit.platforms.win32 = () => config({
    def: {
        _WIN32                   : 1
        ,WIN32                   : 1
        ,WIN32_LEAN_AND_MEAN     : 1
        ,UNICODE                 : 1
        ,NOMINMAX                : 1
        ,_CRT_SECURE_NO_WARNINGS : 1
    },

    precompiledHeader: 'src/pch.cpp',

    src: [
        'src/platform/**/win32/**/*',
        'src/platform/**/*_win32*',
        'src/platform/**/*_Win32*'
    ],

    ignore: [
        
    ]
});

bladebit.platforms.linux = () => config({
    src: UNIX_SOURCES.concat([
        
        'src/platform/**/linux/**/*'

        // x86 blake
        // 'src/b3/blake3_sse2_x86-64_unix.S',
        ,'src/b3/blake3_sse41_x86-64_unix.S'
        ,'src/b3/blake3_avx2_x86-64_unix.S'
        ,'src/b3/blake3_avx512_x86-64_unix.S'
    ])

    ,libraryPaths : [
        'lib/linux',
        'lib/linux/x86'
    ]

    ,def: {
        _GNU_SOURCE: 1
        // ,BLAKE3_NO_AVX512: 1
    }
});

bladebit.platforms.macos = () => config({
    src: UNIX_SOURCES.concat([
        'src/platform/**/macos/**/*'
    ])

    ,def: {
        // _GNU_SOURCE: 1
        // ,BLAKE3_NO_AVX512: 1
    }
});


bladebit.configs.test = () => config({
    def: {
        TEST_MODE: 1
    }

    ,src: [

        'src/test/test_main.cpp'
        ,'src/test/test_numa_sort.cpp'
        // ,'src/test/TestNuma.cpp'
    ]

    ,ignore: [
        'src/main.cpp'
    ]
});


const projects = { bladebit: bladebit };
return projects;