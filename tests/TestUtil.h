#pragma once

#include "catch2/catch_test_macros.hpp"
#include "util/Util.h"
#include "util/Log.h"
#include "threading/MTJob.h"
#include "threading/ThreadPool.h"
#include "util/jobs/MemJobs.h"
#include "plotdisk/jobs/IOJob.h"
#include "io/FileStream.h"

// Helper to break on the assertion failure when debugging a test.
// You can pass an argument to the catch to break on asserts, however,
// REQUIRE itself is extremely heavy to call directly.
#define ENSURE( x ) \
    if( !(x) ) { ASSERT( x ); REQUIRE( x ); }



//-----------------------------------------------------------
template<typename T>
T* LoadReferenceTable( const char* path, uint64& outEntryCount )
{
    FileStream file;
    FatalIf( !file.Open( path, FileMode::Open, FileAccess::Read, FileFlags::LargeFile | FileFlags::NoBuffering ),
                 "Failed to open reference meta table file %s.", path );

    byte* pBlock = (byte*)alloca( file.BlockSize() * 2 );
    pBlock = (byte*)RoundUpToNextBoundary( (uintptr_t)pBlock, (uintptr_t)file.BlockSize() );

    FatalIf( file.Read( pBlock, file.BlockSize() ) != (ssize_t)file.BlockSize(),
        "Failed to read table length." );
    
    const uint64 entryCount = *(uint64*)pBlock;

    const size_t allocSize = RoundUpToNextBoundaryT( sizeof( T ) * (size_t)entryCount, file.BlockSize() );
    T* entries = bbvirtallocbounded<T>( allocSize );

    int err = 0;
    FatalIf( !IOJob::ReadFromFile( file, entries, entryCount * sizeof( T ), pBlock, file.BlockSize(), err ),
        "Failed to read table file '%s' with error: %d", path, err );

    outEntryCount = entryCount;
    return entries;
}
