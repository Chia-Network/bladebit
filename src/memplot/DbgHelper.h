#pragma once

#include "PlotContext.h"
#include "util/Log.h"
#include "io/FileStream.h"

#define DBG_TABLES_PATH ".sandbox/"

#define DBG_P1_TABLE1_FNAME   DBG_TABLES_PATH "p1.t1.tmp"
#define DBG_P1_TABLE2_FNAME   DBG_TABLES_PATH "p1.t2.tmp"
#define DBG_P1_TABLE3_FNAME   DBG_TABLES_PATH "p1.t3.tmp"
#define DBG_P1_TABLE4_FNAME   DBG_TABLES_PATH "p1.t4.tmp"
#define DBG_P1_TABLE5_FNAME   DBG_TABLES_PATH "p1.t5.tmp"
#define DBG_P1_TABLE6_FNAME   DBG_TABLES_PATH "p1.t6.tmp"
#define DBG_P1_TABLE7_FNAME   DBG_TABLES_PATH "p1.t7.tmp"
#define DBG_P1_TABLE7_Y_FNAME DBG_TABLES_PATH "p1.t7.y.tmp"

#define DBG_P2_TABLE2_FNAME   DBG_TABLES_PATH "p2.t2.tmp"
#define DBG_P2_TABLE3_FNAME   DBG_TABLES_PATH "p2.t3.tmp"
#define DBG_P2_TABLE4_FNAME   DBG_TABLES_PATH "p2.t4.tmp"
#define DBG_P2_TABLE5_FNAME   DBG_TABLES_PATH "p2.t5.tmp"
#define DBG_P2_TABLE6_FNAME   DBG_TABLES_PATH "p2.t6.tmp"

#define DBG_PRUNED_TABLE2_FNAME   DBG_TABLES_PATH "pruned.t2.tmp"

#define DBG_LP_TABLE1_FNAME   DBG_TABLES_PATH "lp.t1.tmp"
#define DBG_LP_TABLE2_FNAME   DBG_TABLES_PATH "lp.t2.tmp"
#define DBG_LP_TABLE3_FNAME   DBG_TABLES_PATH "lp.t3.tmp"
#define DBG_LP_TABLE4_FNAME   DBG_TABLES_PATH "lp.t4.tmp"
#define DBG_LP_TABLE5_FNAME   DBG_TABLES_PATH "lp.t5.tmp"
#define DBG_LP_TABLE6_FNAME   DBG_TABLES_PATH "lp.t6.tmp"
#define DBG_LP_TABLE7_FNAME   DBG_TABLES_PATH "lp.t7.tmp"


template<typename T>
bool DbgReadTableFromFile( ThreadPool& pool, const char* path, uint64& outEntryCount, T* entries, bool unBuffered = false );

template<typename T>
void DbgWriteTableToFile( ThreadPool& pool, const char* path, uint64 entryCount, const T* entries, bool unBuffered = false );

void DumpTestProofs( const MemPlotContext& cx, const uint64 f7Index );
void WritePhaseTableFiles( MemPlotContext& cx );

void DbgReadPhase2MarkedEntries( MemPlotContext& cx );
void DbgWritePhase2MarkedEntries( MemPlotContext& cx );

FILE* CreateHashFile( const char* fileName );
void PrintHash( FILE* file, uint64 index, const void* src, size_t size );

//-----------------------------------------------------------
template<typename T>
inline bool DbgReadTableFromFile( ThreadPool& pool, const char* path, uint64& outEntryCount, T* entries, bool unBuffered )
{
    ASSERT( path    );
    ASSERT( entries );

    Log::Line( "Reading table file '%s'.", path );
    auto timer = TimerBegin();

    FileStream file;
    if( !file.Open( path, FileMode::Open, FileAccess::Read, unBuffered ? FileFlags::NoBuffering : FileFlags::None ) )
    {
        Log::Line( "Error: Failed to open file." );
        return false;
    }
    
    size_t blockSize = file.BlockSize();

    /// Prepare an aligned block buffer
    static void* block = nullptr;
    if( !block )
    {
        int r = posix_memalign( &block, blockSize, blockSize );
        if( r != 0 )
        {
            Log::Line( "Failed to get aligned block with error %d.", r );
            return false;
        }
    }
    ASSERT( block );
    memset( block, 0, blockSize );
    
    if( file.Read( block, blockSize ) != (ssize_t)blockSize )
    {
        Log::Line( "Failed to read entry count." );
        return false;
    }

    file.Close();

    outEntryCount = 0;
    uint64 entryCount = *(uint64*)block;
    if( entryCount == 0 )
    {
        Log::Line( "Invalid entry count." );
        return false;
    }

    ///
    /// Parallel read
    ///

    struct ReadJob
    {
        FileStream file = FileStream();
        size_t     readSize;
        byte*      buffer;
        bool       success;
    };

    ReadJob jobs[MAX_THREADS] = {};

    const uint threadCount = pool.ThreadCount();
    ASSERT( threadCount <= MAX_THREADS );

    const size_t totalSize       = sizeof( T ) * entryCount;
    const size_t blockCount      = totalSize / blockSize;
    const size_t blocksPerThread = blockCount / threadCount;
    const size_t totalBlockSize  = blocksPerThread * threadCount * blockSize;
    const size_t remainder       = totalSize - totalBlockSize;
    
    
    byte*  buffer = (byte*)entries;

    for( uint i = 0; i < threadCount; i++ )
    {
        ReadJob& job = jobs[i];

        const size_t offset = i * blocksPerThread * blockSize;

        if( !job.file.Open( path, FileMode::Open, FileAccess::Read ) )
        {
            Log::Line( "Error: Failed to open file for reading." );
            return false;
        }

        if( !job.file.Seek( (int64)(blockSize + offset), SeekOrigin::Begin ) )
        {
            Log::Line( "Error: Failed to seek file to read position." );
            return false;
        }

        job.buffer   = buffer + offset;
        job.readSize = blocksPerThread * blockSize;
        job.success  = false;
    }
    

    pool.RunJob( (JobFunc)[]( void* pdata ) {

        ReadJob& job  = *(ReadJob*)pdata;

        byte*  buffer = job.buffer;
        size_t size   = job.readSize;
        
        do {
            ssize_t read = job.file.Read( buffer, size );
            ASSERT( (size_t)read <= size );
            
            if( read == 0 )
            {
                Log::Line( "Failed to ready any bytes." );
                return;
            }
            else if( read < 0 )
            {
                Log::Line( "Failed to read with error: %d.", job.file.GetError() );
                return;
            }

            buffer += read;
            size -= (size_t)read;

        } while( size );

        job.success = true;

    }, jobs, threadCount, sizeof( ReadJob ) );

    for( uint i = 0; i < threadCount; i++ )
    {
        if( !jobs[i].success )
        {
            Log::Line( "Error: Failed to read table file." );
            return false;
        }
    }

    // Read remainder
    if( remainder )
    {
        if( !file.Open( path, FileMode::Open, FileAccess::Read ) )
        {
            Log::Line( "Error: Failed to open file to write remainder." );
            return false;
        }

        if( !file.Seek( (int64)(blockSize + totalBlockSize), SeekOrigin::Begin ) )
        {
            Log::Line( "Error: Failed to seek to remainder read position." );
            return false;
        }

        byte* remainderBuffer = buffer + totalBlockSize;

        size_t szRemainder = remainder;
        do
        {
            ssize_t read = file.Read( remainderBuffer, szRemainder );

            if( read < 1 )
            {
                Log::Line( "Error: Failed to read remainder." );
                return false;
            }

            remainderBuffer += read;
            szRemainder -= (size_t)read;

        } while( szRemainder );
    }

    
    double elapsed = TimerEnd( timer );
    Log::Line( "Finished reading table in %.2lf seconds.", elapsed );

    outEntryCount = entryCount;
    return true;
}


//-----------------------------------------------------------
template<typename T>
inline void DbgWriteTableToFile( ThreadPool& pool, const char* path, uint64 entryCount, const T* entries, bool unBuffered )
{
    if( entryCount < 1 )
        return;

    const FileFlags flags = unBuffered ? FileFlags::None : FileFlags::NoBuffering; 

    Log::Line( "Started writing table file @ %s", path );
    auto timer = TimerBegin();

    FileStream file;
    if( !file.Open( path, FileMode::Create, FileAccess::Write ,flags ) )
    {
        Log::Line( "Error: Failed to open table file '%s'.", path );
        return;
    }

    // Reserve the total size
    const size_t size        = sizeof( T ) * entryCount;

    const size_t blockSize   = file.BlockSize();
    const size_t totalBlocks = size / blockSize;

    ASSERT( size > blockSize );

    if( !file.Reserve( blockSize + CDiv( size, blockSize ) ) )
    {
        Log::Line( "Failed to reserve size with error %d for table '%s'.", file.GetError(), path );
    }

    /// Write the entry count as a whole block
    static void* block = nullptr;
    if( !block )
    {
        int r = posix_memalign( &block, blockSize, blockSize );
        if( r != 0 )
        {
            Log::Line( "Failed to get aligned block with error %d.", r );
            return;
        }
    }
    ASSERT( block );
    
    memset( block, 0, blockSize );
    *((uint64*)block) = entryCount;

    if( file.Write( block, blockSize ) != (ssize_t)blockSize )
    {
        Log::Line( "Error: Failed to write count on table file '%s'.", path );
        return;
    }
    file.Close();

    ///
    /// Multi-threaded writing
    ///
    struct WriteJob
    {
        FileStream file     ;
        byte*      buffer   ;
        size_t     writeSize;
        size_t     blockSize;
    };
    
    #if __GNUC__ > 7
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wclass-memaccess"
    #endif

    WriteJob jobs[MAX_THREADS];
    memset( jobs, 0, sizeof( jobs ) );

    #if __GNUC__ > 7
    #pragma GCC diagnostic pop
    #endif

    const uint threadCount = pool.ThreadCount();
    ASSERT( threadCount <= MAX_THREADS );
   
    const size_t blocksPerThread = totalBlocks / threadCount;
    const size_t threadWriteSize = blocksPerThread * blockSize;

    for( uint i = 0; i < threadCount; i++ )
    {
        auto& job = jobs[i];

        const size_t offset = threadWriteSize * i;

        // Open a new handle to the file and seek it to the correct position
        job.file = FileStream();
        if( !job.file.Open( path, FileMode::Open, FileAccess::Write, flags ) )
        {
            Log::Line( "Error: Failed to open table file '%s'.", path );
            return;
        }

        if( !job.file.Seek( (int64)(blockSize + offset), SeekOrigin::Begin ) )
        {
            Log::Line( "Error: Failed to seek table file '%s'.", path );
            return;
        }

        job.buffer    = ((byte*)entries) + offset;
        job.writeSize = threadWriteSize;
        job.blockSize = blockSize;
    }

    pool.RunJob( (JobFunc)[]( void* pdata ) {

        WriteJob& job = *(WriteJob*)pdata;

        size_t size = job.writeSize;

        // Write blocks until we run out
        while( size > 0 )
        {
            ssize_t written = job.file.Write( job.buffer, size );
            if( written < 0 )
            {
                Log::Line( "Error: Write failure." );
                return;
            }
            
            ASSERT( size >= (size_t)written );

            size       -= (size_t)written;
            job.buffer += written;
        }

        job.file.Close();

    }, jobs, threadCount, sizeof( WriteJob ) );
    
    // Write any remainder
    const size_t totalThreadWrite = threadWriteSize * threadCount;
    size_t       remainder        = size - totalThreadWrite;

    // Re-open without block-aligned writes
    if( remainder > 0 )
    {
        if( !file.Open( path, FileMode::Open, FileAccess::Write ) )
        {
            Log::Line( "Error: Failed to open table file '%s'.", path );
            return;
        }

        const size_t seek = blockSize + totalThreadWrite;

        if( !file.Seek( (int64)seek, SeekOrigin::Begin ) )
        {
            Log::Line( "Error: Failed to seek table file." );
            return;
        }

        byte* buffer = ((byte*)entries) + totalThreadWrite;

        do {

            ssize_t written = file.Write( buffer, remainder );
            if( written <= 0 )
            {
                Log::Line( "Error: Failed to write final data to table file." );
                file.Close();
                return;
            }

            ASSERT( written <= (ssize_t)remainder );
            remainder -= (size_t)written;

        } while( remainder > 0 );
        
        file.Close();
    }


    const double elapsed = TimerEnd( timer );
    Log::Line( "Finished writing table file in %.2lf seconds.", elapsed );
}



//-----------------------------------------------------------
template<typename T>
inline bool DbgVerifySorted( const uint64 entryCount, const T* entries )
{
    ASSERT( entryCount );
    
    T last = entries[0];
    for( uint64 i = 1; i < entryCount; i++ )
    {
        const T e = entries[i];
        
        if( last >= e )
        {
            ASSERT(0);
            return false;
        }
        last = e;
    }

    return true;
}


