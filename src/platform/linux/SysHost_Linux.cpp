#include "SysHost.h"
#include "Platform.h"
#include "Util.h"

#include <sys/random.h>
#include <execinfo.h>
#include <signal.h>
#include <atomic>

#if _DEBUG
    #include "util/Log.h"
#endif

std::atomic<bool> _crashed = false;

//-----------------------------------------------------------
size_t SysHost::GetPageSize()
{
    return (size_t)getpagesize();
}

//-----------------------------------------------------------
size_t SysHost::GetTotalSystemMemory()
{
    const size_t pageSize = GetPageSize();
    return (size_t)get_phys_pages() * pageSize;
}

//-----------------------------------------------------------
size_t SysHost::GetAvailableSystemMemory()
{
    const size_t pageSize = GetPageSize();
    return (size_t)get_avphys_pages() * pageSize;
}

//-----------------------------------------------------------
void* SysHost::VirtualAlloc( size_t size, bool initialize )
{
    // Align size to page boundary
    const size_t pageSize = GetPageSize();

    size = RoundUpToNextBoundary( size, (int)pageSize );

    // #TODO: Don't use a whole page size. But provide a VirtualAllocAligned for the block-aligned allocations
    // Add one page to store our size (yup a whole page for it...)
    size += pageSize;

    void* ptr = mmap( NULL, size, 
        PROT_READ | PROT_WRITE, 
        MAP_ANONYMOUS | MAP_PRIVATE,
        -1, 0
    );

    if( ptr == MAP_FAILED )
    {
        #if _DEBUG
            const int err = errno;
            Log::Line( "Error: mmap() returned %d (0x%x).", err, err );
            ASSERT( 0 );
        #endif
    
        return nullptr;
    }

    if( initialize )
    {
        byte* page = (byte*)ptr;

        const size_t pageCount = size / pageSize;
        const byte*  endPage   = page + pageCount * pageSize;

        do
        {
            *page = 0;
            page += pageSize;
        } while( page < endPage );
    }

    *((size_t*)ptr) = size;

    return ((byte*)ptr)+pageSize;
}

//-----------------------------------------------------------
void SysHost::VirtualFree( void* ptr )
{
    ASSERT( ptr );
    if( !ptr )
        return;

    const size_t pageSize = GetPageSize();

    byte* realPtr    = ((byte*)ptr) - pageSize;
    const size_t size = *((size_t*)realPtr);

    munmap( realPtr, size );
}

//-----------------------------------------------------------
bool SysHost::VirtualProtect( void* ptr, size_t size, VProtect flags )
{
    ASSERT( ptr );

    int prot = 0;

    if( IsFlagSet( flags, VProtect::NoAccess ) )
    {
        prot = PROT_NONE;
    }
    else
    {
        if( IsFlagSet( flags, VProtect::Read ) )
            prot |= PROT_READ;
        if( IsFlagSet( flags, VProtect::Write ) )
            prot |= PROT_WRITE;
    }

    int r = mprotect( ptr, size, prot );
    ASSERT( !r );

    return r == 0;
}

//-----------------------------------------------------------
uint64 SysHost::SetCurrentProcessAffinityMask( uint64 mask )
{
    return SetCurrentThreadAffinityMask( mask );
}

//-----------------------------------------------------------
uint64 SysHost::SetCurrentThreadAffinityMask( uint64 mask )
{
    pthread_t thread = pthread_self();

    cpu_set_t cpuSet;
    CPU_ZERO( &cpuSet );
    CPU_SET( mask, &cpuSet );

    int r = pthread_setaffinity_np( thread, sizeof(cpu_set_t), &cpuSet );
    if( r != 0 )
        return 0;

    r = pthread_getaffinity_np( thread, sizeof(cpu_set_t), &cpuSet );
    if( r != 0 )
        return 0;

    return mask;
}

//-----------------------------------------------------------
void CrashHandler( int signal )
{
    // Only let the first thread handle the crash
    bool crashed = false;
    if( !_crashed.compare_exchange_strong( crashed, true, 
        std::memory_order_release, std::memory_order_relaxed ) )
    {
        return;
    }

    const size_t MAX_POINTERS = 256;
    void* stackTrace[256] = { 0 };

    fprintf( stderr, "*** Crashed! ***\n" );
    fflush( stderr );

    int traceSize = backtrace( stackTrace, (int)MAX_POINTERS );
    backtrace_symbols_fd( stackTrace, traceSize, fileno( stderr ) );
    fflush( stderr );
    
    exit( 1 );
}

//-----------------------------------------------------------
void SysHost::InstallCrashHandler()
{
    signal( SIGSEGV, CrashHandler ); 
}

//-----------------------------------------------------------
void SysHost::Random( byte* buffer, size_t size )
{
    // See: https://man7.org/linux/man-pages/man2/getrandom.2.html

    ssize_t sizeRead;
    byte* writer = buffer;
    const byte* end = writer + size;

    const size_t BLOCK_SIZE = 256;
    
    while( writer < end )
    {
        size_t readSize = (size_t)(end - writer);
        if( readSize > BLOCK_SIZE )
            readSize = BLOCK_SIZE;
            
        sizeRead = getrandom( writer, readSize, 0 );

        // Should never get EINTR, but docs say to check anyway.
        if( sizeRead < 0 && errno != EINTR )
            Fatal( "getrandom syscall failed." );

        writer += (size_t)sizeRead;
    }
}

