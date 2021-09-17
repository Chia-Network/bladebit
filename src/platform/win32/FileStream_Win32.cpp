#include "io/FileStream.h"
#include "Util.h"
#include "util/Log.h"

#include <locale> 
#include <codecvt> 

//----------------------------------------------------------
bool FileStream::Open( const char* path, FileMode mode, FileAccess access, FileFlags flags )
{
    return Open( path, *this, mode, access, flags );
}

//----------------------------------------------------------
bool FileStream::Open( const char* path, FileStream& file, FileMode mode, FileAccess access, FileFlags flags )
{
    if( path == nullptr )
        return false;

    if( file.HasValidFD() )
        return false;

    size_t len = strlen( path );
    if( len < 1 )
        return false;

    if( len > std::numeric_limits<int>::max() )
    {
        Log::Error( "File path is too long." );
        return false;
    }
    
    // Encode utf-8 path to wchar_t
    const size_t BUF16_STACK_LEN = 1024;
    wchar_t path16Stack[BUF16_STACK_LEN];

    const int requiredLen16 = MultiByteToWideChar( CP_UTF8, MB_PRECOMPOSED, path, (int)len, NULL, 0 );

    if( requiredLen16 < 1 )
    {
        Log::Error( "Could not get encoded file path length." );
        return false;
    }

    wchar_t* path16 = nullptr;

    if( requiredLen16 <= BUF16_STACK_LEN )
    {
        path16 = path16Stack;
    }
    else
    {
        path16 = (wchar_t*)malloc( sizeof( wchar_t ) * requiredLen16 );
        if( !path16 )
        {
            Log::Error( "Failed to allocate file path buffer." );
            return false;
        }
    }

    const int numEncoded = MultiByteToWideChar(
        CP_UTF8, MB_PRECOMPOSED, path, (int)len, path16, requiredLen16
    );

    ASSERT( numEncoded == requiredLen16 );

    if( access == FileAccess::None )
        access = FileAccess::Read;

    const DWORD dwShareMode           = FILE_SHARE_READ;
    const DWORD dwCreationDisposition = mode == FileMode::Create ? CREATE_ALWAYS : 
                                        mode == FileMode::Open   ? OPEN_ALWAYS   :
                                                                   OPEN_EXISTING;
    DWORD dwFlags  = FILE_ATTRIBUTE_NORMAL;
    DWORD dwAccess = 0;

    if( IsFlagSet( flags, FileFlags::NoBuffering ) )
        dwFlags = FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH;

    if( IsFlagSet( access, FileAccess::Read ) )
        dwAccess = GENERIC_READ;
    
    if( IsFlagSet( access, FileAccess::Write ) )
        dwAccess |= GENERIC_WRITE;

    HANDLE fd = CreateFile( path16, dwAccess, dwShareMode, NULL,
                            dwCreationDisposition, dwFlags, NULL );

    if( fd != INVALID_HANDLE_VALUE )
    {
        // #TODO: Get the block (cluster) size
        //int blockSize = 0;

        file._fd            = fd;
        //file._blockSize     = (size_t)blockSize;
        file._writePosition = 0;
        file._readPosition  = 0;
        file._access        = access;
        file._flags         = flags;
        file._error         = 0;

        // #TODO: Seek to end if appending.
    }
    else
    {
        // #TODO: Use GetLastError report error in utf8
        file. _error = (int)GetLastError();
    }

    if( path16 != path16Stack )
        free( path16 );

    return fd != INVALID_HANDLE_VALUE;
}

//-----------------------------------------------------------
void FileStream::Close()
{
    if( !HasValidFD() )
        return;

    #if _DEBUG
        BOOL r = 
    #endif

    CloseHandle( _fd );
    
    #if _DEBUG
        ASSERT( r );
    #endif

    _fd            = INVALID_HANDLE_VALUE;
    _writePosition = 0;
    _readPosition  = 0;
    _access        = FileAccess::None;
    _error         = 0;
    //_blockSize     = 0;
}

//-----------------------------------------------------------
ssize_t FileStream::Read( void* buffer, size_t size )
{
    ASSERT( buffer );

    if( buffer == nullptr )
        return -1;

    if( !IsFlagSet( _access, FileAccess::Read ) )
        return -1;

    if( HasValidFD() )
        return -1;

    if( size < 1 )
        return 0;
    
    const ssize_t sizeRead = read( _fd, buffer, size );
    if( sizeRead >= 0 )
        _readPosition += (size_t)sizeRead;
    else 
        _error = errno;

    return sizeRead;
}

//-----------------------------------------------------------
ssize_t FileStream::Write( const void* buffer, size_t size )
{
    ASSERT( buffer );
    ASSERT( size   );
    ASSERT( _fd    );
    
    if( buffer == nullptr )
        return -1;

    if( ! IsFlagSet( _access, FileAccess::Write ) )
        return -1;

    if( _fd < 0 )
        return -1;

    if( size < 1 )
        return 0;

    // Note that this can return less than size if size > SSIZE_MAX
    ssize_t written = write( _fd, buffer, size );
    
    if( written >= 0 )
        _writePosition += (size_t)written;
    else 
        _error = errno;
    
    return written;
}

//----------------------------------------------------------
bool FileStream::Reserve( ssize_t size )
{
    #if PLATFORM_IS_LINUX
        int r = posix_fallocate( _fd, 0, (off_t)size );
        if( r != 0 )
        {
            _error = errno;
            return false;
        }
    #else
    
    // #TODO: Use F_PREALLOCATE on macOS
    // #SEE: https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fcntl.2.html
        return false;
    #endif
    return true;
}

//----------------------------------------------------------
bool FileStream::Seek( int64 offset, SeekOrigin origin )
{
    if( !IsOpen() )
        return false;

    int whence;
    switch( origin )
    {
        case SeekOrigin::Begin  : whence = SEEK_SET; break;
        case SeekOrigin::Current: whence = SEEK_CUR; break;
        case SeekOrigin::End    : whence = SEEK_END; break;
        default: return false;
    }

    off_t r = lseek( _fd, (off_t)offset, whence );
    if( r == -1 )
    {
        _error = errno;
        return false;
    }

    return true;
}

//-----------------------------------------------------------
bool FileStream::Flush()
{
    if( !IsOpen() )
        return false;

    int r = fsync( _fd );

    if( r )
    {
        _error = errno;
        return false;
    }

    return true;
}

//-----------------------------------------------------------
bool FileStream::IsOpen() const
{
    return _fd >= 0;
}

//-----------------------------------------------------------
bool FileStream::Exists( const char* path )
{
    struct stat fileStat;
    if( lstat( path, &fileStat ) == 0 )
    {
        // #TODO: Check if it is a folder...
        
        return true;
    }

    return false;
}
