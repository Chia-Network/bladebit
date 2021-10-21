#include "io/FileStream.h"
#include "Util.h"
#include "util/Log.h"

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

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

    if( file._fd >= 0 )
        return false;

    if( access == FileAccess::None )
        access = FileAccess::Read;

    mode_t fmode = 0;

    int fdFlags = access == FileAccess::Read  ? O_RDONLY :
                  access == FileAccess::Write ? O_WRONLY : O_RDWR;
        
    fdFlags |= mode == FileMode::Create ? O_CREAT  :
               mode == FileMode::Append ? O_APPEND : 0;

    #if PLATFORM_IS_LINUX
        if( IsFlagSet( flags, FileFlags::NoBuffering ) )
            fdFlags |= O_DIRECT | O_SYNC;

        if( IsFlagSet( flags, FileFlags::LargeFile )  )
            fdFlags |= O_LARGEFILE;
    #endif

    if( mode == FileMode::Create )
        fmode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;

    int fd = open( path, fdFlags, fmode );
    if( fd < 0 )
        return false;

    #if PLATFORM_IS_MACOS
        if( IsFlagSet( flags, FileFlags::NoBuffering ) )
        {
            int r = fcntl( fd, F_NOCACHE, 1 );
            if( r == -1 )
            {
                file._error = errno;
                close( fd );
                return false;
            }
        }
    #endif

    // Get the block size (useful when using O_DIRECT)
    int blockSize = 0;
    {
        struct stat fs;
        int r = fstat( fd, &fs );

        if( r == 0 )
        {
            blockSize = (size_t)fs.st_blksize;
        }
        else
        {
            file._error = errno;
            close( fd );

            return false;
        }

        ASSERT( blockSize > 0 );
    }
    
    file._fd            = fd;
    file._blockSize     = (size_t)blockSize;
    file._writePosition = 0;
    file._readPosition  = 0;
    file._access        = access;
    file._flags         = flags;
    file._error         = 0;

    return true;
}

//-----------------------------------------------------------
void FileStream::Close()
{
    if( _fd <= 0 )
        return;

    #if _DEBUG
    int r =
    #endif
    close( _fd );
    
    ASSERT( !r );

    _fd            = -1;
    _writePosition = 0;
    _readPosition  = 0;
    _access        = FileAccess::None;
    _error         = 0;
    _blockSize     = 0;
}

//-----------------------------------------------------------
ssize_t FileStream::Read( void* buffer, size_t size )
{
    ASSERT( buffer );

    if( buffer == nullptr )
        return -1;

    if( ! IsFlagSet( _access, FileAccess::Read ) )
        return -1;

    if( _fd < 0 )
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
