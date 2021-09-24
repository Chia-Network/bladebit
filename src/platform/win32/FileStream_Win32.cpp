#include "io/FileStream.h"
#include "Util.h"
#include "util/Log.h"

#include <winioctl.h>
#include <shlwapi.h>
#pragma comment( lib, "Shlwapi.lib" )

const size_t BUF16_STACK_LEN = 1024;

//bool GetFileClusterSize( wchar_t* filePath, size_t& outClusterSize );
bool GetFileClusterSize( HANDLE hFile, size_t& outClusterSize );

wchar_t* Utf8ToUtf16( const char* utf8Str, wchar_t* stackBuffer16, const size_t stackBuf16Size );

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

   
    // Encode utf-8 path to wchar_t
    wchar_t path16Stack[BUF16_STACK_LEN];

    wchar_t* path16 = Utf8ToUtf16( path, path16Stack, BUF16_STACK_LEN );
    if( !path16 )
        return false;

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
        // Get the block (cluster) size
        size_t blockSize;

        if( !GetFileClusterSize( fd, blockSize ) )
            Log::Error( "Failed to obtain file block size. Defaulting to %llu, but writes may fail.", blockSize );

        file._fd            = fd;
        file._blockSize     = blockSize;
        file._writePosition = 0;
        file._readPosition  = 0;
        file._access        = access;
        file._flags         = flags;
        file._error         = 0;

        // #TODO: Seek to end if appending?
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
    _blockSize     = 0;
}

//-----------------------------------------------------------
ssize_t FileStream::Read( void* buffer, size_t size )
{
    ASSERT( buffer );

    if( buffer == nullptr )
        return -1;

    if( !IsFlagSet( _access, FileAccess::Read ) )
        return -1;

    if( !HasValidFD() )
        return -1;

    if( size < 1 )
        return 0;

    DWORD bytesToRead = size > std::numeric_limits<DWORD>::max() ?
                               std::numeric_limits<DWORD>::max() : 
                               (DWORD)size;

    if( IsFlagSet( _flags, FileFlags::NoBuffering ) )
    {
        // #NOTE: See comment on Write() about this.
        bytesToRead = (DWORD)( bytesToRead / _blockSize * _blockSize );
    }

    DWORD bytesRead = 0;

    // Cap size to 32-bit range
    const BOOL r = ReadFile( _fd, buffer, bytesToRead, &bytesRead, NULL );
    
    if( r )
        _readPosition += (size_t)bytesRead;
    else
    {
        _error    = (int)GetLastError();
        bytesRead = -1;
    }

    return (ssize_t)bytesRead;
}

//-----------------------------------------------------------
ssize_t FileStream::Write( const void* buffer, size_t size )
{
    ASSERT( buffer );
    ASSERT( size   );
    ASSERT( _fd    );
    
    if( buffer == nullptr )
        return -1;

    if( !IsFlagSet( _access, FileAccess::Write ) )
        return -1;

    if( !HasValidFD() )
        return -1;

    if( size < 1 )
        return 0;

    DWORD bytesToWrite = size > std::numeric_limits<DWORD>::max() ?
                                std::numeric_limits<DWORD>::max() : 
                                (DWORD)size;

    if( IsFlagSet( _flags, FileFlags::NoBuffering ) )
    {
        // We can only write in block sizes. But since the user may have
        // specified a size greater than DWORD, our clamping it to 
        // DWORD's max can cause it to become not bounded to block size,
        // even if the user's original size was block-bound.
        // So let's limit this to a block size.
        bytesToWrite = (DWORD)(bytesToWrite / _blockSize * _blockSize);
    }

    DWORD bytesWritten = 0;
    BOOL r = WriteFile( _fd, buffer, bytesToWrite, &bytesWritten, NULL );

    if( r )
        _writePosition += (size_t)bytesWritten;
    else
    {
        _error       = (int)GetLastError();
        bytesWritten = -1;
    }

    return bytesWritten;
}

//----------------------------------------------------------
bool FileStream::Reserve( ssize_t size )
{
    // #TODO: Use SetFileValidData()?
    // #See: https://docs.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-setfilevaliddata
    
    return false;
}

//----------------------------------------------------------
bool FileStream::Seek( int64 offset, SeekOrigin origin )
{
    if( !IsOpen() || !HasValidFD() )
        return false;

    DWORD whence;
    switch( origin )
    {
        case SeekOrigin::Begin  : whence = FILE_BEGIN  ; break;
        case SeekOrigin::Current: whence = FILE_CURRENT; break;
        case SeekOrigin::End    : whence = FILE_END    ; break;
        default: return false;
    }

    LARGE_INTEGER distanceToMove, newPosition;
    distanceToMove.QuadPart = offset;

    const BOOL r = ::SetFilePointerEx( _fd, distanceToMove, &newPosition, whence );

    if( !r )
        _error = GetLastError();

    _writePosition = (size_t)newPosition.QuadPart;
    _readPosition  = (size_t)newPosition.QuadPart;
    
    return (bool)r;
}

//-----------------------------------------------------------
bool FileStream::Flush()
{
    if( !IsOpen() || !HasValidFD() )
        return false;

    const BOOL r = FlushFileBuffers( _fd );

    if( !r )
        _error = GetLastError();
    
    return (bool)r;
}

//-----------------------------------------------------------
bool FileStream::IsOpen() const
{
    return HasValidFD();
}

//-----------------------------------------------------------
bool FileStream::Exists( const char* path )
{
    ASSERT( path );
    if( !path || !*path )
        return false;

    wchar_t stackBuffer[BUF16_STACK_LEN];

    wchar_t* path16 = Utf8ToUtf16( path, stackBuffer, BUF16_STACK_LEN );
    if( !path16 )
    {
        Log::Error( "FileStream::Exists() Failed to convert path to utf16." );
        return false;
    }

    bool exists = true;

    const DWORD r = GetFileAttributesW( path16 );
    
    if( r == INVALID_FILE_ATTRIBUTES )
        exists = false;

    if( path16 != stackBuffer )
        free( path16 );

    return exists;
}

//-----------------------------------------------------------
wchar_t* Utf8ToUtf16( const char* utf8Str, wchar_t* stackBuffer16, const size_t stackBuf16Size )
{
    const size_t length8 = strlen( utf8Str );

    if( length8 < 1 )
        return nullptr;

    if( length8 > std::numeric_limits<int>::max() )
    {
        Log::Error( "File path is too long." );
        return nullptr;
    }


    const int requiredLen16 = MultiByteToWideChar( CP_UTF8, MB_ERR_INVALID_CHARS, utf8Str, (int)length8, NULL, 0 ) + 1;

    if( requiredLen16 <= 1 )
    {
        Log::Error( "Could not get encoded file path length." );
        return nullptr;
    }

    wchar_t* str16 = nullptr;

    if( requiredLen16 <= stackBuf16Size )
    {
        str16 = stackBuffer16;
    }
    else
    {
        str16 = (wchar_t*)malloc( sizeof( wchar_t ) * (size_t)requiredLen16 );
        if( !str16 )
        {
            Log::Error( "Failed to allocate file path buffer." );
            return nullptr;
        }
    }

    const int numEncoded = MultiByteToWideChar(
        CP_UTF8, MB_PRECOMPOSED, 
        utf8Str, (int)length8, 
        str16, requiredLen16
    );

    ASSERT( numEncoded == requiredLen16-1 );

    str16[numEncoded] = 0;

    return str16;
}

//-----------------------------------------------------------
//bool GetFileClusterSize( wchar_t* filePath, size_t& outClusterSize )
bool GetFileClusterSize( HANDLE hFile, size_t& outClusterSize )
{
    outClusterSize = 4096;

    ASSERT( hFile != INVALID_HANDLE_VALUE );

    FILE_STORAGE_INFO info = { 0 };
    const BOOL r = GetFileInformationByHandleEx( hFile, FileStorageInfo, &info, (DWORD)sizeof( info ) );

    if( r )
        outClusterSize = info.PhysicalBytesPerSectorForPerformance;

    return (bool)r;
    //
    //ASSERT( filePath );

    //// Get path to the device
    //if (!PathStripToRootW( filePath ) )
    //    return false;
    //
    //const size_t len = std::char_traits<wchar_t>::length( filePath );

    //// #TODO: Do this properly by copying to another buffer that we're sure has enough size
    //memmove( filePath + 4, filePath, (len + 1) * sizeof( wchar_t ) );
    //filePath[0] = u'\\';
    //filePath[1] = u'\\';
    //filePath[2] = u'.' ;
    //filePath[3] = u'\\';

    //HANDLE hDevice = INVALID_HANDLE_VALUE;

    //// #See: https://docs.microsoft.com/en-us/windows/win32/devio/calling-deviceiocontrol
    //hDevice = CreateFileW( filePath,    // drive to open
    //            0,                      // no access to the drive
    //            FILE_SHARE_READ |       // share mode
    //            FILE_SHARE_WRITE, 
    //            NULL,                   // default security attributes
    //            OPEN_EXISTING,          // disposition
    //            0,                      // file attributes
    //            NULL ); 

    //if( hDevice == INVALID_HANDLE_VALUE )
    //    return false;

    //STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR desc = { 0 };
    //DWORD bytesReturned;

    //STORAGE_PROPERTY_QUERY spq = { StorageAccessAlignmentProperty, PropertyStandardQuery }; 

    //BOOL r = DeviceIoControl(
    //            hDevice,
    //            IOCTL_STORAGE_QUERY_PROPERTY,
    //            &spq, sizeof( spq ),
    //            &desc, sizeof( desc ),
    //            &bytesReturned,
    //            NULL );
    //
    //// MS recommends the use the physical sector size
    //// #See: https://docs.microsoft.com/en-us/windows/win32/fileio/file-buffering
    //if( r )
    //    outClusterSize = desc.BytesPerPhysicalSector;
    //else
    //{
    //    const DWORD err = GetLastError();
    //    Log::Error( "Error getting block size: %d (0x%x)", err );
    //}

    //CloseHandle( hDevice );

    //return (bool)r;
}