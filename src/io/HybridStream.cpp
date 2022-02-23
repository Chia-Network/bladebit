#include "HybridStream.h"
#include "util/Util.h"

//-----------------------------------------------------------
bool HybridStream::Open( void* memory, ssize_t memorySize, const char* path, FileMode mode, FileAccess access, FileFlags flags )
{
    ASSERT( memory );
    FatalIf( memorySize < 0, "Invalid memory size." );

    // Open the backing file
    if( !_file.Open( path, mode, access, flags ) )
    {
        _error = _file.GetError();
        return false;
    }

    _memory   = (byte*)memory;
    _memSize  = (size_t)memorySize;
    _position = 0;
    _error    = 0;

    return true;
}

// //-----------------------------------------------------------
// bool HybridStream::Open( ssize_t memorySize, const char* path, FileMode mode, FileAccess access, FileFlags flags )
// {
//     
// }

//-----------------------------------------------------------
ssize_t HybridStream::Read( void* buffer, size_t size )
{
    if( size < 1 )
        return 0;

    ASSERT( buffer );
    if( !buffer )
        return -14; // EFAULT

    // #TODO: Check if write-only

    // #TODO: Prevent overflow ssize_t max

    // Bring down to our max read size
    size = std::min( size, (size_t)std::numeric_limits<ssize_t>::max() );

    // Check if we can read from memory
    size_t read = 0;
    if( _position < _memSize )
    {
        read = std::min( _memSize - _position, size );
        
        memcpy( buffer, _memory + _position, read );

        _position += read;
        size      -= read;
        buffer     = ((byte*)buffer) + read;
    }

    if( size )
    {
        ssize_t diskRead = _file.Read( buffer, size );
        if( diskRead < 0 )
        {
            _error = _file.GetError();
            return diskRead;
        }

        _position += (size_t)diskRead;
        read      += (size_t)diskRead;
    }

    return (ssize_t)read;
}


//-----------------------------------------------------------
ssize_t HybridStream::Write( const void* buffer, size_t size )
{
    if( size < 1 )
        return 0;

    ASSERT( buffer );
    if( !buffer )
        return -14; // EFAULT

    // #TODO: Check if read-only

    // #TODO: Prevent overflow ssize_t max

    // Bring down to our max write size
    size = std::min( size, (size_t)std::numeric_limits<ssize_t>::max() );

    // See if we can write to our memory region
    size_t written = 0;
    if( _position < _memSize )
    {
        written = std::min( _memSize - _position, size );
        
        memcpy( _memory + _position, buffer, written );

        _position += written;
        size      -= written;
        buffer = ((byte*)buffer) + written;
    }

    // Write any remaining data to disk
    if( size )
    {
        ssize_t diskWritten = _file.Write( buffer, size );
        if( diskWritten < 0 )
        {
            _error = _file.GetError();
            return diskWritten;
        }

        _position += (size_t)diskWritten;
        written   += (size_t)diskWritten;
    }
    
    return (ssize_t)written;
}

//-----------------------------------------------------------
bool HybridStream::Seek( int64 offset, SeekOrigin origin )
{
    switch( origin )
    {
        case SeekOrigin::Begin:
            if( offset < 0 )
            {
                ASSERT( 0 );
                _error = -22; // EINVAL
                return false;
            }
            _position = (size_t)offset;
            break;
        
        case SeekOrigin::Current:
            if( offset < 0 )
            {
                const size_t subtrahend = (size_t)std::abs( offset );
                if( subtrahend > _position )
                {
                    ASSERT( 0 );
                    _error = -22; // EINVAL
                    return false;
                }

                _position -= subtrahend;
            }
            else
            {
                if( _position + (size_t)offset < _position )
                {
                    ASSERT( 0 );
                    _error = -22; // EINVAL
                    return false;
                }

                _position += (size_t)offset;
            }
            break;
        
        case SeekOrigin::End:
        {
            const size_t end = _memSize + (size_t)_file.Size();
            if( offset < 0 )
            {
                const size_t subtrahend = (size_t)std::abs( offset );
                if( subtrahend > end )
                {
                    ASSERT( 0 );
                    _error = -22; // EINVAL
                    return false;
                }

                _position = end - (size_t)subtrahend;
            }
            else
            {
                if( _position + (ssize_t)offset > (size_t)std::numeric_limits<ssize_t>::max() )
                {
                    ASSERT( 0 );
                    _error = -22; // EINVAL
                    return false;
                }

                // Go beyond the file end
                if( !_file.Seek( (int64)offset, SeekOrigin::End ) )
                {
                    ASSERT( 0 );
                    _error = _file.GetError();
                    return false;
                }

                _position = end + (ssize_t)offset;
                return true;
            }
        }
            break;
    
        default:
            _error = -1;  // EPERM
            return false;
    }

    ASSERT( _position <= (size_t)std::numeric_limits<ssize_t>::max() );

    // Seek file
    if( _position < _memSize )
    {
        if( !_file.Seek( 0, SeekOrigin::Begin ) )
        {
            ASSERT( 0 );
            _error = _file.GetError();
            return false;
        }
    }
    else
    {
        const size_t filePos = _position - _memSize;
        
        if( !_file.Seek( filePos, SeekOrigin::Begin ) )
        {
            ASSERT( 0 );
            _error = _file.GetError();
            return false;
        }
    }

    return true;
}

//-----------------------------------------------------------
bool HybridStream::Flush()
{
    return _file.Flush();
}

//-----------------------------------------------------------
size_t HybridStream::BlockSize() const
{
    return _file.BlockSize();
}

//-----------------------------------------------------------
ssize_t HybridStream::Size()
{
    return _memSize + _file.Size();
}

//-----------------------------------------------------------
int HybridStream::GetError()
{
    // #TODO: Do not clear error here.
    //        Only clear on successful operations.
    int err = _error;
    _error = 0;
    return err;
}