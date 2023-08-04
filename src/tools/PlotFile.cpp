#include "PlotReader.h"
#include "ChiaConsts.h"

///
/// Plot Files
///
//-----------------------------------------------------------
bool IPlotFile::ReadHeader( int& error )
{
    error = 0;

    _version = PlotVersion::v1_0;

    // Check if the plot file is v2
    {
        uint32 v2Magic = 0;
        if( Read( sizeof( v2Magic ), &v2Magic ) != sizeof( v2Magic ) )
        {
            error = GetError();
            return false;
        }

        // It's > v1
        if( v2Magic == CHIA_PLOT_V2_MAGIC )
        {
            // Get the plot version
            if( Read( sizeof( _version ), &_version ) != sizeof( _version ) )
            {
                error = GetError();
                return false;
            }

            // For now only version 2 is supported
            if( _version != PlotVersion::v2_0 )
            {
                error = -1;
                return false;
            }
        }
        else
        {
            // Check for v1 magic
            char v1Magic[sizeof( kPOSMagic )-1] = { 0 };
            memcpy( v1Magic, &v2Magic, sizeof( v2Magic ) );

            const size_t v1ReadMagicSize = sizeof( v1Magic ) - sizeof( v2Magic );
            if( Read( v1ReadMagicSize, &v1Magic[sizeof(v2Magic)] ) != v1ReadMagicSize )
            {
                error = GetError();
                return false;
            }

            // Ensure it is indeed v1 magic
            if( !MemCmp( v1Magic, kPOSMagic, sizeof( v1Magic ) ) )
            {
                error = -1;       // #TODO: Set actual user error
                return false;
            }
        }
    }
    
    // Plot Id
    {
        if( Read( sizeof( _header.id ), _header.id ) != sizeof( _header.id ) )
            return false;

        // char str[65] = { 0 };
        // size_t numEncoded = 0;
        // BytesToHexStr( _header.id, sizeof( _header.id ), str, sizeof( str ), numEncoded );
        // ASSERT( numEncoded == sizeof( _header.id ) );
        // _idString = str;
    }

    // K
    {
        byte k = 0;
        if( Read( 1, &k ) != 1 )
            return false;

        _header.k = k;
    }

    // Format Descritption
    if( _version < PlotVersion::v2_0 )
    {
        const uint formatDescSize =  ReadUInt16();
        FatalIf( formatDescSize != sizeof( kFormatDescription ) - 1, "Invalid format description size." );

        char desc[sizeof( kFormatDescription )-1] = { 0 };
        if( Read( sizeof( desc ), desc ) != sizeof( desc ) )
            return false;
        
        if( !MemCmp( desc, kFormatDescription, sizeof( desc ) ) )
        {
            error = -1; // #TODO: Set proper user error
            return false;
        }
    }

    // Memo
    {
        uint memoSize = ReadUInt16();
        if( memoSize > sizeof( _header.memo ) )
        {
            error = -1; // #TODO: Set proper user error
            return false;
        } 

        _header.memoLength = memoSize;

        if( Read( memoSize, _header.memo ) != memoSize )
        {
            error = -1; // #TODO: Set proper user error
            return false;
        }

        // char str[BB_PLOT_MEMO_MAX_SIZE*2+1] = { 0 };
        // size_t numEncoded = 0;
        // BytesToHexStr( _memo, memoSize, str, sizeof( str ), numEncoded );
        
        // _memoString = str;
    }

    // Flags
    if( _version >= PlotVersion::v2_0 )
    {
        if( Read( sizeof( _header.flags ), &_header.flags ) != sizeof( _header.flags ) )
        {
            error = GetError();
            return false;
        }

        // Compression level
        if( IsFlagSet( _header.flags, PlotFlags::Compressed ) )
        {
            if( Read( sizeof( _header.compressionLevel ), &_header.compressionLevel ) != sizeof( _header.compressionLevel ) )
            {
                error = GetError();
                return false;
            }
        }
    }

    // Table pointers
    if( Read( sizeof( _header.tablePtrs ), _header.tablePtrs ) != sizeof( _header.tablePtrs ) )
    {
        error = GetError();
        return false;
    }

    for( int i = 0; i < 10; i++ )
        _header.tablePtrs[i] = Swap64( _header.tablePtrs[i] );

    // Table sizes
    if( _version >= PlotVersion::v2_0 )
    {
        if( Read( sizeof( _header.tableSizes ), _header.tableSizes ) != sizeof( _header.tableSizes ) )
        {
            error = GetError();
            return false;
        }
    }

    // What follows is table data
    return true;
}


