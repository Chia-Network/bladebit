#include "PlotReader.h"
#include "io/FileStream.h"
#include "ChiaConsts.h"
#include "util/BitView.h"
#include "plotshared/PlotTools.h"
#include "memplot/CTables.h"
#include "DTables.h"

///
/// Plot Reader
///

//-----------------------------------------------------------
PlotReader::PlotReader( IPlotFile& plot )
    : _plot( plot )
    , _c3DeltasBuffer( bbcalloc<byte>( kCheckpoint1Interval ) )
{}

//-----------------------------------------------------------
PlotReader::~PlotReader()
{
    free( _c3DeltasBuffer );
}

//-----------------------------------------------------------
// uint64 PlotReader::GetC3ParkCount() const
// {
//     // We know how many C3 parks there are by how many 
//     // entries we have in the C1 table - 1 (extra 0 entry added)
//     // However, to make sure this is the case, we'll have to 
//     // read-in all C1 entries and ensure we hit an empty one,
//     // to ensure we don't run into dead/alignment-space
//     const size_t c1Size = _plot.TableSize( PlotTable::C1 );
// }

// //-----------------------------------------------------------
// uint64 PlotReader::GetF7EntryCount() const
// {
// }

//-----------------------------------------------------------
bool PlotReader::ReadC3Park( uint64 parkIndex, uint64* f7Buffer )
{
    const size_t f7SizeBytes    = CDiv( _plot.K(), 8 );
    const size_t c3ParkSize     = CalculateC3Size();
    const uint64 c1Address      = _plot.TableAddress( PlotTable::C1 );
    const uint64 c3Address      = _plot.TableAddress( PlotTable::C3 );
    const uint64 c1EntryAddress = c1Address + parkIndex * f7SizeBytes;
    const uint64 parkAddress    = c3Address + parkIndex * c3ParkSize;

    // First we need to read the root F7 entry for the park, 
    // which is at in C1 table.
    if( !_plot.Seek( SeekOrigin::Begin, (int64)parkAddress ) )
        return false;

    uint64 c1 = 0;
    if( _plot.Read( f7SizeBytes, &c1 ) != (ssize_t)f7SizeBytes )
        return false;

    c1 = Swap64( c1 );

    // Read the park into our buffer
    if( !_plot.Seek( SeekOrigin::Begin, (int64)parkAddress ) )
        return false;

    // Read the size of the compressed C3 deltas
    uint16 deltasSize = 0;
    if( _plot.Read( sizeof( uint16 ), &deltasSize ) != (ssize_t)sizeof( uint16 ) )
        return false;

    deltasSize = Swap64( deltasSize );
    if( deltasSize > c3ParkSize )
        return false;

    memset( _c3ParkBuffer, 0, sizeof( _c3ParkBuffer ) );
    if( _plot.Read( deltasSize, _c3ParkBuffer ) != (ssize_t)deltasSize )
        return false;

    // Now we can read the f7 deltas from the C3 park
    const size_t deltasSize = 0;
    
    const size_t err = FSE_decompress_usingDTable( 
                        _c3DeltasBuffer, kCheckpoint1Interval, 
                        _c3ParkBuffer, deltasSize, 
                        (const FSE_DTable*)DTable_C3 );

    // #TODO: Set error message locally
    if( FSE_isError(err) )
        return false;

    for( uint32 i = 0; i < kCheckpoint1Interval; i++ )
        if( _c3DeltasBuffer[i] == 0xFF )
            return false;

    // Unpack deltas into absolute values
    memset( f7Buffer, 0, kCheckpoint1Interval * sizeof( uint64 ) );

    uint64 f7 = c1;
    f7Buffer[0] = f7;

    for( uint32 i = 1; i < kCheckpoint1Interval; i++ )
    {
        uint64 delta = _c3DeltasBuffer[i-1];
    }
}

///
/// Plot Files
///
// #TODO: Move to other source files
//-----------------------------------------------------------
bool IPlotFile::ReadHeader( int& error )
{
    error = 0;

    // Magic
    {
        char magic[sizeof( kPOSMagic )-1] = { 0 };
        if( Read( sizeof( magic ), magic ) != sizeof( magic ) )
            return false;
        
        if( !MemCmp( magic, kPOSMagic, sizeof( magic ) ) )
        {
            error -1;       // #TODO: Set actual user error
            return false;
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

    // Table pointers
    if( Read( sizeof( _header.tablePtrs ), _header.tablePtrs ) != sizeof( _header.tablePtrs ) )
    {
        error = -1; // #TODO: Set proper user error
        return false;
    }

    for( int i = 0; i < 10; i++ )
        _header.tablePtrs[i] = Swap64( _header.tablePtrs[i] );

    // What follows is table data
}


///
/// Memory Plot
///
//-----------------------------------------------------------
MemoryPlot::MemoryPlot()
    : _bytes( nullptr, 0 )
{}

//-----------------------------------------------------------
MemoryPlot::~MemoryPlot()
{
    if( _bytes.values )
        SysHost::VirtualFree( _bytes.values );

    _bytes = Span<byte>( nullptr, 0 );
}

//-----------------------------------------------------------
bool MemoryPlot::Open( const char* path )
{
    ASSERT( path );
    if( !path )
        return;

    if( IsOpen() )
        return false;

    FileStream file;
    if( !file.Open( path, FileMode::Open, FileAccess::Read, FileFlags::LargeFile | FileFlags::NoBuffering ) )
    {
        _err = file.GetError();
        return false;
    }

    const ssize_t plotSize = file.Size();
    if( plotSize <= 0 )
    {
        if( plotSize < 0 )
            _err = file.GetError();
        else
            _err = -1;  // #TODO: Assign an actual user error.
        return false;
    }

    const size_t allocSize = RoundUpToNextBoundary( (size_t)plotSize, file.BlockSize() );

    byte* bytes = (byte*)SysHost::VirtualAlloc( allocSize );
    if( !bytes )
    {
        _err = -1;      // #TODO: Assign an actual user error.
        return false;
    }

    // Read the whole thing to memory
    size_t readSize = allocSize;
    byte*  reader   = bytes;
    
    while( readSize )
    {
        const ssize_t read = file.Read( reader, readSize );
        ASSERT( read );

        if( read < 0 )
        {
            _err = file.GetError();
            SysHost::VirtualFree( bytes );

            return false;
        }

        readSize -= (size_t)read;
        reader   += read;
    }

    // Read the header
    int headerError = 0;
    if( !ReadHeader( headerError ) )
    {
        if( headerError )
            _err = headerError;

        if( _err == 0 )
            _err = -1; // #TODO: Set generic plot header read error

        SysHost::VirtualFree( bytes );
        return false;
    }

    // Save data, good to go
    _bytes    = Span<byte>( bytes, (size_t)plotSize );
    _plotPath = path;

    return true;
}

//-----------------------------------------------------------
bool MemoryPlot::IsOpen()
{
    return _bytes.values != nullptr;
}

//-----------------------------------------------------------
size_t MemoryPlot::PlotSize() const
{
    return _bytes.length;
}

//-----------------------------------------------------------
bool MemoryPlot::Seek( SeekOrigin origin, int64 offset )
{
    ssize_t absPosition = 0;

    switch( origin )
    {
        case SeekOrigin::Begin:
            absPosition = offset;
            break;

        case SeekOrigin::Current:
            absPosition = _position + offset;
            break;

        case SeekOrigin::End:
            absPosition = (ssize_t)_bytes.length + offset;
            break;
    
        default:
            _err =  -1;     // #TODO: Set proper user error.
            return false;
    }

    if( absPosition < 0 || absPosition > (ssize_t)_bytes.length )
    {
        _err =  -1;     // #TODO: Set proper user error.
        return false;
    }

    _position = absPosition;
    return true;
}

//-----------------------------------------------------------
ssize_t MemoryPlot::Read( size_t size, void* buffer )
{
    if( size < 1 || !buffer )
        return 0;

    ASSERT( buffer );

    const size_t endPos = (size_t)_position + size;

    if( endPos > _bytes.length )
    {
        _err = -1; // #TODO: Set proper user error
        return false;
    }

    memcpy( buffer, _bytes.values + _position, size );
    _position = (ssize_t)endPos;

    return (ssize_t)size;
}

